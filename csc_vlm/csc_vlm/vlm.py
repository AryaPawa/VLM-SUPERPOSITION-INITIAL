"""
csc_vlm.vlm  --  one model handle for every pipeline-v2 stage
=============================================================

``VLMHandle`` owns the model, the prompt, the visual-span/grid detection, the
answer tokens, and a single ``run()`` that applies a compression plan
(``csc_vlm.compression``) inside the forward pass. Stage 0 (forward only),
Stage 1 (carrier-space Jacobians) and the preflight checks all go through it, so
the three can never disagree about what "pool at keep 0.25" means.

What changed from the p2 HFVLMBackend (and why)
-----------------------------------------------
* The hook goes on ONE module: the text backbone (the module that has both
  ``embed_tokens`` and ``layers``). We do not hook every candidate module.
* Reads come from a forward hook on decoder layer ``read_layer-1``. That output
  is exactly ``hidden_states[read_layer]`` and stays in the autograd graph, so
  Stage 1 can take dB/dh_read. We never call output_hidden_states=True (memory).
* The model loads directly in the requested dtype. The old path loaded fp16 and
  then upcast, which keeps fp16 rounding in the weights; Qwen2.5-VL is also known
  to overflow in fp16. "auto" picks bf16 on Ampere+ GPUs, fp16 on older GPUs and
  fp32 on CPU.
* Answer margin = logsumexp over answer-token variants (' cat', 'cat', 'Cat',
  ...), with whitespace-only lead tokens skipped. This fixes two problems: Llama
  SentencePiece tokenizers can encode ' cat' as ['_', '_cat'], and a model may
  answer with a capitalised word.
* There is NO silent fallback to "the first 576 positions". If the image token is
  not found in input_ids, we raise an error.
"""
from __future__ import annotations

import inspect
import math
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

from .compression import Plan, Run, build_plan, materialize, update_lm_kwargs

__all__ = ["VLMHandle", "resolve_dtype", "DEFAULT_IDS"]

DEFAULT_IDS = {
    "llava": "llava-hf/llava-1.5-7b-hf", "llava-7b": "llava-hf/llava-1.5-7b-hf",
    "llava-13b": "llava-hf/llava-1.5-13b-hf", "llava-next": "llava-hf/llava-v1.6-vicuna-7b-hf",
    "qwen": "Qwen/Qwen2.5-VL-7B-Instruct", "qwen-7b": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen-3b": "Qwen/Qwen2.5-VL-3B-Instruct",
    "internvl": "OpenGVLab/InternVL3-8B-hf", "internvl-8b": "OpenGVLab/InternVL3-8B-hf",
    "internvl-2b": "OpenGVLab/InternVL3-2B-hf",
    "smolvlm": "HuggingFaceTB/SmolVLM2-2.2B-Instruct", "smolvlm-2b": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
    "smolvlm-500m": "HuggingFaceTB/SmolVLM-500M-Instruct",
}

_IMG_TOKEN_ATTRS = ("image_token_id", "image_token_index", "img_context_token_id")
_IMG_TOKEN_STRS = ("<image>", "<|image_pad|>", "<IMG_CONTEXT>", "<image_soft_token>")


def resolve_dtype(name: str, device: str) -> torch.dtype:
    name = (name or "auto").lower()
    if name == "auto":
        if device.startswith("cuda") and torch.cuda.is_available():
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32
    return {"float32": torch.float32, "fp32": torch.float32, "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16, "float16": torch.float16, "fp16": torch.float16}[name]


def _find_text_model(model):
    """Return the module that owns embed_tokens + a ModuleList of decoder layers.
    If several match, take the one with the most layers (the LLM, not a tower)."""
    best = None
    for name, m in model.named_modules():
        layers = getattr(m, "layers", None)
        if hasattr(m, "embed_tokens") and isinstance(layers, torch.nn.ModuleList) and len(layers):
            if best is None or len(layers) > len(best[1].layers):
                best = (name, m)
    if best is None:
        raise RuntimeError("could not find the text backbone (module with embed_tokens + layers)")
    return best


class VLMHandle:
    """Model + prompt + span detection + compression-aware forward."""

    def __init__(self, model, processor=None, *, prepare_fn: Optional[Callable] = None,
                 tokenizer=None, read_layer_frac: float = 0.70, query_layer: int = 2,
                 image_token_id: Optional[int] = None, name: str = ""):
        self.model = model
        self.model.eval()
        self.model.requires_grad_(False)
        self.processor = processor
        self.tokenizer = tokenizer if tokenizer is not None else getattr(processor, "tokenizer", processor)
        self._prepare_fn = prepare_fn
        self.name = name or type(model).__name__
        self.text_name, self.text_model = _find_text_model(model)
        self.layers = self.text_model.layers
        self.n_layers = len(self.layers)
        self.hidden = int(self.text_model.embed_tokens.embedding_dim)
        self.dtype = self.text_model.embed_tokens.weight.dtype
        self.device = self.text_model.embed_tokens.weight.device
        self.read_layer = int(min(self.n_layers - 1, max(1, int(read_layer_frac * self.n_layers))))
        self.query_layer = int(min(self.n_layers - 1, max(1, query_layer)))
        self.image_token_id = image_token_id if image_token_id is not None else self._detect_image_token_id()
        sig = inspect.signature(self.model.forward).parameters
        self._logits_kw = ("logits_to_keep" if "logits_to_keep" in sig else
                           "num_logits_to_keep" if "num_logits_to_keep" in sig else None)

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_pretrained(cls, model: str, *, model_id: str = "", dtype: str = "auto",
                        device: str = "cuda", single_device: bool = False,
                        load_4bit: bool = False, read_layer_frac: float = 0.70,
                        query_layer: int = 2, attn_implementation: Optional[str] = None):
        import importlib
        from transformers import AutoProcessor
        mid = model_id or DEFAULT_IDS.get(model, model)
        if device.startswith("cuda") and not torch.cuda.is_available():
            print("[vlm][warn] CUDA not available; falling back to CPU (slow).")
            device = "cpu"
        td = resolve_dtype(dtype, device)
        processor = AutoProcessor.from_pretrained(mid, trust_remote_code=True)
        base = dict(trust_remote_code=True)
        if device.startswith("cuda"):
            base["device_map"] = {"": device if ":" in device else 0} if single_device else "auto"
        if attn_implementation:
            base["attn_implementation"] = attn_implementation
        if load_4bit:
            from transformers import BitsAndBytesConfig
            base["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=td)
        tfm = importlib.import_module("transformers")
        tried, mdl = [], None
        for cls_name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq", "AutoModel",
                         "AutoModelForCausalLM"):
            auto = getattr(tfm, cls_name, None)
            if auto is None:
                continue
            for dk in ("dtype", "torch_dtype"):
                kw = dict(base)
                if not load_4bit:
                    kw[dk] = td
                try:
                    mdl = auto.from_pretrained(mid, **kw)
                    break
                except TypeError as e:
                    if "dtype" in str(e).lower():
                        continue
                    tried.append(f"{cls_name}: {str(e)[:150]}")
                    break
                except Exception as e:  # noqa: BLE001
                    tried.append(f"{cls_name}: {type(e).__name__}: {str(e)[:150]}")
                    break
            if mdl is not None:
                break
        if mdl is None:
            raise RuntimeError(f"could not load {mid!r}:\n  " + "\n  ".join(tried))
        if not device.startswith("cuda"):
            mdl = mdl.to(device)
        h = cls(mdl, processor, read_layer_frac=read_layer_frac, query_layer=query_layer, name=mid)
        if td == torch.float16 and "qwen" in mid.lower():
            print("[vlm][warn] Qwen2.5-VL in float16 is prone to activation overflow; "
                  "prefer --dtype bfloat16 (Ampere+) or float32.")
        return h

    def device_layout(self) -> Dict:
        """Where things live. With device_map="auto" on several GPUs the embedding, the read
        layer and the LM head can sit on different devices; all code indexes a tensor on
        its own device, so this is informational."""
        def dev(mod):
            for p in mod.parameters():
                return str(p.device)
            return "none"
        out_emb = self.model.get_output_embeddings()
        layer_devs = [dev(l) for l in self.layers]
        return {"embed": str(self.device), "read_layer": layer_devs[self.read_layer - 1],
                "query_layer": layer_devs[self.query_layer - 1],
                "lm_head": dev(out_emb) if out_emb is not None else "none",
                "layer_devices": sorted(set(layer_devs)),
                "multi_device": len(set(layer_devs) | {str(self.device)}) > 1}

    def describe(self) -> Dict:
        return {"model": self.name, "text_backbone": self.text_name, "n_layers": self.n_layers,
                "hidden": self.hidden, "read_layer": self.read_layer,
                "query_layer": self.query_layer, "dtype": str(self.dtype),
                "image_token_id": self.image_token_id, "devices": self.device_layout()}

    # ------------------------------------------------------------------ prompts
    def prepare(self, image, prompt: str) -> Dict:
        if self._prepare_fn is not None:
            inp = self._prepare_fn(image, prompt)
        else:
            proc = self.processor
            try:
                msgs = [{"role": "user", "content": [{"type": "image"},
                                                     {"type": "text", "text": prompt}]}]
                text = proc.apply_chat_template(msgs, add_generation_prompt=True)
                inp = proc(images=image, text=text, return_tensors="pt")
            except Exception:
                inp = proc(images=image, text=f"USER: <image>\n{prompt} ASSISTANT:",
                           return_tensors="pt")
            inp = dict(inp)
        out = {}
        for k, v in inp.items():
            if torch.is_tensor(v):
                v = v.to(self.device)
                if v.is_floating_point():
                    v = v.to(self.dtype)
            out[k] = v
        return out

    def _detect_image_token_id(self) -> Optional[int]:
        cfg = self.model.config
        for a in _IMG_TOKEN_ATTRS:
            v = getattr(cfg, a, None)
            if isinstance(v, int):
                return v
        tok = self.tokenizer
        if tok is not None and hasattr(tok, "convert_tokens_to_ids"):
            unk = getattr(tok, "unk_token_id", None)
            for s in _IMG_TOKEN_STRS:
                try:
                    tid = tok.convert_tokens_to_ids(s)
                except Exception:
                    tid = None
                if isinstance(tid, int) and tid != unk:
                    return tid
        return None

    # ------------------------------------------------------------------ spans
    def visual_runs(self, inp: Dict) -> List[Run]:
        """Visual positions split into contiguous runs, each with its grid if known."""
        ids = inp["input_ids"][0].tolist()
        if self.image_token_id is None or self.image_token_id not in ids:
            raise RuntimeError(
                f"image token id {self.image_token_id} not found in input_ids for {self.name}. "
                "The processor did not expand image placeholders; update transformers, or pass "
                "image_token_id explicitly.")
        pos = [i for i, t in enumerate(ids) if t == self.image_token_id]
        runs: List[List[int]] = [[pos[0]]]
        for p in pos[1:]:
            if p == runs[-1][-1] + 1:
                runs[-1].append(p)
            else:
                runs.append([p])
        thw = inp.get("image_grid_thw", None)
        merge = int(getattr(getattr(self.model.config, "vision_config", None),
                            "spatial_merge_size", 2) or 2)
        out = []
        for i, r in enumerate(runs):
            grid = None
            if torch.is_tensor(thw) and thw.shape[0] == len(runs):
                t, h, w = [int(x) for x in thw[i].tolist()]
                H, W = h // merge, w // merge
                if t * H * W == len(r):
                    grid = (t, H, W)
            mtype = str(getattr(self.model.config, "model_type", "")).lower()
            multi_tile = any(t in mtype for t in ("internvl", "llava_next", "llava-next"))
            if grid is None and not multi_tile:
                # square runs: LLaVA-1.5 24x24, SmolVLM/Idefics3 8x8 tiles. Multi-tile models whose
                # tiles are concatenated in one run (InternVL, LLaVA-NeXT) fall back to 1-D pooling.
                s = int(round(math.sqrt(len(r))))
                if s * s == len(r) and s > 1:
                    grid = (1, s, s)
            out.append(Run(r, grid))
        return out

    # ------------------------------------------------------------------ answers
    def answer_ids(self, words: Sequence[str]) -> List[int]:
        """First non-whitespace token id of each case/space variant of each word."""
        tok = self.tokenizer
        ids = []
        for w in words:
            for v in (w, " " + w, w.capitalize(), " " + w.capitalize(), w.upper()):
                try:
                    enc = tok.encode(v, add_special_tokens=False)
                except TypeError:
                    enc = tok.encode(v)
                enc = list(enc)
                while enc and tok.decode([enc[0]]).strip() == "":
                    enc = enc[1:]
                if enc and enc[0] not in ids:
                    ids.append(int(enc[0]))
        return ids

    # ------------------------------------------------------------------ forward
    def run(self, inp: Dict, plan_fn: Optional[Callable[[torch.Tensor, List[Run]], Plan]] = None,
            *, make_leaf: bool = False, capture: Sequence[int] = (), grad: bool = False,
            runs: Optional[List[Run]] = None, carriers_override=None,
            span_delta: Optional[torch.Tensor] = None, grad_emb: bool = False) -> Dict:
        """One forward pass with an optional compression plan.

        plan_fn(emb_span [n_vis, D], runs) -> Plan  is called inside the text-backbone
        pre-hook. With make_leaf=True the carriers become a fresh leaf tensor
        (requires_grad). ``capture`` lists hidden_states indices to record
        (index i = output of decoder layer i-1, i.e. HF hidden_states[i]).
        Returns dict(logits [V] float32, hs {i: [T', D]} (views), hs_raw {i: [1, T', D]}
        (graph tensors, differentiate w.r.t. these), plan, leaf, T, new_len, runs).

        Attack leaves (pipeline v3, ``attack.py``):
          span_delta  [n_vis, D] perturbation added to the PRE-compression visual-token
                      embeddings (the fixed "patch" space, same size at every keep).
                      Carriers become Wc @ (emb_span + span_delta), so gradients reach
                      span_delta through the compression. Requires a plan_fn (use the
                      identity plan for keep = 1).
          grad_emb    keep the graph from inputs_embeds into the carriers (pixel leaf:
                      gradients flow pixels -> vision tower -> emb_span -> Wc -> LM).
          plan_fn receives the (possibly perturbed) embeddings DETACHED, so a live
          selection (drop_norm / drop_query) is recomputed on the perturbed input but is
          piecewise constant for autograd (BPDA / straight-through on the selection).
        """
        if span_delta is not None and (plan_fn is None or make_leaf or carriers_override is not None):
            raise ValueError("span_delta needs a plan_fn and excludes make_leaf / carriers_override")
        runs = runs if runs is not None else self.visual_runs(inp)
        span = [p for r in runs for p in r.positions]
        T_ids = int(inp["input_ids"].shape[1])
        state: Dict = {"done": False, "plan": None, "leaf": None}
        hs: Dict[int, torch.Tensor] = {}

        def pre_hook(module, args, kwargs):
            if state["done"]:
                return None
            emb = kwargs.get("inputs_embeds", None)
            loc = "kw"
            if emb is None:
                for j, a in enumerate(args):
                    if torch.is_tensor(a) and a.dim() == 3 and a.shape[-1] == self.hidden \
                            and a.is_floating_point():
                        emb, loc = a, j
                        break
            if emb is None:
                return None
            state["done"] = True
            if emb.shape[1] != T_ids:
                raise RuntimeError(f"inputs_embeds length {emb.shape[1]} != input_ids length {T_ids}; "
                                   "visual positions cannot be located")
            if plan_fn is None:
                return None
            emb_span = emb[0, span]
            if span_delta is not None:
                if tuple(span_delta.shape) != tuple(emb_span.shape):
                    raise ValueError(f"span_delta shape {tuple(span_delta.shape)} != visual span "
                                     f"{tuple(emb_span.shape)}")
                base = emb_span.detach() + span_delta.to(emb_span.device, emb_span.dtype)
            elif grad_emb:
                base = emb_span
            else:
                base = emb_span.detach()
            plan = plan_fn(base.detach(), runs)
            state["plan"] = plan
            passthrough = (not make_leaf and carriers_override is None and span_delta is None
                           and not plan.shortened and plan.mask_zero_rows.numel() == 0
                           and plan.method in ("none", "readout_only"))
            if passthrough:
                return None
            if carriers_override is not None:
                if tuple(carriers_override.shape) != (plan.n_carriers, emb_span.shape[-1]):
                    raise ValueError("carriers_override shape does not match the plan")
                carriers = carriers_override.detach().to(emb_span.device, emb_span.dtype)
            else:
                carriers = plan.Wc.to(base.dtype) @ base
            if make_leaf:
                carriers = carriers.detach().clone().requires_grad_(True)
            state["leaf"] = carriers
            new_emb = materialize(plan, emb, carriers)
            kwargs = update_lm_kwargs(plan, dict(kwargs))
            if loc == "kw":
                kwargs["inputs_embeds"] = new_emb
                return (args, kwargs)
            na = list(args)
            na[loc] = new_emb
            return (tuple(na), kwargs)

        def make_cap(i):
            def fwd_hook(module, args, output):
                hs[i] = output[0] if isinstance(output, (tuple, list)) else output
            return fwd_hook

        handles = [self.text_model.register_forward_pre_hook(pre_hook, with_kwargs=True)]
        for i in sorted(set(int(c) for c in capture)):
            if not (1 <= i <= self.n_layers):
                raise ValueError(f"capture index {i} out of range 1..{self.n_layers}")
            handles.append(self.layers[i - 1].register_forward_hook(make_cap(i)))
        extra = {"use_cache": False}
        if self._logits_kw:
            extra[self._logits_kw] = 1
        try:
            ctx = torch.enable_grad() if grad else torch.no_grad()
            with ctx:
                out = self.model(**inp, **extra)
        finally:
            for h in handles:
                h.remove()
        if not state["done"]:
            raise RuntimeError("the text-backbone pre-hook never fired (inputs_embeds not seen)")
        logits = out.logits[0, -1].float()
        hs_raw = dict(hs)        # the exact tensors in the graph (use these for autograd.grad)
        hs = {i: (v[0] if v.dim() == 3 else v) for i, v in hs.items()}
        return {"logits": logits, "hs": hs, "hs_raw": hs_raw, "plan": state["plan"],
                "leaf": state["leaf"],
                "T": T_ids, "new_len": (state["plan"].new_len if state["plan"] else T_ids),
                "runs": runs}

    # ------------------------------------------------------------------ helpers
    def query_scores(self, inp: Dict, runs: Optional[List[Run]] = None,
                     span_delta: Optional[torch.Tensor] = None) -> np.ndarray:
        """drop_query relevance: cos(h_q[visual], h_q[last]) at hidden_states[query_layer]
        from one uncompressed no-grad pass (optionally with a patch-space perturbation)."""
        runs = runs if runs is not None else self.visual_runs(inp)
        span = [p for r in runs for p in r.positions]
        if span_delta is not None:
            T = int(inp["input_ids"].shape[1])
            o = self.run(inp, lambda e, rr: build_plan("none", 1.0, T, rr, e),
                         capture=(self.query_layer,), runs=runs, span_delta=span_delta.detach())
        else:
            o = self.run(inp, None, capture=(self.query_layer,), runs=runs)
        h = o["hs"][self.query_layer].float()
        v, q = h[span], h[-1]
        s = torch.nn.functional.cosine_similarity(v, q[None, :], dim=-1)
        return s.cpu().numpy()

    def compressed_run(self, inp: Dict, method: str, keep: float, *, scores=None,
                       image_key: int = 0, seed: int = 0, make_leaf: bool = False,
                       capture: Sequence[int] = (), grad: bool = False,
                       carriers_override=None) -> Dict:
        """Convenience wrapper: build the plan for (method, keep) and run."""
        runs = self.visual_runs(inp)
        T = int(inp["input_ids"].shape[1])
        if method == "drop_query" and scores is None and keep < 1.0:
            scores = self.query_scores(inp, runs)

        def fn(emb_span, rr):
            return build_plan(method, keep, T, rr, emb_span, scores=scores,
                              image_key=image_key, seed=seed)
        use_plan = make_leaf or carriers_override is not None or not (method == "none" or keep >= 1.0)
        return self.run(inp, fn if use_plan else None, make_leaf=make_leaf,
                        capture=capture, grad=grad, runs=runs, carriers_override=carriers_override)
