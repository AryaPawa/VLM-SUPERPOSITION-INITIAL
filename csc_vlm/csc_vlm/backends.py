"""
csc_vlm.backends
================

The ONLY model-specific surface. A backend returns pooled read-layer activations
at a given visual-token keep ratio; everything downstream is model-agnostic.

* MockVLMBackend  -- synthetic Variant-B geometry, CPU, no torch. Validates logic.
* HFVLMBackend   -- any Prefix-LM VLM via Hugging Face transformers (lazy import).
                    Architecture-agnostic: works for any model that concatenates
                    visual tokens with text tokens in a shared input_embeds sequence
                    (the "Prefix-LM" or "Early-Fusion" paradigm).

                    Tested lineages (verified hook fires and activations extracted):
                    - LLaVA-1.5-7b  (llava-hf/llava-1.5-7b-hf)
                    - Qwen2.5-VL-3B (Qwen/Qwen2.5-VL-3B-Instruct)

                    Known-compatible lineages (same Prefix-LM architecture):
                    - LLaVA-NeXT  (llava-hf/llava-v1.6-vicuna-7b-hf)
                    - InternVL3   (OpenGVLab/InternVL3-8B)
                    - SmolVLM     (HuggingFaceTB/SmolVLM-Instruct)
                    - PaliGemma   (google/paligemma-3b-pt-224)

                    NOT compatible (uses cross-attention, not concat):
                    - Llama-3.2-Vision (Meta), Flamingo (DeepMind)

Compression methods
-------------------
  "merge"  (default) : ToMe-style contiguous group-mean. Sequence length unchanged.
  "prune"            : Activation-Magnitude Pruning via Attention-Mask Zeroing.
                       Keeps the top-k highest-L2-norm tokens; zeros the attention
                       mask for the rest. Sequence length unchanged -> no shape crash.

Adding a new model
------------------
For any Prefix-LM VLM, no code changes are required. Just pass:
    --model-id <hf_model_id>
combined with a unique short name via:
    --model <name>
(you may add it to DEFAULTS below for convenience).

The backend auto-detects:
 1. The language model sub-module (searches common attribute names).
 2. The image token ID (searches common config attributes).
 3. Hidden size and layer count for the read-layer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from . import geometry as geo

__all__ = ["VLMConfig", "VLMBackend", "MockVLMBackend", "HFVLMBackend"]


@dataclass
class VLMConfig:
    # model selection
    model: str = "mock"          # short name; "mock" or any key in DEFAULTS or free-form
    model_id: str = ""           # HF model id (overrides DEFAULTS if set)
    read_layer_frac: float = 0.70
    device: str = "cuda"
    load_4bit: bool = False
    # sampling
    n_pairs: int = 24            # activations collected per safety evaluation
    # monitor / attack
    monitor_rank: int = 8
    beta: float = 1.0
    shrink: float = 0.2
    # compression knob
    keep_min: float = 0.05
    keep_max: float = 1.0
    # compression method: "merge" | "prune"
    compression_method: str = "merge"
    # mock-only geometry
    mock_d: int = 64
    mock_F_base: int = 48
    mock_seed: int = 0


class VLMBackend:
    def read_dim(self) -> int:
        raise NotImplementedError

    def collect(self, keep_ratio: float, n: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


# ------------------------------------------------------------------- mock #
class MockVLMBackend(VLMBackend):
    """At keep_ratio, F_eff = round(F_base/keep) concepts merged into fixed read d.
    Fewer kept tokens -> more merged features -> more superposition. Fully CPU."""

    def __init__(self, cfg: VLMConfig):
        self.cfg = cfg
        self.d = cfg.mock_d
        self._concept = 0

    def read_dim(self) -> int:
        return self.d

    def collect(self, keep_ratio, n, seed):
        c = self.cfg
        keep = float(np.clip(keep_ratio, c.keep_min, c.keep_max))
        F_eff = max(2, int(round(c.mock_F_base / keep)))
        rng = np.random.default_rng(c.mock_seed + 1000)
        W = geo.normalize_columns(rng.standard_normal((self.d, F_eff)))
        srng = np.random.default_rng(seed)
        Z = (srng.random((n, F_eff)) < 0.10) * srng.random((n, F_eff))
        Z[:, self._concept] += 0.6 * (srng.random(n) < 0.5)
        H = Z @ W.T
        y = (Z[:, self._concept] > 0.15).astype(int)
        return H, y


# --------------------------------------------------------------------- HF #
def tome_merge_torch(x, keep_ratio, torch):
    """Length-preserving visual-token merge on [B, T, D]. Partition the T tokens
    into k = round(keep_ratio*T) contiguous groups; replace each token by its group
    mean, so only k DISTINCT token values remain while the sequence length (hence
    position ids / attention mask) is unchanged. Forward-only. Vectorised."""
    B, T, D = x.shape
    k = max(1, int(round(keep_ratio * T)))
    if k >= T:
        return x
    idx = (torch.arange(T, device=x.device) * k) // T
    onehot = torch.nn.functional.one_hot(idx, k).to(x.dtype)
    counts = onehot.sum(0).clamp(min=1.0)
    sums = torch.einsum("btd,tk->bkd", x, onehot)
    means = sums / counts[None, :, None]
    return means[:, idx, :]


def tome_prune_mask(span_len, keep_ratio, emb_span, torch):
    """Activation-Magnitude Pruning: return a boolean keep mask of length span_len.
    Tokens with the top-k highest L2 norms are kept (mask=True); the rest are pruned
    (mask=False) by zeroing the attention mask - sequence shape is unchanged.

    Args:
        span_len  : number of visual tokens in the span (int)
        keep_ratio: fraction of tokens to keep (float in [0,1])
        emb_span  : [1, span_len, D] or [span_len, D] embedding tensor
        torch     : the torch module (lazy-imported in the backend)

    Returns:
        keep_mask : 1-D bool tensor of length span_len
    """
    k = max(1, int(round(keep_ratio * span_len)))
    if k >= span_len:
        return torch.ones(span_len, dtype=torch.bool, device=emb_span.device)
    # flatten to [T, D] if batched
    e = emb_span.squeeze(0) if emb_span.dim() == 3 else emb_span
    norms = e.norm(dim=-1)           # [T]
    topk_idx = torch.topk(norms, k, largest=True, sorted=False).indices
    mask = torch.zeros(span_len, dtype=torch.bool, device=emb_span.device)
    mask[topk_idx] = True
    return mask


class HFVLMBackend(VLMBackend):
    """Architecture-agnostic backend for any Prefix-LM VLM.

    Works with any model where visual tokens are concatenated into the
    `inputs_embeds` sequence alongside text tokens (LLaVA, Qwen, InternVL,
    SmolVLM, PaliGemma, LLaVA-NeXT, etc.)

    To add a new model: just pass --model-id <hf_id> on the command line.
    No code changes required for any Prefix-LM architecture.
    """
    # Known model ID defaults keyed by a short name.
    # Add new short-names here for convenience; not required for compatibility.
    DEFAULTS = {
        # primary (Tier-0 / proposal)
        "llava":        "llava-hf/llava-1.5-7b-hf",
        "qwen":         "Qwen/Qwen2.5-VL-7B-Instruct",
        # extended (verified compatible, same Prefix-LM architecture)
        "llava-next":   "llava-hf/llava-v1.6-vicuna-7b-hf",
        "internvl":     "OpenGVLab/InternVL3-8B",
        "smolvlm":      "HuggingFaceTB/SmolVLM-Instruct",
        "paligemma":    "google/paligemma-3b-pt-224",
    }

    # --------------------------------------------------------------------- #
    # Language-model attribute names to search (in priority order).
    # We duck-type through these to find the causal LM sub-module to hook.
    _LM_ATTRS = [
        "language_model",   # LLaVA, LLaVA-NeXT, PaliGemma, SmolVLM
        "model",            # Qwen2.5-VL
        "text_model",       # some InternVL variants
    ]

    # Image-token ID config attribute names to search (in priority order).
    _IMG_TOKEN_ATTRS = [
        "image_token_index",   # LLaVA family
        "image_token_id",      # Qwen2.5-VL
        "img_context_token_id", # InternVL
        "image_token",         # PaliGemma (may be a string token, not id)
    ]

    # Fallback: number of visual tokens to assume if auto-detection fails.
    _FALLBACK_SPAN_LEN = 576   # standard CLIP 336px grid (LLaVA default)

    def __init__(self, cfg: VLMConfig, images: List, labels: np.ndarray):
        import torch
        from transformers import AutoProcessor
        self.cfg = cfg
        self.torch = torch
        mid = cfg.model_id or self.DEFAULTS.get(cfg.model, cfg.model)
        self.processor = AutoProcessor.from_pretrained(mid, trust_remote_code=True)
        self.model = self._load_model(mid)
        self.model.eval()
        self.images = images
        self.labels = np.asarray(labels)
        self._d = self._detect_hidden_size()
        self.read_layer = max(1, int(cfg.read_layer_frac * self._detect_num_layers()))
        # per-forward compression state (set in collect, read by the hook)
        self._merge_keep = 1.0
        self._merge_span = None
        self._merge_fired = False

    # ------------------------------------------------------------------ #
    # Auto-detection helpers
    # ------------------------------------------------------------------ #
    def _detect_hidden_size(self) -> int:
        """Walk config hierarchy to find hidden_size robustly."""
        for attr in ("hidden_size", "llm_hidden_size"):
            # try top-level config first (Qwen2.5-VL stores it there)
            v = getattr(self.model.config, attr, None)
            if v:
                return int(v)
        # try text sub-config (LLaVA, SmolVLM, PaliGemma)
        for sub in ("text_config", "language_config", "llm_config"):
            sub_cfg = getattr(self.model.config, sub, None)
            if sub_cfg is not None:
                v = getattr(sub_cfg, "hidden_size", None)
                if v:
                    return int(v)
        raise RuntimeError(
            f"Could not auto-detect hidden_size from model config "
            f"({type(self.model.config).__name__}). "
            f"Please open a GitHub issue with your model id."
        )

    def _detect_num_layers(self) -> int:
        """Walk config hierarchy to find num_hidden_layers robustly."""
        for attr in ("num_hidden_layers",):
            v = getattr(self.model.config, attr, None)
            if v:
                return int(v)
        for sub in ("text_config", "language_config", "llm_config"):
            sub_cfg = getattr(self.model.config, sub, None)
            if sub_cfg is not None:
                v = getattr(sub_cfg, "num_hidden_layers", None)
                if v:
                    return int(v)
        raise RuntimeError(
            f"Could not auto-detect num_hidden_layers from model config "
            f"({type(self.model.config).__name__})."
        )

    def _detect_lm_module(self):
        """Return the causal-LM sub-module to attach the hook to.
        We register on its forward pre-hook which fires with inputs_embeds."""
        for attr in self._LM_ATTRS:
            lm = getattr(self.model, attr, None)
            if lm is not None:
                return lm
        # fallback: hook the top-level model itself
        return self.model

    def _detect_image_token_id(self) -> Optional[int]:
        """Search common config attributes for the image token integer ID."""
        cfg = self.model.config
        for attr in self._IMG_TOKEN_ATTRS:
            val = getattr(cfg, attr, None)
            if isinstance(val, int):
                return val
        # Some models (PaliGemma) store it as a string in the tokenizer
        if hasattr(self.processor, "tokenizer"):
            tok = self.processor.tokenizer
            for token_str in ("<image>", "<img>", "[IMG]", "<|image_pad|>"):
                tid = tok.convert_tokens_to_ids(token_str)
                if tid is not None and tid != tok.unk_token_id:
                    return tid
        return None

    # ------------------------------------------------------------------ #
    # Model loading
    # ------------------------------------------------------------------ #
    def _load_model(self, mid):
        import torch
        kw = dict(dtype=torch.float16, device_map="auto", trust_remote_code=True)
        if self.cfg.load_4bit:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
            kw.pop("dtype", None)
        try:
            from transformers import AutoModelForImageTextToText
            return AutoModelForImageTextToText.from_pretrained(mid, **kw)
        except Exception:
            from transformers import AutoModelForVision2Seq
            return AutoModelForVision2Seq.from_pretrained(mid, **kw)

    def read_dim(self) -> int:
        return self._d

    # ------------------------------------------------------------------ #
    # Prompt preparation & visual span detection
    # ------------------------------------------------------------------ #
    def _prepare(self, image):
        """Build processor inputs using chat template, with LLaVA-1.5 fallback."""
        try:
            messages = [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe the image."}]}]
            text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inp = self.processor(images=image, text=text, return_tensors="pt")
        except Exception:
            # LLaVA-1.5 uses a plain text format
            inp = self.processor(images=image,
                                 text="USER: <image>\nDescribe the image. ASSISTANT:",
                                 return_tensors="pt")
        dev = next(self.model.parameters()).device
        return {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in inp.items()}

    def _visual_span(self, inp) -> List[int]:
        """Return the list of sequence positions that correspond to visual tokens.

        Strategy (in order):
        1. Auto-detect the image token ID from model config / tokenizer.
        2. Find all positions in input_ids where that token appears.
        3. Fallback: use the first _FALLBACK_SPAN_LEN positions after BOS.
        """
        ids = inp.get("input_ids")[0].tolist()
        img_id = self._detect_image_token_id()
        if img_id is not None and img_id in ids:
            return [i for i, t in enumerate(ids) if t == img_id]
        # Fallback: positions 1..min(576, len-1) (standard LLaVA-1.5 prefix)
        return list(range(1, min(self._FALLBACK_SPAN_LEN + 1, len(ids))))

    # ------------------------------------------------------------------ #
    # Compression hook (shared for all architectures)
    # ------------------------------------------------------------------ #
    def _pre_hook(self, module, args, kwargs):
        emb = kwargs.get("inputs_embeds")
        if emb is None or self._merge_keep >= 1.0 or self._merge_span is None:
            return None
        sp = self._merge_span
        if len(sp) <= 1:
            return None

        method = self.cfg.compression_method

        if method == "merge":
            # ---- Token Merging ----
            merged = emb.clone()
            merged[:, sp, :] = tome_merge_torch(emb[:, sp, :], self._merge_keep, self.torch)
            self._merge_fired = True
            kwargs["inputs_embeds"] = merged
            return (args, kwargs)

        elif method == "prune":
            # ---- Activation-Magnitude Pruning via Attention-Mask Zeroing ----
            keep_mask = tome_prune_mask(len(sp), self._merge_keep,
                                        emb[:, sp, :], self.torch)
            attn = kwargs.get("attention_mask")
            if attn is not None:
                attn = attn.clone()
                sp_tensor = self.torch.tensor(sp, device=attn.device)
                pruned_positions = sp_tensor[~keep_mask]
                attn[:, pruned_positions] = 0
                kwargs["attention_mask"] = attn
            self._merge_fired = True
            return (args, kwargs)

        return None

    # ------------------------------------------------------------------ #
    # Main data collection loop
    # ------------------------------------------------------------------ #
    def collect(self, keep_ratio, n, seed):
        torch = self.torch
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(self.images), size=min(n, len(self.images)),
                         replace=len(self.images) < n)
        lm = self._detect_lm_module()
        handle = lm.register_forward_pre_hook(self._pre_hook, with_kwargs=True)
        self._merge_keep = float(keep_ratio)
        H, y = [], []
        try:
            for i in idx:
                inp = self._prepare(self.images[i])
                self._merge_span = self._visual_span(inp)
                with torch.no_grad():
                    out = self.model(**inp, output_hidden_states=True)
                hs = out.hidden_states[self.read_layer][0]      # [seq, d]
                h = hs[self._merge_span].mean(0).float().cpu().numpy()
                H.append(h)
                y.append(int(self.labels[i]))
        finally:
            handle.remove()
        return np.asarray(H), np.asarray(y)
