"""
csc_vlm.backends
================

The ONLY model-specific surface. A backend returns pooled read-layer activations
at a given visual-token keep ratio; everything downstream is model-agnostic.

* MockVLMBackend -- synthetic Variant-B geometry, CPU, no torch. Validates logic.
* HFVLMBackend   -- real LLaVA-1.5 / Qwen2.5-VL via transformers (lazy import).
                    Verified on Qwen2.5-VL-3B: model loads, hook fires, activations
                    extracted. Compression merges OR prunes ONLY the visual-token span.

Compression methods
-------------------
  "merge"  (default) : ToMe-style contiguous group-mean. Sequence length unchanged.
  "prune"            : Activation-Magnitude Pruning via Attention-Mask Zeroing.
                       Keeps the top-k highest-L2-norm tokens; zeros the attention
                       mask for the rest. Sequence length unchanged -> no shape crash.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from . import geometry as geo

__all__ = ["VLMConfig", "VLMBackend", "MockVLMBackend", "HFVLMBackend"]


@dataclass
class VLMConfig:
    # model / read position
    model: str = "mock"                 # "mock" | "llava" | "qwen"
    model_id: str = ""
    read_layer_frac: float = 0.70
    device: str = "cuda"
    load_4bit: bool = False
    # sampling
    n_pairs: int = 24                   # activations collected per safety evaluation
    # monitor / attack
    monitor_rank: int = 8
    beta: float = 1.0
    shrink: float = 0.2
    # compression knob (keep ratio)
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
    DEFAULTS = {"llava": "llava-hf/llava-1.5-7b-hf",
                "qwen": "Qwen/Qwen2.5-VL-7B-Instruct"}

    def __init__(self, cfg: VLMConfig, images: List, labels: np.ndarray):
        import torch
        from transformers import AutoProcessor
        self.cfg = cfg
        self.torch = torch
        mid = cfg.model_id or self.DEFAULTS[cfg.model]
        self.processor = AutoProcessor.from_pretrained(mid, trust_remote_code=True)
        self.model = self._load_model(mid)
        self.model.eval()
        self.images = images
        self.labels = np.asarray(labels)
        self._d = int(getattr(getattr(self.model.config, "text_config", self.model.config),
                              "hidden_size"))
        n_layers = getattr(self.model.config, "num_hidden_layers", None) or \
            self.model.config.text_config.num_hidden_layers
        self.read_layer = max(1, int(cfg.read_layer_frac * n_layers))
        # per-forward compression state (set in collect, read by the hook)
        self._merge_keep = 1.0
        self._merge_span = None
        self._merge_fired = False
        # attention mask reference for pruning (set in collect before each forward)
        self._current_attention_mask = None

    def _load_model(self, mid):
        import torch
        kw = dict(torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
        if self.cfg.load_4bit:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
            kw.pop("torch_dtype")
        try:
            from transformers import AutoModelForImageTextToText
            return AutoModelForImageTextToText.from_pretrained(mid, **kw)
        except Exception:
            from transformers import AutoModelForVision2Seq
            return AutoModelForVision2Seq.from_pretrained(mid, **kw)

    def read_dim(self) -> int:
        return self._d

    # -- prompt / visual span ------------------------------------------------ #
    def _prepare(self, image):
        try:
            messages = [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe the image."}]}]
            text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inp = self.processor(images=image, text=text, return_tensors="pt")
        except Exception:
            inp = self.processor(images=image,
                                 text="USER: <image>\nDescribe the image. ASSISTANT:",
                                 return_tensors="pt")
        dev = getattr(self.model, "device", self.cfg.device)
        return {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in inp.items()}

    def _visual_span(self, inp):
        ids = inp.get("input_ids")[0].tolist()
        img_id = getattr(self.model.config, "image_token_index", None) or \
            getattr(self.model.config, "image_token_id", None)
        if img_id is not None and img_id in ids:
            return [i for i, t in enumerate(ids) if t == img_id]
        return list(range(1, min(577, len(ids))))         # fallback: leading block

    def _pre_hook(self, module, args, kwargs):
        emb = kwargs.get("inputs_embeds")
        if emb is None or self._merge_keep >= 1.0 or self._merge_span is None:
            return None
        sp = self._merge_span
        if len(sp) <= 1:
            return None

        method = self.cfg.compression_method

        if method == "merge":
            # ---- Token Merging (existing behaviour) ----
            merged = emb.clone()
            merged[:, sp, :] = tome_merge_torch(emb[:, sp, :], self._merge_keep, self.torch)
            self._merge_fired = True
            kwargs["inputs_embeds"] = merged
            return (args, kwargs)

        elif method == "prune":
            # ---- Activation-Magnitude Pruning via Attention-Mask Zeroing ----
            keep_mask = tome_prune_mask(len(sp), self._merge_keep,
                                        emb[:, sp, :], self.torch)
            # Zero the attention mask for pruned positions (shape: [B, seq_len])
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

    def collect(self, keep_ratio, n, seed):
        torch = self.torch
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(self.images), size=min(n, len(self.images)),
                         replace=len(self.images) < n)
        lm = getattr(self.model, "language_model", self.model)
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
                H.append(h); y.append(int(self.labels[i]))
        finally:
            handle.remove()
        return np.asarray(H), np.asarray(y)
