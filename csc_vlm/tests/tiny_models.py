"""
tests/tiny_models.py  --  tiny, randomly initialised REAL Hugging Face VLMs for CPU tests.

No downloads. Each builder returns a VLMHandle that wraps a genuine transformers
model class (LlavaForConditionalGeneration, Qwen2_5_VLForConditionalGeneration)
with a toy config, a fake tokenizer, and a prepare_fn that emits correctly shaped
input_ids / pixel_values. The hooks, plans, masks, position ids and gradients
then run through the real model code paths of the installed transformers.
"""
from __future__ import annotations

import numpy as np
import torch

VOCAB = 320
WORDS = {"cat": 11, " cat": 11, "Cat": 12, " Cat": 12, "CAT": 12,
         "dog": 13, " dog": 13, "Dog": 14, " Dog": 14, "DOG": 14,
         "yes": 15, " yes": 15, "Yes": 16, " Yes": 16, "YES": 16,
         "no": 17, " no": 17, "No": 18, " No": 18, "NO": 18}


class FakeTokenizer:
    unk_token_id = 0

    def encode(self, s, add_special_tokens=False):
        if s in WORDS:
            ids = [WORDS[s]]
            return ([3] + ids) if s.startswith(" ") else ids   # mimic SentencePiece '_' lead
        return [20 + (sum(map(ord, s)) % 50)]

    def decode(self, ids):
        inv = {11: "cat", 12: "Cat", 13: "dog", 14: "Dog", 15: "yes", 16: "Yes", 17: "no", 18: "No"}
        return " ".join(inv.get(int(i), " " if int(i) == 3 else f"t{int(i)}") for i in ids)

    def convert_tokens_to_ids(self, s):
        return self.unk_token_id


def _img_tensor(image, size):
    """Turn a PIL image / ndarray / int seed into a deterministic [3,size,size] tensor."""
    if isinstance(image, (int, np.integer)):
        g = torch.Generator().manual_seed(int(image))
        return torch.randn(3, size, size, generator=g)
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, -1)
    t = torch.from_numpy(arr).permute(2, 0, 1)[:3]
    t = torch.nn.functional.interpolate(t[None], size=(size, size), mode="bilinear",
                                        align_corners=False)[0]
    return (t - t.mean()) / (t.std() + 1e-6)


def tiny_llava(seed: int = 0, text_layers: int = 4):
    from transformers import CLIPVisionConfig, LlamaConfig, LlavaConfig, LlavaForConditionalGeneration
    from csc_vlm.vlm import VLMHandle
    torch.manual_seed(seed)
    IMG = 300
    vc = CLIPVisionConfig(image_size=32, patch_size=4, hidden_size=32, intermediate_size=64,
                          num_hidden_layers=2, num_attention_heads=2, projection_dim=32)
    tc = LlamaConfig(vocab_size=VOCAB, hidden_size=64, intermediate_size=128,
                     num_hidden_layers=text_layers, num_attention_heads=4, num_key_value_heads=4,
                     max_position_embeddings=512)
    cfg = LlavaConfig(vision_config=vc, text_config=tc, image_token_index=IMG,
                      vision_feature_layer=-2, vision_feature_select_strategy="default")
    cfg.image_token_id = IMG
    model = LlavaForConditionalGeneration(cfg).eval().float()
    n_img = (32 // 4) ** 2           # 64 visual tokens -> 8x8 grid

    def prepare(image, prompt):
        pre = [1, 40, 41]
        post = [42] + [50 + (ord(c) % 40) for c in prompt[:6]] + [43, 44]
        ids = pre + [IMG] * n_img + post
        return {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
                "pixel_values": _img_tensor(image, 32)[None]}
    return VLMHandle(model, None, prepare_fn=prepare, tokenizer=FakeTokenizer(),
                     image_token_id=IMG, name="tiny-llava")


def tiny_qwen25vl(seed: int = 0, text_layers: int = 4):
    from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
    from csc_vlm.vlm import VLMHandle
    torch.manual_seed(seed)
    IMG, VS, VE = 300, 301, 302
    text = dict(vocab_size=VOCAB, hidden_size=64, intermediate_size=128, num_hidden_layers=text_layers,
                num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024,
                rope_scaling={"type": "mrope", "mrope_section": [2, 3, 3]},
                rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                                 "mrope_section": [2, 3, 3]})
    vision = dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64,
                  patch_size=4, spatial_merge_size=2, temporal_patch_size=2, window_size=16,
                  fullatt_block_indexes=[1], in_channels=3)
    try:
        cfg = Qwen2_5_VLConfig(text_config=text, vision_config=vision, image_token_id=IMG,
                               vision_start_token_id=VS, vision_end_token_id=VE)
    except TypeError:
        cfg = Qwen2_5_VLConfig(**text, vision_config=vision, image_token_id=IMG,
                               vision_start_token_id=VS, vision_end_token_id=VE)
    model = Qwen2_5_VLForConditionalGeneration(cfg).eval().float()
    gt, gh, gw = 1, 12, 8                      # 96 patches -> 24 merged tokens (6x4 grid)
    n_img = gt * gh * gw // 4
    patch_dim = 3 * 2 * 4 * 4

    def prepare(image, prompt):
        ids = [1, 40, VS] + [IMG] * n_img + [VE, 42] + [50 + (ord(c) % 40) for c in prompt[:6]] + [43, 44]
        g = torch.Generator().manual_seed(int(image) if isinstance(image, (int, np.integer)) else 0)
        pv = torch.randn(gt * gh * gw, patch_dim, generator=g)
        return {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
                "pixel_values": pv, "image_grid_thw": torch.tensor([[gt, gh, gw]])}
    return VLMHandle(model, None, prepare_fn=prepare, tokenizer=FakeTokenizer(),
                     image_token_id=IMG, name="tiny-qwen2.5-vl")
