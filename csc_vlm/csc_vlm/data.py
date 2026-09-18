"""
csc_vlm.data
============

Turns a dataset spec into (images: list[PIL.Image], labels: np.ndarray[int in {0,1}]).

Specs
-----
  synthetic                         built-in generator (plumbing only, no download)
  folder:/path/to/dir               dir/<classA>/*.jpg, dir/<classB>/*.jpg
  coco_person[:/path]               safety-facing person vs no_person, built from
                                     COCO by scripts/build_person_dataset.py
                                     (defaults to data/concept)
  hf:<name>[:col=..,label=..,pos=..] HuggingFace image-classification dataset,
                                     auto-downloaded and cached. Binarised.

Default real dataset: hf:microsoft/cats_vs_dogs (a clean, benign binary concept:
"is there a dog"). Swap in COCO/POPE person-presence folders for the safety-facing
concept once the pipeline is confirmed -- e.g. folder:data/concept with
no_person/ and person/ subfolders.
"""
from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np

__all__ = ["load_images", "make_synthetic", "DEFAULT_DATASET"]

DEFAULT_DATASET = "hf:microsoft/cats_vs_dogs"


def make_synthetic(n: int = 64, size: int = 448, seed: int = 0) -> Tuple[List, np.ndarray]:
    """Two visually-distinct classes on noisy gray (red block top-left = 0,
    green block bottom-right = 1). Plumbing validation only."""
    from PIL import Image
    rng = np.random.default_rng(seed)
    imgs, labels = [], []
    s = size // 2
    for i in range(n):
        arr = rng.integers(90, 130, (size, size, 3)).astype("uint8")
        lab = i % 2
        if lab == 0:
            arr[10:10 + s, 10:10 + s] = [220, 40, 40]
        else:
            arr[size - s - 10:size - 10, size - s - 10:size - 10] = [40, 200, 60]
        imgs.append(Image.fromarray(arr, "RGB")); labels.append(lab)
    return imgs, np.asarray(labels)


def _load_folder(path: str) -> Tuple[List, np.ndarray]:
    from PIL import Image
    classes = sorted(d for d in os.listdir(path)
                     if os.path.isdir(os.path.join(path, d)))
    if len(classes) < 2:
        raise ValueError(f"folder dataset needs >=2 class subfolders, found {classes}")
    images, labels = [], []
    for ci, cls in enumerate(classes):
        for fn in os.listdir(os.path.join(path, cls)):
            if fn.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")):
                try:
                    im = Image.open(os.path.join(path, cls, fn)).convert("RGB")
                    images.append(im); labels.append(0 if ci == 0 else 1)
                except Exception:
                    continue
    return images, np.asarray(labels)


def _load_hf(name: str, per_class: int = 200, split: str = "train",
             image_col: str = None, label_col: str = None,
             pos_labels=None, seed: int = 0) -> Tuple[List, np.ndarray]:
    from datasets import load_dataset
    ds = load_dataset(name, split=split)
    cols = ds.column_names
    icol = image_col or ("image" if "image" in cols else ("img" if "img" in cols else cols[0]))
    lcol = label_col or ("label" if "label" in cols else ("labels" if "labels" in cols else cols[-1]))
    labels = np.asarray(ds[lcol])
    uniq = sorted(set(labels.tolist()))
    if pos_labels is not None:
        pos = set(int(x) for x in pos_labels)
        ybin = np.isin(labels, list(pos)).astype(int)
    else:                                    # binary default: first label -> 0, rest -> 1
        ybin = (labels != uniq[0]).astype(int)
    rng = np.random.default_rng(seed)
    idx0 = np.where(ybin == 0)[0]
    idx1 = np.where(ybin == 1)[0]
    sel0 = rng.choice(idx0, size=min(per_class, len(idx0)), replace=False)
    sel1 = rng.choice(idx1, size=min(per_class, len(idx1)), replace=False)
    sel = np.concatenate([sel0, sel1])
    rng.shuffle(sel)
    images, y = [], []
    for i in sel:
        try:
            im = ds[int(i)][icol].convert("RGB")
            images.append(im); y.append(int(ybin[i]))
        except Exception:
            continue                          # skip undecodable images
    return images, np.asarray(y)


def load_images(spec: str, per_class: int = 200, smoke: bool = False,
                seed: int = 0) -> Tuple[List, np.ndarray]:
    """Resolve a dataset spec to (images, labels)."""
    spec = (spec or "").strip()
    if spec == "" or spec == "synthetic":
        return make_synthetic(n=24 if smoke else 64, seed=seed)
    if spec.startswith("folder:"):
        return _load_folder(spec[len("folder:"):])
    if spec.startswith("coco_person"):
        # Safety-facing concept: person vs no_person ImageFolder built from COCO
        # by scripts/build_person_dataset.py. `coco_person` -> data/concept;
        # `coco_person:/path` -> that path. Balanced-subsampled to per_class so
        # --per-class behaves the same as the hf path.
        parts = spec.split(":")
        root = parts[1] if len(parts) > 1 and parts[1] else "data/concept"
        if not os.path.isdir(root):
            raise ValueError(
                f"coco_person dataset not found at '{root}'. Build it first:\n"
                f"  python scripts/build_person_dataset.py --download --output {root}\n"
                f"then re-run (or point --dataset coco_person:/your/path at a build).")
        imgs, y = _load_folder(root)             # sorted classes: no_person=0, person=1
        pc = 24 if smoke else per_class
        rng = np.random.default_rng(seed)
        sel = []
        for cls in (0, 1):
            idx = np.where(y == cls)[0]
            if len(idx) == 0:
                raise ValueError(f"coco_person: class {cls} has no images under '{root}'")
            sel.append(rng.choice(idx, size=min(pc, len(idx)), replace=False))
        sel = np.concatenate(sel)
        rng.shuffle(sel)
        return [imgs[int(i)] for i in sel], y[sel]
    if spec.startswith("hf:"):
        rest = spec[len("hf:"):]
        parts = rest.split(":")
        name = parts[0]
        kw = {}
        for kv in parts[1:]:
            if "=" in kv:
                k, v = kv.split("=", 1)
                if k == "pos":
                    kw["pos_labels"] = [int(x) for x in v.split(",")]
                elif k == "col":
                    kw["image_col"] = v
                elif k == "label":
                    kw["label_col"] = v
                elif k == "split":
                    kw["split"] = v
        pc = 24 if smoke else per_class
        return _load_hf(name, per_class=pc, seed=seed, **kw)
    # bare path -> treat as folder
    if os.path.isdir(spec):
        return _load_folder(spec)
    raise ValueError(f"unrecognised dataset spec: {spec!r}")
