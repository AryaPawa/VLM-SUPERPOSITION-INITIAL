#!/usr/bin/env python3
"""
build_person_dataset.py
=======================
Build the safety-facing "person" concept dataset from COCO val2017 as a standard
ImageFolder:

    data/concept/
        person/       images containing >= 1 person   (label 1)
        no_person/    images containing no person      (label 0)

Consumed by the harness with `--dataset folder:data/concept` or, equivalently,
`--dataset coco_person`.

One-command build (downloads COCO val2017 + annotations if missing, then sorts):

    python scripts/build_person_dataset.py --download --output data/concept

Manual (if you already have COCO on disk):

    python scripts/build_person_dataset.py \
        --coco-images coco_download/val2017 \
        --coco-ann    coco_download/annotations/instances_val2017.json \
        --output      data/concept --max-per-class 500

Notes
-----
* Files are COPIED by default (works everywhere, incl. Windows). Pass --symlink to
  symlink instead (saves disk on Linux; needs privileges on Windows).
* val2017 is 5,000 images (~1 GB) + annotations (~241 MB): enough for these runs.
"""
import argparse
import json
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

COCO_IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"
COCO_ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"


def _download(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  [skip] {dest.name} already present")
        return
    print(f"  downloading {url}")

    def _hook(blocks, bs, total):
        if total > 0:
            pct = min(100, blocks * bs * 100 // total)
            sys.stdout.write(f"\r    {pct:3d}%")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, dest, _hook)
    sys.stdout.write("\r    done\n")


def _unzip(zip_path: Path, out_dir: Path, marker: Path):
    if marker.exists():
        print(f"  [skip] already extracted ({marker})")
        return
    print(f"  extracting {zip_path.name}")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out_dir)


def download_coco(coco_root: Path):
    coco_root.mkdir(parents=True, exist_ok=True)
    img_zip = coco_root / "val2017.zip"
    ann_zip = coco_root / "annotations_trainval2017.zip"
    _download(COCO_IMAGES_URL, img_zip)
    _download(COCO_ANN_URL, ann_zip)
    _unzip(img_zip, coco_root, coco_root / "val2017")
    _unzip(ann_zip, coco_root, coco_root / "annotations")


def main():
    ap = argparse.ArgumentParser(description="Build person/no_person from COCO")
    ap.add_argument("--download", action="store_true",
                    help="fetch COCO val2017 + annotations into --coco-root if missing")
    ap.add_argument("--coco-root", default="coco_download",
                    help="where to download/find COCO (default: coco_download)")
    ap.add_argument("--coco-images", default="",
                    help="images dir (default: <coco-root>/val2017)")
    ap.add_argument("--coco-ann", default="",
                    help="instances json (default: <coco-root>/annotations/instances_val2017.json)")
    ap.add_argument("--output", default="data/concept",
                    help="output dir (gets person/ and no_person/ subfolders)")
    ap.add_argument("--max-per-class", type=int, default=500)
    ap.add_argument("--symlink", action="store_true",
                    help="symlink instead of copying (Linux; copy is the default)")
    args = ap.parse_args()

    coco_root = Path(args.coco_root)
    if args.download:
        print(f"[coco] preparing COCO under {coco_root} ...")
        download_coco(coco_root)

    coco_images = Path(args.coco_images) if args.coco_images else coco_root / "val2017"
    coco_ann = Path(args.coco_ann) if args.coco_ann \
        else coco_root / "annotations" / "instances_val2017.json"

    if not coco_ann.exists():
        raise FileNotFoundError(
            f"annotations not found at {coco_ann}. Run with --download, or pass "
            f"--coco-ann.")
    if not coco_images.is_dir():
        raise FileNotFoundError(
            f"image dir not found at {coco_images}. Run with --download, or pass "
            f"--coco-images.")

    # ---- load annotations ------------------------------------------------- #
    print(f"Loading annotations from {coco_ann} ...")
    with open(coco_ann) as f:
        coco = json.load(f)
    id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}
    all_image_ids = set(id_to_file.keys())
    print(f"  Total images in annotation file: {len(all_image_ids)}")

    # ---- person category id ---------------------------------------------- #
    person_cat_id = next((c["id"] for c in coco["categories"]
                          if c["name"] == "person"), None)
    if person_cat_id is None:
        raise ValueError("Could not find 'person' category in COCO annotations!")
    print(f"  Person category ID: {person_cat_id}")

    # ---- split by person presence ---------------------------------------- #
    person_ids = {a["image_id"] for a in coco["annotations"]
                  if a["category_id"] == person_cat_id}
    no_person_ids = all_image_ids - person_ids
    print(f"  Images with person:    {len(person_ids)}")
    print(f"  Images without person: {len(no_person_ids)}")

    n = min(len(person_ids), len(no_person_ids), args.max_per_class)
    print(f"  Using {n} images per class (balanced, capped at {args.max_per_class})")
    person_sel = sorted(person_ids)[:n]
    no_person_sel = sorted(no_person_ids)[:n]

    # ---- place files ------------------------------------------------------ #
    out = Path(args.output)
    person_dir = out / "person"
    no_person_dir = out / "no_person"
    person_dir.mkdir(parents=True, exist_ok=True)
    no_person_dir.mkdir(parents=True, exist_ok=True)

    def place(img_ids, dst_dir):
        placed = missing = 0
        for img_id in img_ids:
            src = coco_images / id_to_file[img_id]
            dst = dst_dir / id_to_file[img_id]
            if not src.exists():
                missing += 1
                continue
            if not dst.exists():
                if args.symlink:
                    os.symlink(src.resolve(), dst)
                else:
                    shutil.copy2(src, dst)
            placed += 1
        return placed, missing

    p1, m1 = place(person_sel, person_dir)
    p0, m0 = place(no_person_sel, no_person_dir)

    print("\nDone!")
    print(f"  Placed: {p1 + p0} images  (missing on disk: {m1 + m0})")
    print(f"  person/    : {len(list(person_dir.iterdir()))} images")
    print(f"  no_person/ : {len(list(no_person_dir.iterdir()))} images")
    print(f"\nOutput: {out}")
    print(f"Use with:  --dataset folder:{out}   (or  --dataset coco_person"
          f"{'' if str(out) == 'data/concept' else ':' + str(out)})")


if __name__ == "__main__":
    main()
