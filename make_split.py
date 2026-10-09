#!/usr/bin/env python3
"""
make_split.py -- reproducible stratified train/monitor split of an ImageFolder training set.

For each class (in ImageFolder order), the image indices are shuffled with random.Random(seed)
and the first round(fraction * n_class) go to the monitor split. The JSON stores the indices,
the per-class counts and a SHA-256 of the sample list (paths relative to the folder), which
vit2snn_distill.py checks before using the split.

Usage:
    .venv/bin/python make_split.py --train-dir data/imagenette2-320/train \
        --fraction 0.1 --seed 0 --out splits/imagenette_train90_monitor10_seed0.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torchvision

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vit2snn_distill import samples_sha256  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-dir", required=True)
    p.add_argument("--fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    ds = torchvision.datasets.ImageFolder(a.train_dir)
    by_class: dict[int, list[int]] = {}
    for i, (_, y) in enumerate(ds.samples):
        by_class.setdefault(y, []).append(i)
    rng = random.Random(a.seed)
    train_idx, monitor_idx, per_class = [], [], {}
    for y in sorted(by_class):
        idx = list(by_class[y])
        rng.shuffle(idx)
        k = round(a.fraction * len(idx))
        monitor_idx += idx[:k]
        train_idx += idx[k:]
        per_class[ds.classes[y]] = {"train": len(idx) - k, "monitor": k}

    out = Path(a.out)
    if out.exists():
        sys.exit(f"{out} already exists; refusing to overwrite a split that runs may depend on")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "train_dir": a.train_dir, "fraction": a.fraction, "seed": a.seed,
        "n_total": len(ds), "samples_sha256": samples_sha256(ds), "classes": ds.classes,
        "per_class": per_class, "n_train": len(train_idx), "n_monitor": len(monitor_idx),
        "train_idx": sorted(train_idx), "monitor_idx": sorted(monitor_idx)}, indent=1))
    print(f"wrote {out}: {len(train_idx)} train / {len(monitor_idx)} monitor")
    for c, n in per_class.items():
        print(f"  {c}: {n['train']} / {n['monitor']}")


if __name__ == "__main__":
    main()
