#!/usr/bin/env python3
"""
prediction_support.py -- where do the 1000-way top-1 predictions fall? (no training)

For the teacher and for each final student checkpoint of the multi-seed series, on the official
Imagenette validation set: the fraction of images whose 1000-way top-1 prediction lies inside
the 10 Imagenette classes, and the 1000-way / 10-way top-1 accuracies (cross-check with
metrics.json). Output: runs/prediction_support.json

Usage: .venv/bin/python prediction_support.py [--max-images 64]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import timm
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vit2snn_distill as m  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--series", default="runs/series_v2")
    p.add_argument("--out", default="runs/prediction_support.json")
    a = p.parse_args()
    dev = m.get_device("auto")
    cfg = m.Config(data="imagefolder", train_dir="data/imagenette2-320/train",
                   val_dir="data/imagenette2-320/val", num_workers=2, batch_size=32)
    teacher = timm.create_model(cfg.teacher_name, pretrained=True).to(dev).eval()
    test = m.build_loaders(cfg, teacher, dev).test
    if a.max_images:
        ds = test.dataset
        test = DataLoader(Subset(ds, list(range(0, len(ds), max(1, len(ds) // a.max_images)))[:a.max_images]),
                          batch_size=32)
    idx = torch.tensor(sorted(m.IMAGENETTE_TO_IN1K.values()), device=dev)

    @torch.no_grad()
    def support(fn) -> dict:
        inside = c1000 = c10 = n = 0
        for x, y in test:
            x, y = x.to(dev), y.to(dev)
            z = fn(x)
            top = z.argmax(-1)
            inside += torch.isin(top, idx).sum().item()
            c1000 += (top == y).sum().item()
            c10 += (idx[z[:, idx].argmax(-1)] == y).sum().item()
            n += y.numel()
        return {"n": n, "top1_inside_10_classes": inside, "frac_inside": inside / n,
                "correct_1000way": c1000, "correct_10way": c10}

    res = {"teacher": support(lambda x: teacher(x))}
    print("teacher", res["teacher"], flush=True)
    for d in sorted(Path(a.series).glob("seed*/*/student_final.pt")):
        if d.parent.name == "phase1":
            continue
        s = m.SpikingViT(teacher, cfg).to(dev)
        s.load_state_dict(torch.load(d, map_location=dev)["student"])
        s.eval()
        key = f"{d.parent.parent.name}/{d.parent.name}"
        res[key] = support(lambda x: s(x).mean(0))
        print(key, res[key], flush=True)
        del s
        m.free_memory(dev)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
