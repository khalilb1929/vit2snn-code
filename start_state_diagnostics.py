#!/usr/bin/env python3
"""
start_state_diagnostics.py -- what does phase 1 change in the starting point of fine-tuning?
(no training; evaluation mode; monitoring split only)

For each seed of the series, compares two students that are the two possible starting points
of phase 2:
  A = teacher-initialised student (start of the nop1_* runs; identical for every seed)
  B = student after phase 1 (runs/series_v2/seed<s>/phase1/distill_latest.pt)
on the held-out monitoring split (eval transform), against the frozen teacher:
  1. BatchNorm: per layer, mean of gamma, beta, running_mean, running_var, and their change B-A;
  2. firing rates per LIF layer and per time step;
  3. hidden states per stage: RMS of the time-averaged student state, relative error rho_l
     (definition of the report), share of the error in the top-energy channel and in channel 41;
  4. 10-way and 1000-way top-1 of A and B on the monitoring split.
These are descriptive measurements of two checkpoints; they do not show which of the
differences causes the better fine-tuning outcome.

Usage:
    .venv/bin/python start_state_diagnostics.py --seeds 0 1 2            # full
    .venv/bin/python start_state_diagnostics.py --seeds 0 --max-images 32  # smoke test
Output: runs/series_v2/start_state_diagnostics.json
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


def bn_stats(model: torch.nn.Module) -> dict:
    out = {}
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.BatchNorm1d):
            out[name] = {"gamma_mean": float(mod.weight.mean()), "beta_mean": float(mod.bias.mean()),
                         "running_mean_mean": float(mod.running_mean.mean()),
                         "running_var_mean": float(mod.running_var.mean())}
    return out


@torch.no_grad()
def analyse(student, teacher, loader, dev, idx10) -> dict:
    lifs = {n: mm for n, mm in student.named_modules() if isinstance(mm, m.MultiStepLIF)}
    for mm in lifs.values():
        mm.reset_stats(); mm.record = True
    D, S = teacher.embed_dim, student.num_stages
    z = lambda: torch.zeros(D, dtype=torch.float64)  # noqa: E731
    acc = [dict(err=z(), sq=z(), sm=z(), hs2=0.0, n=0) for _ in range(S)]
    c1000 = c10 = n_img = 0
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        ht, hs = m.teacher_embed(teacher, x), student.encoder(x)
        for l in range(S):
            if l > 0:
                ht, hs = teacher.blocks[l - 1](ht), student.blocks[l - 1](hs)
            hm = hs.mean(0)
            r = acc[l]
            r["err"] += ((hm - ht) ** 2).sum((0, 1)).cpu().double()
            r["sq"] += (ht ** 2).sum((0, 1)).cpu().double()
            r["sm"] += ht.sum((0, 1)).cpu().double()
            r["hs2"] += float((hm ** 2).sum())
            r["n"] += ht.shape[0] * ht.shape[1]
        logits = student.forward_head(hs).mean(0)
        c1000 += (logits.argmax(-1) == y).sum().item()
        c10 += (idx10[logits[:, idx10].argmax(-1)] == y).sum().item()
        n_img += y.numel()
    for mm in lifs.values():
        mm.record = False
    rows = []
    for l, r in enumerate(acc):
        den = (r["sq"] - r["sm"] ** 2 / r["n"]).sum()
        top = int(torch.argmax(r["sq"]))
        rows.append({"stage": l, "rho": float(r["err"].sum() / den),
                     "student_rms": (r["hs2"] / (r["n"] * D)) ** 0.5,
                     "teacher_rms": float((r["sq"].sum() / (r["n"] * D)) ** 0.5),
                     "top_energy_channel": top, "err_share_top_channel": float(r["err"][top] / r["err"].sum()),
                     "err_share_ch41": float(r["err"][41] / r["err"].sum())})
    rates = {}
    for name, mm in lifs.items():
        if mm.spike_sum_t is not None:
            T = mm.spike_sum_t.numel()
            rates[name] = {"rate": float(mm.spike_sum_t.sum()) / (mm.neuron_count * T),
                           "rate_per_t": (mm.spike_sum_t / mm.neuron_count).tolist()}
        mm.reset_stats()
    return {"n_images": n_img, "top1_1000way": 100 * c1000 / n_img, "top1_10way": 100 * c10 / n_img,
            "hidden": rows, "firing": rates}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--series", default="runs/series_v2")
    p.add_argument("--split-file", default="splits/imagenette_train90_monitor10_seed0.json")
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    dev = m.get_device("auto")
    cfg = m.Config(data="imagefolder", train_dir="data/imagenette2-320/train",
                   val_dir="data/imagenette2-320/val", split_file=a.split_file,
                   num_workers=2, batch_size=32)
    teacher = timm.create_model(cfg.teacher_name, pretrained=True).to(dev).eval()
    monitor = m.build_loaders(cfg, teacher, dev).monitor   # held-out split, eval transform
    if a.max_images:
        monitor = DataLoader(Subset(monitor.dataset, list(range(a.max_images))), batch_size=32)
    idx10 = torch.tensor(sorted(m.IMAGENETTE_TO_IN1K.values()), device=dev)

    torch.manual_seed(0)
    A = m.SpikingViT(teacher, cfg).to(dev)
    m.init_student_from_teacher(A, teacher, cfg.bipolar_input)
    A.eval()
    res = {"split_file": a.split_file, "A_teacher_init": {"bn": bn_stats(A), **analyse(A, teacher, monitor, dev, idx10)}}
    print("A (teacher init): top1 10-way %.2f" % res["A_teacher_init"]["top1_10way"], flush=True)
    for s in a.seeds:
        ck = Path(a.series) / f"seed{s}" / "phase1" / "distill_latest.pt"
        B = m.SpikingViT(teacher, cfg).to(dev)
        B.load_state_dict(torch.load(ck, map_location=dev)["student"])
        B.eval()
        bn_b = bn_stats(B)
        bn_delta = {k: {f: bn_b[k][f] - res["A_teacher_init"]["bn"][k][f] for f in bn_b[k]} for k in bn_b}
        res[f"B_seed{s}"] = {"checkpoint": str(ck), "bn": bn_b, "bn_delta_vs_A": bn_delta,
                             **analyse(B, teacher, monitor, dev, idx10)}
        print(f"B seed{s}: top1 10-way %.2f" % res[f"B_seed{s}"]["top1_10way"], flush=True)
        del B
        m.free_memory(dev)
    out = a.out or str(Path(a.series) / "start_state_diagnostics.json")
    Path(out).write_text(json.dumps(res, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
