#!/usr/bin/env python3
"""
step1_diagnostics.py -- Step-1 diagnostics on EXISTING checkpoints (no training).

A. Inference-time T sweep (T in {1,2,4,8}) for the trained students: top-1 (1000-way and
   10-way) and global firing rate. The students were trained with T=4; other values are
   evaluated without retraining (BatchNorm statistics were collected at T=4).
B. Relative hidden-state error rho_l per stage (all tokens, and CLS token only) for each
   student against the frozen teacher, on a fixed validation subset (every 15th image).
   rho_l = sum_{i,c}(mean_t hs - ht)^2 / sum_{i,c}(ht - mu_c)^2  (1 = per-channel-mean predictor)
C. Teacher-only channel interventions at the output of given stages (default 8 and 12):
   replace the top-energy channel by its calibration mean, clip it, or add norm-matched
   noise to the other channels; controls = second-energy channel and random channels.
   Calibration statistics (means, clip threshold, energy ranking) come from a TRAINING subset
   (every 15th training image), never from the validation set.

Usage:
    .venv/bin/python step1_diagnostics.py                     # full run (~25 min on an M2)
    .venv/bin/python step1_diagnostics.py --max-images 64     # smoke test
Output: runs/step1_diagnostics.json (+ printed summary).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import timm
import torch
import torch.nn.functional as F
import torchvision

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vit2snn_distill as m  # noqa: E402

STUDENTS = {
    "phase1+phase2": "runs/imagenette_ssa/student_final.pt",
    "phase2_only_5ep": "runs/imagenette_ssa_ft_only/student_final.pt",
    "phase2_only_12ep": "runs/imagenette_ssa_ft_only_12ep/student_final.pt",
}
PHASE1 = "runs/imagenette_ssa/distill_latest.pt"
IDX10 = sorted(m.IMAGENETTE_TO_IN1K.values())


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-dir", default="data/imagenette2-320/train")
    p.add_argument("--val-dir", default="data/imagenette2-320/val")
    p.add_argument("--parts", default="A,B,C", help="subset of A,B,C")
    p.add_argument("--t-values", default="1,2,4,8")
    p.add_argument("--stages", default="8,12", help="teacher stages for part C (stage l = output of block l)")
    p.add_argument("--n-random", type=int, default=5, help="random control channels in part C")
    p.add_argument("--max-images", type=int, default=None, help="cap on validation images (smoke test)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/step1_diagnostics.json")
    return p.parse_args()


def strided(n: int, step: int, cap: int | None = None) -> list[int]:
    idx = list(range(0, n, step))
    return idx[:cap] if cap else idx


def cache_images(ds, indices: list[int], bs: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Decode once; keep normalised images in float16 on the CPU (about 1.2 GB for 3925 images)."""
    batches = []
    for i in range(0, len(indices), bs):
        xs, ys = zip(*(ds[j] for j in indices[i:i + bs]))
        batches.append((torch.stack(xs).half(), torch.tensor(ys)))
    return batches


def main() -> None:
    a = parse()
    parts = set(a.parts.split(","))
    torch.manual_seed(a.seed)
    dev = m.get_device("auto")
    cfg = m.Config(data="imagefolder", train_dir=a.train_dir, val_dir=a.val_dir,
                   num_workers=0, batch_size=a.batch_size)
    teacher = timm.create_model(cfg.teacher_name, pretrained=True).to(dev).eval().requires_grad_(False)
    val_loader = m.build_loaders(cfg, teacher, dev).test  # official validation set
    val_ds = val_loader.dataset
    train_ds = torchvision.datasets.ImageFolder(cfg.train_dir, val_ds.transform)  # eval transform
    train_ds.target_transform = m._target_transform(train_ds.classes)

    n_val = len(val_ds)
    step = 1 if not a.max_images else max(1, n_val // a.max_images)
    t0 = time.time()
    val_b = cache_images(val_ds, strided(n_val, step, a.max_images), a.batch_size)
    sub_b = cache_images(val_ds, strided(n_val, 15, a.max_images), a.batch_size)
    cal_b = cache_images(train_ds, strided(len(train_ds), 15, a.max_images), a.batch_size)
    n_eval = sum(len(y) for _, y in val_b)
    print(f"cached {n_eval} val / {sum(len(y) for _, y in sub_b)} val-subset / "
          f"{sum(len(y) for _, y in cal_b)} train-calibration images in {time.time() - t0:.0f}s", flush=True)
    idx10 = torch.tensor(IDX10, device=dev)
    D = teacher.embed_dim

    def acc(fn) -> tuple[float, float]:
        c1000 = c10 = n = 0
        with torch.no_grad():
            for xh, y in val_b:
                x, y = xh.to(dev).float(), y.to(dev)
                z = fn(x)
                c1000 += (z.argmax(-1) == y).sum().item()
                c10 += (idx10[z[:, idx10].argmax(-1)] == y).sum().item()
                n += y.numel()
        return 100 * c1000 / n, 100 * c10 / n

    def load_student(path: str) -> m.SpikingViT:
        s = m.SpikingViT(teacher, cfg).to(dev)
        s.load_state_dict(torch.load(path, map_location=dev)["student"])
        return s.eval()

    def set_T(s: m.SpikingViT, T: int) -> None:
        s.T = T
        s.encoder.T = T

    out: dict = {"n_eval_images": n_eval, "args": vars(a)}

    # ---------------------------------------------------------------- A. T sweep
    if "A" in parts:
        print("\n=== A. Inference-time T sweep (trained with T=4) ===", flush=True)
        out["A_T_sweep"] = {}
        Ts = [int(t) for t in a.t_values.split(",")]
        for name, path in STUDENTS.items():
            if not Path(path).exists():
                print(f"  skip {name}: {path} not found"); continue
            s = load_student(path)
            lifs = [mm for mm in s.modules() if isinstance(mm, m.MultiStepLIF)]
            out["A_T_sweep"][name] = {}
            for T in Ts:
                set_T(s, T)
                for lf in lifs:
                    lf.reset_stats(); lf.record = True
                a1000, a10 = acc(lambda x: s(x).mean(0))
                spikes = sum(float(lf.spike_sum_t.sum()) for lf in lifs if lf.spike_sum_t is not None)
                neurons = sum(lf.neuron_count * T for lf in lifs)
                for lf in lifs:
                    lf.record = False; lf.reset_stats()
                r = {"top1_1000way": a1000, "top1_10way": a10, "firing_rate": spikes / max(1, neurons)}
                out["A_T_sweep"][name][T] = r
                print(f"  {name:18s} T={T}: 1000-way {a1000:6.2f}%  10-way {a10:6.2f}%  rate {r['firing_rate']:.4f}", flush=True)
            del s; m.free_memory(dev)

    # ---------------------------------------------------------------- B. relative errors
    if "B" in parts:
        print("\n=== B. Relative hidden-state error rho_l (val subset) ===", flush=True)
        out["B_rel_error"] = {}
        models = {"phase1_only": PHASE1, **STUDENTS}
        for name, path in models.items():
            if not Path(path).exists():
                print(f"  skip {name}"); continue
            s = load_student(path)
            S = s.num_stages
            z64 = lambda: torch.zeros(D, dtype=torch.float64)  # noqa: E731
            acc_ = [dict(err=z64(), sq=z64(), sm=z64(), n=0, err_c=z64(), sq_c=z64(), sm_c=z64(), n_c=0)
                    for _ in range(S)]
            with torch.no_grad():
                for xh, _ in sub_b:
                    x = xh.to(dev).float()
                    ht, hs = m.teacher_embed(teacher, x), s.encoder(x)
                    for l in range(S):
                        if l > 0:
                            ht, hs = teacher.blocks[l - 1](ht), s.blocks[l - 1](hs)
                        d = (hs.mean(0) - ht) ** 2
                        r = acc_[l]
                        r["err"] += d.sum((0, 1)).cpu().double(); r["sq"] += (ht ** 2).sum((0, 1)).cpu().double()
                        r["sm"] += ht.sum((0, 1)).cpu().double(); r["n"] += ht.shape[0] * ht.shape[1]
                        r["err_c"] += d[:, 0].sum(0).cpu().double(); r["sq_c"] += (ht[:, 0] ** 2).sum(0).cpu().double()
                        r["sm_c"] += ht[:, 0].sum(0).cpu().double(); r["n_c"] += ht.shape[0]
            rows = []
            for l, r in enumerate(acc_):
                den = (r["sq"] - r["sm"] ** 2 / r["n"]).sum()
                den_c = (r["sq_c"] - r["sm_c"] ** 2 / r["n_c"]).sum()
                # The CLS token at stage 0 is the same for every image (cls_token + pos_embed[0]):
                # its across-image variance is ~0 and rho_cls is undefined there.
                ok_c = float(den_c) > 1e-6 * float(r["sq_c"].sum())
                rows.append({"stage": l, "rho_all_tokens": float(r["err"].sum() / den),
                             "rho_cls": float(r["err_c"].sum() / den_c) if ok_c else None})
            out["B_rel_error"][name] = rows
            print(f"  {name:18s} rho_all: " + " ".join(f"{x['rho_all_tokens']:.2f}" for x in rows))
            print(f"  {'':18s} rho_cls: " + " ".join("n/a" if x["rho_cls"] is None else f"{x['rho_cls']:.2f}"
                                                         for x in rows), flush=True)
            del s; m.free_memory(dev)

    # ---------------------------------------------------------------- C. teacher interventions
    if "C" in parts:
        print("\n=== C. Teacher channel interventions (calibrated on train subset) ===", flush=True)
        out["C_interventions"] = {}
        with torch.no_grad():  # baseline teacher logits
            base = [teacher(xh.to(dev).float()).cpu() for xh, _ in val_b]
        b1000, b10 = acc(lambda x: teacher(x))
        out["C_interventions"]["baseline"] = {"top1_1000way": b1000, "top1_10way": b10}
        print(f"  baseline: 1000-way {b1000:.2f}%  10-way {b10:.2f}%")
        gen = torch.Generator().manual_seed(a.seed)

        for stage in [int(x) for x in a.stages.split(",")]:
            blk = teacher.blocks[stage - 1]
            sm = torch.zeros(D, dtype=torch.float64); sq = torch.zeros(D, dtype=torch.float64)
            mx = torch.zeros(D, dtype=torch.float64); n = 0; vals = []
            with torch.no_grad():
                for xh, _ in cal_b:
                    h = m.teacher_embed(teacher, xh.to(dev).float())
                    for b_ in teacher.blocks[:stage]:
                        h = b_(h)
                    sm += h.sum((0, 1)).cpu().double(); sq += (h ** 2).sum((0, 1)).cpu().double()
                    mx = torch.maximum(mx, h.abs().amax((0, 1)).cpu().double()); n += h.shape[0] * h.shape[1]
                    vals.append(h.cpu().float())
            mu = (sm / n).float().to(dev)
            order = torch.argsort(sq, descending=True).tolist()
            c_star, c_2 = order[0], order[1]
            tau = float(mx[order[5:]].max())  # largest |h| among channels outside the top-5 energy channels
            H = torch.cat(vals)  # [n_img, N, D] on CPU
            e = H[..., c_star] ** 2
            flat = e.flatten().sort(descending=True).values
            k1 = max(1, int(0.01 * flat.numel()))
            calib = {"top_energy_channel": c_star, "second_energy_channel": c_2,
                     "top_channel_energy_share": float(sq[c_star] / sq.sum()),
                     "clip_threshold_tau": tau,
                     "top_channel_energy_share_on_CLS": float(e[:, 0].sum() / e.sum()),
                     "top_channel_energy_share_on_top1pct_tokens": float(flat[:k1].sum() / flat.sum()),
                     "top_channel_max_abs": float(mx[c_star])}
            del H, vals
            rng = torch.Generator().manual_seed(a.seed + stage)
            pool = torch.tensor(order[5:])
            rand_ch = pool[torch.randperm(len(pool), generator=rng)[:a.n_random]].tolist()
            print(f"\n  stage {stage}: top channel {c_star} (share {calib['top_channel_energy_share']:.3f}, "
                  f"max|h| {calib['top_channel_max_abs']:.1f}, CLS share {calib['top_channel_energy_share_on_CLS']:.3f}, "
                  f"top-1% tokens share {calib['top_channel_energy_share_on_top1pct_tokens']:.3f}), "
                  f"2nd {c_2}, tau={tau:.2f}, random {rand_ch}", flush=True)

            configs = [("mean", c_star), ("clip", c_star), ("noise_matched", c_star), ("mean", c_2)]
            configs += [("mean", c) for c in rand_ch]
            res_stage = {"calibration": calib, "runs": []}
            for kind, c in configs:
                stats = {"rel": 0.0, "n": 0, "d2": 0.0, "h2": 0.0}

                def hook(mod, inp, o, kind=kind, c=c, stats=stats):
                    h = o.clone()
                    if kind == "mean":
                        h[..., c] = mu[c]
                    elif kind == "clip":
                        h[..., c] = h[..., c].clamp(-tau, tau)
                    elif kind == "noise_matched":  # same per-token norm as the mean-replacement of c
                        delta = (o[..., c] - mu[c]).abs()
                        z = torch.randn(o.shape, generator=gen).to(o.device)
                        z[..., c] = 0
                        h = o + z / z.norm(dim=-1, keepdim=True).clamp_min(1e-12) * delta.unsqueeze(-1)
                    d = (h - o).norm(dim=-1) / o.norm(dim=-1).clamp_min(1e-12)
                    stats["rel"] += float(d.sum()); stats["n"] += d.numel()
                    stats["d2"] += float(((h - o) ** 2).sum()); stats["h2"] += float((o ** 2).sum())
                    return h

                handle = blk.register_forward_hook(hook)
                c1000 = c10 = f1000 = f10 = 0; kl = 0.0; nn_ = 0
                try:
                    with torch.no_grad():
                        for (xh, y), zb in zip(val_b, base):
                            x, y, zb = xh.to(dev).float(), y.to(dev), zb.to(dev)
                            z = teacher(x)
                            c1000 += (z.argmax(-1) == y).sum().item()
                            c10 += (idx10[z[:, idx10].argmax(-1)] == y).sum().item()
                            f1000 += (z.argmax(-1) != zb.argmax(-1)).sum().item()
                            f10 += (z[:, idx10].argmax(-1) != zb[:, idx10].argmax(-1)).sum().item()
                            kl += float(F.kl_div(F.log_softmax(z, -1), F.log_softmax(zb, -1),
                                                 log_target=True, reduction="sum"))
                            nn_ += y.numel()
                finally:
                    handle.remove()
                r = {"kind": kind, "channel": c, "top1_1000way": 100 * c1000 / nn_, "top1_10way": 100 * c10 / nn_,
                     "flip_rate_1000way": 100 * f1000 / nn_, "flip_rate_10way": 100 * f10 / nn_,
                     "mean_KL_base_to_perturbed": kl / nn_, "mean_rel_perturbation": stats["rel"] / max(1, stats["n"]),
                     "global_rel_perturbation": (stats["d2"] / max(stats["h2"], 1e-12)) ** 0.5}
                res_stage["runs"].append(r)
                print(f"    {kind:13s} ch {c:3d}: 1000-way {r['top1_1000way']:6.2f}%  10-way {r['top1_10way']:6.2f}%  "
                      f"flip1000 {r['flip_rate_1000way']:5.1f}%  flip10 {r['flip_rate_10way']:5.1f}%  "
                      f"KL {r['mean_KL_base_to_perturbed']:.3f}  |dh|/|h| tok {r['mean_rel_perturbation']:.3f} "
                      f"glob {r['global_rel_perturbation']:.3f}", flush=True)
            out["C_interventions"][f"stage_{stage}"] = res_stage

    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\nsaved {a.out}")


if __name__ == "__main__":
    main()
