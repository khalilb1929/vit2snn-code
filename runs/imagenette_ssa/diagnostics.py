"""Post-hoc diagnostics for the Imagenette run (no training).
1) Teacher hidden-state statistics per stage and student relative error after phase 1
   (checkpoint distill_latest.pt, chained inputs, eval mode = same protocol as evaluate_stage).
2) Operation breakdown of the final student (student_final.pt).
Subset: every 15th validation image (262 images, all 10 classes)."""
import json, sys, torch
sys.argv = ["x", "--data", "imagefolder", "--train-dir", "data/imagenette2-320/train",
            "--val-dir", "data/imagenette2-320/val", "--num-workers", "0", "--batch-size", "32"]
sys.path.insert(0, ".")
import timm
import vit2snn_distill as m
from torch.utils.data import DataLoader, Subset

cfg = m.parse_args(); dev = torch.device("mps")
t = timm.create_model(cfg.teacher_name, pretrained=True).to(dev).eval()
_, val = m.build_loaders(cfg, t, dev)
sub = Subset(val.dataset, list(range(0, len(val.dataset), 15)))
dl = DataLoader(sub, batch_size=32)

def load(path):
    s = m.SpikingViT(t, cfg).to(dev)
    s.load_state_dict(torch.load(path, map_location=dev)["student"]); return s.eval()

# ---- 1) hidden-state statistics -------------------------------------------------------
s = load("runs/imagenette_ssa/distill_latest.pt")
D = t.embed_dim; S = 13
acc = [dict(err=torch.zeros(D), sq=torch.zeros(D), sm=torch.zeros(D), n=0, maxabs=0.0) for _ in range(S)]
with torch.no_grad():
    for x, _ in dl:
        x = x.to(dev); ht = m.teacher_embed(t, x); hs = s.encoder(x)
        for l in range(S):
            if l > 0:
                ht = t.blocks[l - 1](ht); hs = s.blocks[l - 1](hs)
            a = acc[l]
            a["err"] += ((hs.mean(0) - ht) ** 2).sum((0, 1)).cpu()
            a["sq"] += (ht ** 2).sum((0, 1)).cpu(); a["sm"] += ht.sum((0, 1)).cpu()
            a["n"] += ht.shape[0] * ht.shape[1]; a["maxabs"] = max(a["maxabs"], ht.abs().max().item())
rows = []
for l, a in enumerate(acc):
    n = a["n"]; var_ch = a["sq"] / n - (a["sm"] / n) ** 2
    order = torch.argsort(a["sq"], descending=True); top = order[:5]; rest = order[5:]
    rows.append(dict(stage=l,
        mse=float(a["err"].sum() / (n * D)),
        mean_sq=float(a["sq"].sum() / (n * D)),
        rel_err=float(a["err"].sum() / (var_ch.sum() * n)),
        max_abs=a["maxabs"],
        top1_energy_share=float(a["sq"][order[0]] / a["sq"].sum()),
        top5_energy_share=float(a["sq"][top].sum() / a["sq"].sum()),
        top5_err_share=float(a["err"][top].sum() / a["err"].sum()),
        mse_excl_top5=float(a["err"][rest].sum() / (n * (D - 5))),
        rel_err_excl_top5=float(a["err"][rest].sum() / (var_ch[rest].sum() * n)),
        top_channels=order[:3].tolist()))
print(f"{'st':>2} {'MSE':>8} {'E[h^2]':>8} {'relErr':>7} {'max|h|':>7} {'top1E':>6} {'top5E':>6} {'top5Err':>7} {'MSE-x5':>7} {'rel-x5':>7}  top ch")
for r in rows:
    print(f"{r['stage']:>2} {r['mse']:8.3f} {r['mean_sq']:8.3f} {r['rel_err']:7.3f} {r['max_abs']:7.1f} "
          f"{r['top1_energy_share']:6.3f} {r['top5_energy_share']:6.3f} {r['top5_err_share']:7.3f} "
          f"{r['mse_excl_top5']:7.3f} {r['rel_err_excl_top5']:7.3f}  {r['top_channels']}")

# ---- 2) operation breakdown of the final student ----------------------------------------
s = load("runs/imagenette_ssa/student_final.pt")
lifs = [mm for mm in s.modules() if isinstance(mm, m.MultiStepLIF)]
attns = [mm for mm in s.modules() if isinstance(mm, m.SpikingSelfAttention)]
for mm in lifs: mm.reset_stats(); mm.record = True
for a in attns: a.ac_ops, a.record = 0.0, True
cnt = m.OpCounter().attach(s); n_img = 0
with torch.no_grad():
    for x, _ in dl:
        s(x.to(dev)); n_img += x.shape[0]
cnt.detach()
T, N = cfg.T, 197
lin_ac = cnt.ac / n_img; attn_ac = sum(a.ac_ops for a in attns) / n_img
lif_updates = sum(mm.neuron_count for mm in lifs) * T / n_img
resid_adds = len(s.blocks) * 2 * T * N * D
bn_bias_adds = sum(mm.linear.out_features for mm in s.modules() if isinstance(mm, m.TBLinearBN)) * T * N
dense_eq = (N * 2 * D * D + len(s.blocks) * N * 12 * D * D) * T
ops = dict(images=n_img, mac=cnt.mac / n_img, linear_ac=lin_ac, attn_ac=attn_ac,
           rbar_linear=lin_ac / dense_eq, lif_updates=lif_updates, residual_adds=resid_adds,
           bn_bias_adds_after_folding=bn_bias_adds)
for k, v in ops.items():
    print(f"{k:28s} {v/1e6:10.2f} M" if k not in ("images", "rbar_linear") else f"{k:28s} {v:10.4f}")
json.dump(dict(hidden=rows, ops=ops), open("runs/imagenette_ssa/diagnostics.json", "w"), indent=2)
