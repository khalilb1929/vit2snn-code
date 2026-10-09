"""10-way evaluation on Imagenette of already trained models (no training).
Logits are restricted to the 10 ImageNet-1k classes of Imagenette; top-1 is computed among
those 10 classes. Also reports the 1000-way top-1 for cross-checking with metrics.json."""
import json, sys, torch
sys.argv = ["x", "--data", "imagefolder", "--train-dir", "data/imagenette2-320/train",
            "--val-dir", "data/imagenette2-320/val", "--num-workers", "0", "--batch-size", "32"]
sys.path.insert(0, ".")
import timm
import vit2snn_distill as m

cfg = m.parse_args(); dev = torch.device("mps")
t = timm.create_model(cfg.teacher_name, pretrained=True).to(dev).eval()
print("teacher pretrained tag:", t.pretrained_cfg.get("tag"), "| hf_hub_id:", t.pretrained_cfg.get("hf_hub_id"))
val = m.build_loaders(cfg, t, dev).test  # official validation set
idx = torch.tensor(sorted(m.IMAGENETTE_TO_IN1K.values()), device=dev)   # 10 IN-1k indices

def load(path):
    s = m.SpikingViT(t, cfg).to(dev)
    s.load_state_dict(torch.load(path, map_location=dev)["student"]); return s.eval()

models = {"teacher": lambda x: t(x),
          "phase1_only": None, "phase1+phase2": None, "phase2_only_5ep": None}
paths = {"phase1_only": "runs/imagenette_ssa/distill_latest.pt",
         "phase1+phase2": "runs/imagenette_ssa/student_final.pt",
         "phase2_only_5ep": "runs/imagenette_ssa_ft_only/student_final.pt"}
res = {}
with torch.no_grad():
    for name in models:
        f = models[name]
        if f is None:
            s = load(paths[name]); f = (lambda s: (lambda x: s(x).mean(0)))(s)
        c10 = c1000 = n = 0
        for x, y in val:
            x, y = x.to(dev), y.to(dev)
            z = f(x)
            c1000 += (z.argmax(-1) == y).sum().item()
            c10 += (idx[z[:, idx].argmax(-1)] == y).sum().item()
            n += y.numel()
        res[name] = {"top1_10way": 100 * c10 / n, "top1_1000way": 100 * c1000 / n, "n": n}
        print(f"{name:18s} 10-way top-1 = {res[name]['top1_10way']:.2f} %   (1000-way {res[name]['top1_1000way']:.2f} %, n={n})")
json.dump(res, open("runs/eval10.json", "w"), indent=2)
