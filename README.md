# vit2snn: layer-wise distillation of ViT-Tiny into a spiking ViT (T = 4)

Code and logs for the technical report *Layer-wise Distillation of ViT-Tiny into a Spiking
Vision Transformer: a preliminary implementation study on Apple Silicon*
(`vit2snn_report.pdf`).

This is an implementation study and an exploratory evaluation, not a new method. All results
come from single-seed runs on Imagenette, evaluated with the 1000-way ImageNet head. The
energy figures are compute-only analytical estimates, not measurements. See Section 12 of the
report for the full list of limitations.

## Results (Imagenette validation, single seed per row)

| Model | Training time | Top-1, 1000-way | Top-5, 1000-way | Top-1, 10-way | Firing rate | Est. energy* |
|---|---|---|---|---|---|---|
| Teacher ViT-Tiny/16 (not fine-tuned) | – | 79.11 % | 95.49 % | 99.16 % | – | 5.77 mJ |
| Student, phase 1 + phase 2 | 3.6 h | 63.67 % | 93.48 % | 63.67 % | 7.1 % | 0.52 mJ |
| Student, phase 2 only, 5 epochs | 1.5 h | 48.23 % | 90.22 % | 48.23 % | 9.1 % | 0.75 mJ |
| Student, phase 2 only, 12 epochs (matched time) | 3.9 h | 59.72 % | 91.77 % | – | 9.1 % | 0.77 mJ |

\* Compute-only analytical estimate (45 nm operation energies, memory traffic ignored). It is
not a measurement.

**How to read this table:**
- **1000-way evaluation:** it favours the students, which are fine-tuned on the 10 Imagenette
  labels, over the teacher, which is not.
- **10-way evaluation:** the teacher–student gap is 35.5 points instead of 15.4. The 12-epoch
  run was not evaluated 10-way.
- **Matched training time:** the student trained without phase 1 is 3.95 top-1 points behind
  the full schedule. With one seed per configuration, this difference is not shown to exceed
  run-to-run variance.

### Three-seed paired series (held-out monitoring split)

Top-1 is restricted to the 10 Imagenette classes and measured on the official validation set
(already inspected in the preliminary runs). Values are mean ± SD over seeds 0, 1 and 2.

| Arm | Top-1, 10-way | Firing rate | Total time |
|---|---|---|---|
| Phase 1 + fine-tuning, 5 epochs | 61.21 ± 0.38 % | 7.0 % | 2.7–3.1 h |
| Fine-tuning only, 5 epochs | 48.23 ± 0.29 % | 9.1 % | 1.4 h |
| Phase 1 + fine-tuning, 12 epochs | 70.19 ± 0.54 % | 7.5 % | 4.7–4.8 h |
| Fine-tuning only, 12 epochs | 59.86 ± 0.27 % | 9.3 % | 3.2–3.3 h |

**Paired differences (with − without phase 1):**
- **Same fine-tuning schedule:** +12.98 ± 0.51 points at 5 epochs and +10.33 ± 0.44 points at
  12 epochs. The difference is positive in all three seeds.
- **Roughly equal total time** (phase 1 + 5 epochs vs. 12 epochs without phase 1): +1.15 to
  +1.45 points. The two learning-rate schedules differ, so this is not a controlled comparison.

Run it with `make_split.py`, then `SEEDS="0 1 2" ./run_series.sh`, then
`summarize_series.py runs/series_v2`. Results are in `runs/series_v2/`.

See Section 12 of the report for the full list of limitations.

## Contents

| File | Purpose |
|---|---|
| `vit2snn_distill.py` | Whole pipeline: spiking student, layer-wise distillation (phase 1), CE + KL fine-tuning (phase 2), firing-rate and operation profiling |
| `diagnostics.py` | Post-hoc diagnostics of the phase-1 checkpoint: relative errors and teacher channel statistics (Table 8); operation breakdown |
| `eval10.py` | 10-way Imagenette evaluation of trained checkpoints: logits restricted to the 10 Imagenette classes |
| `vit2snn_report.tex` / `.pdf` | Report (the `.tex` includes the scripts in its appendices) |
| `runs/` | Logs, `metrics.json` and diagnostics of the reported runs |

## Setup

Tested with Python 3.9.6 on an Apple M2 (16 GB, MPS backend).

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
mkdir -p data && curl -L -o data/imagenette2-320.tgz https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz
tar -xzf data/imagenette2-320.tgz -C data && rm data/imagenette2-320.tgz
```

The teacher weights (`timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k`, about 23 MB) are
downloaded from the Hugging Face Hub on first use.

## Reproducing the reported runs

Smoke test on random images (about 15 s, no weight download):

```bash
.venv/bin/python vit2snn_distill.py --data fake --no-pretrained --batch-size 2 \
    --distill-epochs 1 --finetune-epochs 1 --max-steps 2 --eval-batches 2
```

Main run: phase 1 + phase 2, about 3.6 h on an M2.

```bash
.venv/bin/python vit2snn_distill.py --data imagefolder \
    --train-dir data/imagenette2-320/train --val-dir data/imagenette2-320/val \
    --batch-size 16 --distill-epochs 2 --finetune-epochs 5 --accum-steps 2 \
    --attn-mode ssa --out-dir runs/imagenette_ssa
```

The reported main run was interrupted during stage 6 and continued with the same command plus
`--resume`. As a result, its `metrics.json` only contains stages 6–12; stages 0–5 are in
`runs/imagenette_ssa.log`.

Ablation: phase 2 only, same fine-tuning schedule, about 1.5 h.

```bash
.venv/bin/python vit2snn_distill.py --data imagefolder \
    --train-dir data/imagenette2-320/train --val-dir data/imagenette2-320/val \
    --batch-size 16 --finetune-epochs 5 --accum-steps 2 --skip-distill \
    --attn-mode ssa --out-dir runs/imagenette_ssa_ft_only
```

Ablation: phase 2 only with a matched total training time (12 epochs, about 3.6 h).

```bash
.venv/bin/python vit2snn_distill.py --data imagefolder \
    --train-dir data/imagenette2-320/train --val-dir data/imagenette2-320/val \
    --batch-size 16 --finetune-epochs 12 --accum-steps 2 --skip-distill \
    --attn-mode ssa --out-dir runs/imagenette_ssa_ft_only_12ep
```

Diagnostics and the 10-way evaluation (a few minutes each). They need the checkpoints of the
runs above:

```bash
.venv/bin/python diagnostics.py    # -> runs/imagenette_ssa/diagnostics.json
.venv/bin/python eval10.py         # -> runs/eval10.json
```

All runs use the default seed (0). MPS kernels are not bit-exact, so reruns may differ
slightly from the reported numbers.

## Outputs of a run

`<out-dir>/metrics.json` contains:
- the configuration;
- the train and validation MSE of each distillation stage;
- the accuracy after phase 1 and after each fine-tuning epoch;
- the final accuracy;
- the firing-rate and operation profile.

The run also writes the checkpoints `distill_latest.pt`, `student_best.pt` and
`student_final.pt`. All reported numbers use the last epoch (`student_final.pt`).

## License

To be chosen by the author before publication.
