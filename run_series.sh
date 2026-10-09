#!/usr/bin/env bash
# run_series.sh -- factorial series {phase 1, no phase 1} x {5, 12 fine-tuning epochs} x seeds.
#
# For each seed: phase 1 is run once (phase1/), then four fine-tuning runs:
#   p1_ft05, p1_ft12   start from phase1/distill_latest.pt
#   nop1_ft05, nop1_ft12 start from the teacher-initialised student
# Runs are sequential (one GPU job at a time). Every run uses --resume, so re-launching this
# script after an interruption continues where it stopped; finished runs (DONE file) are
# skipped. The official validation set is touched only by --final-eval at the end of each
# fine-tuning run.
#
#   SEEDS="0 1 2" ./run_series.sh                        # full series
#   ROOT=runs/pilot SEEDS=0 EXTRA="--max-steps 4 --eval-batches 2" ./run_series.sh   # pilot
set -euo pipefail
cd "$(dirname "$0")"

PY=${PY:-.venv/bin/python}
SPLIT=${SPLIT:-splits/imagenette_train90_monitor10_seed0.json}
ROOT=${ROOT:-runs/series_v2}
SEEDS=${SEEDS:-"0 1 2"}
EXTRA=${EXTRA:-}
COMMON=(--data imagefolder --train-dir data/imagenette2-320/train
        --val-dir data/imagenette2-320/val --split-file "$SPLIT"
        --batch-size 16 --accum-steps 2 --attn-mode ssa --num-workers 2 --log-every 100)

[ -f "$SPLIT" ] || { echo "missing $SPLIT (run make_split.py first)"; exit 1; }

run() {
  local dir=$1; shift
  if [ -f "$dir/DONE" ]; then echo "== skip $dir (DONE)"; return; fi
  mkdir -p "$dir"
  echo "== $(date '+%F %T') start $dir"
  # shellcheck disable=SC2086
  caffeinate -i "$PY" -u vit2snn_distill.py "${COMMON[@]}" --out-dir "$dir" --resume "$@" $EXTRA \
    2>&1 | tee -a "$dir/train.log"
  [ -f "$dir/DONE" ] || { echo "!! $dir did not finish"; exit 1; }
  echo "== $(date '+%F %T') done $dir"
}

for s in $SEEDS; do
  D=$ROOT/seed$s
  run "$D/phase1"    --seed "$s" --distill-epochs 2 --skip-finetune
  run "$D/p1_ft05"   --seed "$s" --skip-distill --init-checkpoint "$D/phase1/distill_latest.pt" --finetune-epochs 5  --final-eval
  run "$D/nop1_ft05" --seed "$s" --skip-distill --finetune-epochs 5  --final-eval
  run "$D/p1_ft12"   --seed "$s" --skip-distill --init-checkpoint "$D/phase1/distill_latest.pt" --finetune-epochs 12 --final-eval
  run "$D/nop1_ft12" --seed "$s" --skip-distill --finetune-epochs 12 --final-eval
done
echo "== series finished: $ROOT"
