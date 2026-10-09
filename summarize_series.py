#!/usr/bin/env python3
"""
summarize_series.py -- tables for the factorial series written by run_series.sh.

For each arm and seed: final top-1 (restricted to the dataset's classes and 1000-way), top-5,
firing rate, compute-only energy estimate, optimiser updates, examples and wall-clock time
(phase-1 cost reported separately and added for the p1_* arms). Then mean and sample standard
deviation over seeds (no confidence intervals: with 3 seeds they would be very wide), and the
per-seed paired differences p1 - nop1 at each fine-tuning length, with their sign count.

Usage: .venv/bin/python summarize_series.py runs/series_v2 > runs/series_v2/summary.md
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

ARMS = ["p1_ft05", "nop1_ft05", "p1_ft12", "nop1_ft12"]


def load(p: Path) -> dict | None:
    f = p / "metrics.json"
    if not f.exists():
        return None
    m = json.loads(f.read_text())
    return m if m.get("complete") else None


def row(m: dict, ph1: dict | None) -> dict:
    fin, prof = m["final"], m.get("profile", {})
    ft = m.get("finetune", [])
    r = {"split": fin.get("split"),
         "top1_restricted": fin.get("student_top1_restricted"), "top1_1000": fin["student_top1"],
         "top5_1000": fin["student_top5"], "firing_rate": prof.get("global_rate"),
         "est_energy_mJ": prof.get("student_energy_mJ"),
         "ft_updates": ft[-1]["updates_cum"] if ft else 0, "ft_examples": ft[-1]["examples_cum"] if ft else 0,
         "ft_seconds": sum(e["seconds"] for e in ft), "resumed": m.get("resumed", 0)}
    p1 = {"p1_updates": 0, "p1_examples": 0, "p1_seconds": 0.0}
    if ph1 is not None:
        d = ph1["distill"]
        p1 = {"p1_updates": sum(x["updates"] for x in d), "p1_examples": sum(x["examples"] for x in d),
              "p1_seconds": sum(x["seconds"] for x in d)}
    r.update(p1)
    r["total_seconds"] = r["ft_seconds"] + r["p1_seconds"]
    return r


def fmt(x, nd=2) -> str:
    return "–" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/series_v2")
    seeds = sorted(p.name for p in root.glob("seed*") if p.is_dir())
    data: dict[str, dict[str, dict]] = {a: {} for a in ARMS}
    for s in seeds:
        ph1 = load(root / s / "phase1")
        for a in ARMS:
            m = load(root / s / a)
            if m is not None:
                data[a][s] = row(m, ph1 if a.startswith("p1_") else None)

    print(f"# Series summary: {root}\n")
    print("Final numbers on the official validation set (already inspected in preliminary runs).")
    print("Energy = compute-only analytical estimate, not a measurement.\n")
    cols = ["top1_restricted", "top1_1000", "top5_1000", "firing_rate", "est_energy_mJ",
            "ft_updates", "p1_updates", "ft_examples", "p1_examples", "ft_seconds", "p1_seconds",
            "total_seconds", "resumed", "split"]
    print("## Individual runs\n")
    print("| arm | seed | " + " | ".join(cols) + " |")
    print("|" + "---|" * (len(cols) + 2))
    for a in ARMS:
        for s, r in sorted(data[a].items()):
            print(f"| {a} | {s} | " + " | ".join(fmt(r[c], 4 if c == "firing_rate" else 2) for c in cols) + " |")

    print("\n## Mean and sample standard deviation over seeds\n")
    keys = ["top1_restricted", "top1_1000", "top5_1000", "firing_rate", "total_seconds"]
    print("| arm | n | " + " | ".join(f"{k} mean | {k} sd" for k in keys) + " |")
    print("|" + "---|" * (2 + 2 * len(keys)))
    for a in ARMS:
        rs = list(data[a].values())
        cells = []
        for k in keys:
            v = [r[k] for r in rs if r[k] is not None]
            cells += [fmt(st.mean(v)) if v else "–", fmt(st.stdev(v)) if len(v) > 1 else "–"]
        print(f"| {a} | {len(rs)} | " + " | ".join(cells) + " |")

    print("\n## Paired differences p1 - nop1 (same seed, same fine-tuning schedule)\n")
    for ft in ("ft05", "ft12"):
        for k in ("top1_restricted", "top1_1000"):
            diffs = {s: data[f"p1_{ft}"][s][k] - data[f"nop1_{ft}"][s][k]
                     for s in data[f"p1_{ft}"] if s in data[f"nop1_{ft}"]
                     and data[f"p1_{ft}"][s][k] is not None and data[f"nop1_{ft}"][s][k] is not None}
            if not diffs:
                continue
            v = list(diffs.values())
            pos = sum(d > 0 for d in v)
            print(f"- {ft}, {k}: " + ", ".join(f"{s}: {d:+.2f}" for s, d in sorted(diffs.items()))
                  + f" | mean {st.mean(v):+.2f}" + (f", sd {st.stdev(v):.2f}" if len(v) > 1 else "")
                  + f" | positive in {pos}/{len(v)} seeds")
    print("\nThe p1 arms also include the phase-1 cost (p1_updates, p1_seconds): the factorial "
          "design compares equal fine-tuning schedules, not equal total budgets.")


if __name__ == "__main__":
    main()
