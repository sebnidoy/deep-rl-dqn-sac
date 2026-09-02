"""Summarize logged metrics across all runs in exp/ for the HW3 report."""
import csv
import json
import os
from pathlib import Path

EXP = Path(__file__).parent / "exp"


def load(run_dir: Path):
    rows = []
    with open(run_dir / "log.csv", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def series(rows, key, step_key="step"):
    out = []
    for r in rows:
        v = r.get(key, "")
        s = r.get(step_key, "")
        if v not in ("", None) and s not in ("", None):
            out.append((int(float(s)), float(v)))
    return out


def summarize(run_dir: Path):
    rows = load(run_dir)
    flags = json.loads((run_dir / "flags.json").read_text())
    print("=" * 70)
    print(run_dir.name)
    print("  cfg:", flags.get("config_file"), "seed:", flags.get("seed"))
    keys = [k for k in rows[0].keys() if k != "step"]
    for k in keys:
        s = series(rows, k)
        if not s:
            continue
        vals = [v for _, v in s]
        best_step, best = max(s, key=lambda t: t[1])
        last_step, last = s[-1]
        print(
            f"  {k:22s} n={len(s):5d} max={best:10.3f}@{best_step:<8d} "
            f"min={min(vals):10.3f} last={last:10.3f}@{last_step}"
        )
    # threshold crossings for eval return
    ev = series(rows, "Eval_AverageReturn")
    if ev:
        for thr in (200, 500, 1500, 6000):
            hits = [st for st, v in ev if v >= thr]
            if hits:
                print(f"  Eval>={thr}: first@{hits[0]}  count={len(hits)}")


if __name__ == "__main__":
    for d in sorted(EXP.iterdir()):
        if d.is_dir() and (d / "log.csv").exists():
            summarize(d)
