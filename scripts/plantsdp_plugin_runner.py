#!/usr/bin/env python3
"""Calibration-to-plan adapter for a host PlantSPD-style runner.

The script consumes a FIT-only calibration JSON and emits a portable action
plan. It does not load images or checkpoints.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pv_mfaps_plugin import build_plan  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", required=True, help="FIT-only calibration JSON")
    parser.add_argument("--budget", required=True, type=float)
    parser.add_argument("--output", required=True)
    parser.add_argument("--k-grid", default="0,1,2,4")
    parser.add_argument("--alpha-grid", default="0.25")
    args = parser.parse_args()
    payload = json.loads(Path(args.calibration).read_text())
    k_grid = tuple(int(v) for v in args.k_grid.split(",") if v.strip())
    alpha_grid = tuple(float(v) for v in args.alpha_grid.split(",") if v.strip())
    plan = build_plan(payload, budget=args.budget, k_grid=k_grid, alpha_grid=alpha_grid)
    out = {"status": "EXECUTED", "held_reads": 0, "plan": plan.to_dict()}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": out["status"], "output": args.output, "k": plan.k}))


if __name__ == "__main__":
    main()
