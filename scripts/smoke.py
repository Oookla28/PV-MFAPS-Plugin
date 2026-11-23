#!/usr/bin/env python3
"""Run a dependency-light FIT proxy smoke for the standalone plug-in."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pv_mfaps_plugin import build_plan  # noqa: E402


def main() -> None:
    classes = 12
    calibration = {
        "A": np.linspace(0.7, 1.4, classes).tolist(),
        "rho": np.linspace(0.65, 0.92, classes).tolist(),
        "sigma": np.linspace(0.04, 0.16, classes).tolist(),
        "n": np.linspace(5, 48, classes).tolist(),
    }
    plan = build_plan(calibration, budget=0.42, mu=1.0, k_grid=(0, 1, 2, 4), alpha_grid=(0.25,))
    result = plan.to_dict()
    print(json.dumps({"status": "PASS", "plan": result}, indent=2))


if __name__ == "__main__":
    main()
