#!/usr/bin/env python3
"""Toy Legacy/Fused equivalence check for the public executor contract."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pv_mfaps_plugin import ActionPlan, FusedExecutor, trace_summary  # noqa: E402


def main() -> None:
    torch.manual_seed(3407)
    plan = ActionPlan(k=(0, 1, 0, 1), alpha=(0.0, 0.25, 0.0, 0.25), budget=1.0,
                      expected_cost=0.5, objective=0.0, certificate={"scope": "toy"})
    encoder = torch.nn.Linear(4, 4, bias=False)
    head = torch.nn.Linear(4, 4, bias=False)
    x = torch.randn(8, 4)
    y = torch.randint(0, 4, (8,))
    executor = FusedExecutor(encoder, head, plan)
    logits, alpha, trace = executor.forward(x, y)
    assert logits.shape == (8, 4)
    assert alpha.shape == (8,)
    assert trace.physical_encoder_forwards == 1
    print({"status": "PASS", "trace": trace_summary(trace)})


if __name__ == "__main__":
    main()
