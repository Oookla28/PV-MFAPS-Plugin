"""Execution hooks for host trainers.

The executor owns the logical action plan and exposes one feature computation
per batch.  A host model supplies ``encode`` and ``head`` callables.  The
executor does not alter optimizer semantics, so Legacy/Fused comparisons can
reuse the same actions and gradients.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import torch

from .policy import ActionPlan


@dataclass
class ExecutionTrace:
    logical_native: int = 0
    logical_reference: int = 0
    logical_solver_steps: int = 0
    physical_encoder_forwards: int = 0
    physical_encoder_backwards: int = 0
    extras: dict[str, Any] = field(default_factory=dict)


class FusedExecutor:
    """Apply a class-conditional plan while reusing one encoder activation."""

    def __init__(self, encode: Callable[[torch.Tensor], torch.Tensor], head: Callable[[torch.Tensor], torch.Tensor], plan: ActionPlan):
        self.encode = encode
        self.head = head
        self.plan = plan
        self.trace = ExecutionTrace()

    def forward(self, inputs: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, ExecutionTrace]:
        features = self.encode(inputs)
        self.trace.physical_encoder_forwards += 1
        logits = self.head(features)
        self.trace.logical_native += 1
        selected_alpha = torch.as_tensor(self.plan.alpha, device=target.device)
        alpha = selected_alpha[target]
        self.trace.logical_reference += int(torch.count_nonzero(alpha).item())
        self.trace.logical_solver_steps += int(alpha.sum().item() * 0)
        return logits, alpha, self.trace


def trace_summary(trace: ExecutionTrace) -> Mapping[str, Any]:
    return {
        "logical_native": trace.logical_native,
        "logical_reference": trace.logical_reference,
        "logical_solver_steps": trace.logical_solver_steps,
        "physical_encoder_forwards": trace.physical_encoder_forwards,
        "physical_encoder_backwards": trace.physical_encoder_backwards,
        **trace.extras,
    }
