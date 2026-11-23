"""Standalone policy construction for the PV-MFAPS plug-in.

The module turns FIT-only calibration arrays and a measured cost table into a
class-conditional finite action plan.  It deliberately keeps calibration,
allocation, and execution separate so a host trainer can replace any layer.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .allocator_core import solve_pv_mfaps_dp


@dataclass(frozen=True)
class Action:
    k: int
    alpha: float


@dataclass(frozen=True)
class ActionPlan:
    k: tuple[int, ...]
    alpha: tuple[float, ...]
    budget: float
    expected_cost: float
    objective: float
    certificate: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _array(calibration: Mapping[str, Any], name: str) -> np.ndarray:
    value = np.asarray(calibration[name], dtype=float)
    if value.ndim != 1:
        raise ValueError(f"calibration[{name!r}] must be one-dimensional")
    return value


def build_plan(
    calibration: Mapping[str, Any],
    *,
    budget: float,
    q: Sequence[float] | None = None,
    mu: float = 1.0,
    nu: Sequence[float] | None = None,
    k_grid: Sequence[int] = (0, 1, 2, 4),
    alpha_grid: Sequence[float] = (0.25,),
    xi: float = 1.01,
    budget_scale: int = 1000,
    macro_cap: float | None = None,
) -> ActionPlan:
    """Solve the finite-action allocation problem from FIT-only proxies.

    ``A``, ``rho``, ``sigma``, and ``n`` are calibration quantities.  The
    returned certificate records the exact quantised-cost problem solved by the
    DP.  It does not turn proxy calibration into a population-level theorem.
    """
    A = _array(calibration, "A")
    rho = _array(calibration, "rho")
    sigma = _array(calibration, "sigma")
    n = _array(calibration, "n")
    C = len(A)
    if not (len(rho) == len(sigma) == len(n) == C):
        raise ValueError("A, rho, sigma, and n must have equal lengths")
    if q is None:
        qv = np.full(C, 1.0 / C)
    else:
        qv = np.asarray(q, dtype=float)
        if qv.shape != (C,):
            raise ValueError("q must have one entry per class")
    result = solve_pv_mfaps_dp(
        A=A,
        rho=rho,
        sigma=sigma,
        n=n,
        mu=mu,
        nu=nu,
        q=qv,
        budget=float(budget),
        k_grid=k_grid,
        alpha_grid=alpha_grid,
        xi=xi,
        macro_cap=macro_cap,
        budget_scale=budget_scale,
        return_frontier_stats=True,
    )
    k = tuple(int(v) for v in result["k"])
    alpha = tuple(float(v) for v in result["alpha"])
    if any(kc == 0 and ac != 0.0 for kc, ac in zip(k, alpha)):
        raise RuntimeError("invalid action plan: k=0 must imply alpha=0")
    return ActionPlan(
        k=k,
        alpha=alpha,
        budget=float(budget),
        expected_cost=float(result["certificate"]["actual_cost"]),
        objective=float(result["certificate"]["objective"]),
        certificate={
            "solver": "finite-action Pareto-frontier DP",
            "quantised_budget_scale": int(budget_scale),
            "k_grid": [int(v) for v in k_grid],
            "alpha_grid": [float(v) for v in alpha_grid],
            "xi": float(xi),
            "frontier_stats": result.get("diagnostics", {}),
            "scope": "FIT calibration proxy",
        },
    )


def plan_from_json(path: str, **kwargs: Any) -> ActionPlan:
    import json
    from pathlib import Path
    payload = json.loads(Path(path).read_text())
    return build_plan(payload, **kwargs)
