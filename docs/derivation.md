# From feedback residuals to a finite action plan

This document gives the mathematical contract implemented by the plug-in. The symbols describe the calibration and allocation layer. They do not claim that a FIT proxy is a population-level generalization theorem.

## 1. Feedback residual

Let the encoder parameters be `ψ`, the current readout be `W_t`, and a FIT-only reference readout be `W̃_t`. For a fixed batch, define

$$
e_t = \nabla_\psi L(\psi_t,W_t)-\nabla_\psi L(\psi_t,\widetilde W_t).
$$

A scale-normalised diagnostic is

$$
\kappa_t = \frac{\lVert e_t\rVert_2}{\lVert \nabla_\psi L(\psi_t,\widetilde W_t)\rVert_2+\delta}.
$$

The plug-in treats `κ_t` and the classwise proxy arrays as control signals. They are written into the trace so the host can inspect how an action was selected.

## 2. Classwise finite risk table

For class `c`, action `a=(k,α)`, and calibration quantities `A_c`, `ρ_c`, `σ_c`, and `n_c`, the default proxy is

$$
U_c(k,\alpha) = (1-\alpha)^2 A_c\rho_c^{2k}
+\alpha^2\frac{\sigma_c^2}{n_c}\,\xi^k.
$$

The action grid is finite. The native action is `(0,0)`, and the implementation enforces the legality rule `k=0 ⇒ α=0`. A host can provide a richer measured risk table directly to the DP solver.

## 3. Weighted allocation objective

Let `q_c` be the sampling or class-mass weight and let `ν_c≥0` encode a tail emphasis. The solver uses

$$
\min_{a_1,\ldots,a_C}
\sum_{c=1}^C\left(\frac{\mu}{C}+\nu_c\right)U_c(a_c)
$$

subject to the quantised budget

$$
\sum_{c=1}^C d_c(a_c) \le B.
$$

Measured cost tables can replace the default action cost. In the public solver, every floating-point cost is rounded upward to an integer grid with scale `S`, so a returned action remains feasible for the original budget.

## 4. Pareto-frontier dynamic program

A state after processing classes `1,…,j` is represented by `(b, r, s, π)`, where `b` is quantised cost, `r` is accumulated risk, `s` is the weighted objective, and `π` is the selected action prefix. For each legal action of class `j+1`, the transition is

$$
(b',r',s',\pi') = (b+d_{j+1}(a),\;r+U_{j+1}(a),\;s+w_{j+1}U_{j+1}(a),\;\pi\cup\{a\}).
$$

States with cost above the budget are discarded. For a fixed quantised cost, a state is removed when another state has no larger risk and no larger objective. The remaining states form the Pareto frontier. The final state with minimum objective is the exact solution of the finite quantised problem implemented in `allocator_core.py`.

The certificate records the action grid, quantisation scale, frontier sizes, selected actions, expected cost, and objective. It deliberately states the scope of the result so a host trainer can keep proxy optimisation separate from held-out evaluation.

## 5. Fused execution contract

The allocation plan is independent of the host model. The fused executor receives one batch, computes the encoder activation once, and exposes logical action counters separately from physical encoder counters. A correct host integration should satisfy

$$
\text{actions}_{\mathrm{fused}}=\text{actions}_{\mathrm{legacy}},
$$

while measuring a lower number of physical encoder forwards when multiple logical actions reuse the same activation. The equivalence smoke should compare the resulting state updates under the same seed, checkpoint, optimizer state, and action sequence.

## 6. What the host must report

A complete run should record

- the FIT calibration source and hash,
- the selected action vector and solver certificate,
- logical native/reference/solver counts,
- physical encoder forwards and backwards,
- FIT metrics and post-freeze evaluation metrics,
- measured FLOPs and synchronized wall-clock time.

The plug-in is designed so these fields can be collected without changing the host model's inference interface.
