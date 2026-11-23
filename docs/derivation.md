# From feedback residuals to a finite action plan

This document describes the part of the training loop owned by the plug-in. The host trainer still controls the backbone, optimizer, data loader and checkpoint lifecycle. The plug-in converts a measurable feedback discrepancy into a finite, auditable action plan.

## 1. Two readout heads, one measurable question

Let `ψ` denote the encoder parameters, `W_t` the current readout and `W̃_t` a FIT-only reference readout obtained by a short inner refinement. On a fixed batch, use the same loss

$$
L(\psi,W)=\frac{1}{m}\sum_{i=1}^{m}
\ell\bigl(W z_\psi(x_i),y_i\bigr).
$$

The current and reference readouts produce

$$
g_t=\nabla_{\psi}L(\psi_t,W_t),
\qquad
\widetilde g_t=\nabla_{\psi}L(\psi_t,\widetilde W_t).
$$

The plug-in records their difference

$$
e_t=g_t-\widetilde g_t.
$$

This residual has a narrow interpretation. It measures how much the encoder update changes when the current readout is replaced by a locally refined readout on the same FIT batch.

For comparison across batches and training stages, the normalized diagnostic is

$$
\kappa_t=
\frac{\lVert e_t\rVert_2}
{\lVert\widetilde g_t\rVert_2+\delta},
$$

where `δ` prevents division by a value close to zero. `κ_t` is a trajectory-level diagnostic. It should be stored together with the checkpoint, FIT batch, loss definition and calibration source that produced it.

## 2. A minimal two-class example

Consider two classes and two possible actions, `(0,0)` and `(1,0.25)`. Let

$$
A_1=A_2=1,
\qquad
\rho_1=0.9,
\qquad
\rho_2=0.7,
\qquad
\sigma_1=\sigma_2=0.1,
\qquad
n_1=n_2=10.
$$

For the native action `(0,0)`, both proxy risks are `1`. With `ξ=1.01`, the refined action gives

$$
U_1(1,0.25)
=0.75^2\times0.9^2
+0.25^2\times\frac{0.1^2}{10}\times1.01
\approx0.4557,
$$

$$
U_2(1,0.25)
=0.75^2\times0.7^2
+0.25^2\times\frac{0.1^2}{10}\times1.01
\approx0.2757.
$$

If the budget allows one refined action, uniform weighting selects class 2. The decision follows from the supplied risk table. No additional rule is needed. This is the smallest example of the plug-in's behaviour: calibration supplies the table, the action grid defines what can be done, and the budget determines how many actions can be afforded.

## 3. The classwise risk table

For class `c` and action `a=(k,α)`, the default proxy is

$$
U_c(k,\alpha)=(1-\alpha)^2 A_c\rho_c^{2k}+\alpha^2\frac{\sigma_c^2}{n_c}\xi^k.
$$

The first term describes the residual component that may decrease with additional inner steps. The second term accounts for batch variation, FIT support and the growth factor `ξ`.

`A_c`, `ρ_c`, `σ_c` and `n_c` are FIT calibration quantities. A host trainer may replace the default formula with a measured per-action risk table and pass those values directly to the DP solver.

## 4. Budgeted discrete optimization

Let `q_c` denote the class weight, `ν_c≥0` an optional tail emphasis, `d_c(a_c)` the action cost and `B` the total budget. The allocation problem is

$$
\min_{a_1,\ldots,a_C}
\sum_{c=1}^{C}
\left(\frac{\mu}{C}+\nu_c\right)U_c(a_c)
\quad\text{s.t.}\quad
\sum_{c=1}^{C}d_c(a_c)\le B.
$$

The actions come from a finite set

$$
\mathcal A=\{(0,0),(1,\alpha_1),(2,\alpha_2),(4,\alpha_4),\ldots\}.
$$

The implementation enforces `k=0 ⇒ α=0`. The finite grid makes each decision explicit and allows the selected plan, cost and objective to be written into a manifest.

## 5. Pareto-frontier dynamic programming

After processing classes `1,…,j`, a DP state is

$$
S_j=(b_j,r_j,s_j,\pi_j),
$$

where `b_j` is the quantized cost, `r_j` the accumulated risk, `s_j` the weighted objective and `π_j` the selected action prefix.

Adding action `a` for class `j+1` gives

$$
S_{j+1}=
\bigl(b_j+d_{j+1}(a),
      r_j+U_{j+1}(a),
      s_j+w_{j+1}U_{j+1}(a),
      \pi_j\cup\{a\}\bigr).
$$

States above the budget are discarded. At the same quantized cost, a state is dominated when another state has no larger risk and no larger objective. Dominated states can be removed without changing the optimum of the finite quantized problem.

The solver rounds costs upward after multiplying them by `budget_scale`. This is conservative, but it prevents a returned action from exceeding the original floating-point budget because of rounding. The certificate records the action grid, quantization scale, selected actions, actual cost, objective and frontier sizes.

The phrase “discrete optimum” therefore refers to the supplied risk table, action set and quantized cost model. It does not turn a FIT proxy into a population-level statistical guarantee.

## 6. Why Fused execution reduces work

A Legacy path may evaluate native, reference and solver actions as separate branches, repeating the encoder computation for the same batch. The Fused path computes the feature once

$$
H=\mathrm{Encoder}(X),
$$

and then applies the readout branches

$$
Z_{\mathrm{native}}=W_tH,
\qquad
Z_{\mathrm{ref}}=\widetilde W_tH.
$$

The comparison must preserve the logical action sequence

$$
\text{logical actions}_{\mathrm{Legacy}}
=\text{logical actions}_{\mathrm{Fused}},
$$

while allowing the physical encoder work to decrease

$$
\text{physical forwards}_{\mathrm{Fused}}
<\text{physical forwards}_{\mathrm{Legacy}}.
$$

`ExecutionTrace` stores these counts separately. A host comparison should reset model state, optimizer state and random seeds between paths and use synchronized CUDA timing.

## 7. What a complete host run should leave behind

A reproducible host integration should save:

1. the FIT calibration data and source hash;
2. the class order, action grid and budget;
3. the selected `(k,α)` plan and DP certificate;
4. logical native, reference and solver counters;
5. physical encoder forward and backward counters;
6. FIT metrics, evaluation metrics, FLOPs and wall-clock;
7. the frozen checkpoint, configuration hash and evaluation manifest.

The plug-in then remains responsible for the measurement, allocation and trace layer, while the host project retains control over data, model and evaluation policy.
