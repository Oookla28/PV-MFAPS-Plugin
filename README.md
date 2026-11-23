# PV-MFAPS Plugin

**Feedback-aware precision allocation for long-tailed training**

PV-MFAPS is a small plug-in layer for an existing training loop. It addresses a specific mismatch in joint encoder–readout training. A classifier head can produce reasonable logits while still sending a different gradient to the encoder from the gradient produced by a readout that has been locally refined on the current representation.

The plug-in exposes three separable operations:

- **Measure** the difference between the current and reference readout feedback.
- **Allocate** a finite inner-loop budget across classes with a discrete Pareto solver.
- **Execute** the selected actions through a traceable Legacy or Fused path, so logical actions and physical encoder work can be measured separately.

The host trainer keeps ownership of the backbone, optimizer, data loader and checkpoint lifecycle. PV-MFAPS supplies the calibration, allocation and execution hooks.

## Architecture at a glance

<p align="center">
  <img src="docs/architecture_supplied.png" alt="PV-MFAPS plug-in architecture" width="860">
</p>

<p align="center"><a href="docs/architecture_supplied.png">Open the complete architecture figure</a></p>

The left side of the figure follows one training step. The right side expands the three decisions that determine the plug-in behaviour. A FIT-only reference probe estimates the feedback discrepancy. The risk table turns the calibration quantities into class-conditional action costs. The finite-action solver selects a plan, and the executor applies it without changing the host model interface.

## Mathematical map

The derivation is kept in a separate document so that the README can remain readable during installation and integration.

1. [Feedback residual and normalized gap](docs/derivation.md#1-two-readout-heads-one-measurable-question)
2. [A two-class example](docs/derivation.md#2-a-minimal-two-class-example)
3. [Classwise risk table](docs/derivation.md#3-the-classwise-risk-table)
4. [Budgeted discrete optimization](docs/derivation.md#4-budgeted-discrete-optimization)
5. [Pareto-frontier dynamic programming](docs/derivation.md#5-pareto-frontier-dynamic-programming)
6. [Fused execution and feature reuse](docs/derivation.md#6-why-fused-execution-reduces-work)

## What the plug-in measures

Long-tailed training is often adjusted through class weights, logit corrections or sampling. Those choices act on the classification objective. They do not, by themselves, describe how an incompletely adapted readout changes the next encoder update.

Let `ψ` denote the encoder parameters, `W_t` the current readout and `W̃_t` a FIT-only reference readout obtained by a short inner refinement. On the same batch, define

$$
 e_t = \nabla_{\psi} L(\psi_t,W_t)
      - \nabla_{\psi} L(\psi_t,\widetilde W_t).
$$

The residual `e_t` answers a narrow question. How much does the encoder update change when the current readout is replaced by the locally refined readout?

The normalized diagnostic is

$$
 \kappa_t =
 \frac{\lVert e_t\rVert_2}
 {\lVert\nabla_{\psi}L(\psi_t,\widetilde W_t)\rVert_2+\delta}.
$$

`κ_t` is a trajectory-level diagnostic. It is stored with the checkpoint, FIT batch, loss definition and calibration source that produced it. The plug-in does not treat this quantity as a population-level generalization bound.

## Installation and first run

```bash
git clone https://github.com/Oookla28/PV-MFAPS-Plugin.git
cd PV-MFAPS-Plugin
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the dependency-light smoke test first.

```bash
PYTHONPATH=src python scripts/smoke.py
```

The smoke creates a deterministic twelve-class calibration table, solves the finite-action problem, checks the legality rule `k=0 ⇒ α=0`, and prints the selected action plan with its solver certificate.

To build a plan from a host calibration file:

```bash
PYTHONPATH=src python scripts/build_action_plan.py \
  --calibration examples/calibration.json \
  --budget 0.50 \
  --output artifacts/action_plan.json
```

This command does not load images, checkpoints or an evaluation split. It is intended to verify the calibration-to-policy connection before a training job is started.

For an editable installation:

```bash
pip install -e .
```

## Public API

### Build an action plan

```python
from pv_mfaps_plugin import build_plan

calibration = {
    "A": [1.0, 1.2, 0.8, 1.4],
    "rho": [0.70, 0.80, 0.90, 0.75],
    "sigma": [0.10, 0.12, 0.08, 0.15],
    "n": [5, 8, 16, 32],
}

plan = build_plan(
    calibration,
    budget=0.50,
    k_grid=(0, 1, 2, 4),
    alpha_grid=(0.25,),
)

print(plan.k)
print(plan.alpha)
print(plan.certificate)
```

The calibration arrays use one fixed class order. `A` is the current residual scale, `rho` is the measured contraction proxy, `sigma` is the batch-variation proxy and `n` is the FIT support used for calibration. `plan.k[c]` gives the inner-loop steps for class `c`, while `plan.alpha[c]` gives its reference-feedback mixing coefficient.

The default action set is discrete. It contains `(0,0)`, `(1,0.25)`, `(2,0.25)` and `(4,0.25)`. A discrete plan is easy to inspect in a manifest and avoids hidden behaviour inside a continuous optimizer.

### Use the Fused execution hook

```python
from pv_mfaps_plugin import FusedExecutor, trace_summary

executor = FusedExecutor(
    encode=encoder,
    head=readout,
    plan=plan,
)

logits, alpha, trace = executor.forward(images, labels)
print(trace_summary(trace))
```

The trace separates two kinds of work. Logical counters describe the actions requested by the allocation policy. Physical counters describe the encoder forwards and backwards actually executed by the hardware. The host trainer remains responsible for the optimizer step.

### Use long-tail losses

```python
from pv_mfaps_plugin import balanced_softmax_loss, logit_adjusted_cross_entropy

loss = balanced_softmax_loss(logits, labels, class_counts)
# or
loss = logit_adjusted_cross_entropy(logits, labels, class_counts, tau=1.0)
```

When native and reference branches are compared, they should use the same loss contract. Otherwise the measured feedback residual combines readout refinement with a change in the classification objective.

## The allocation model

For class `c` and action `a=(k,α)`, the default risk proxy is

$$
U_c(k,\alpha)=(1-\alpha)^2 A_c\rho_c^{2k}+\alpha^2\frac{\sigma_c^2}{n_c}\xi^k.
$$

The first term describes the residual component that may decrease with additional inner steps. The second term records a stability cost that depends on batch variation, FIT support and the growth factor `ξ`.

The allocator solves

$$
\min_{a_1,\ldots,a_C}
\sum_{c=1}^{C}
\left(\frac{\mu}{C}+\nu_c\right)U_c(a_c)
\quad\text{subject to}\quad
\sum_{c=1}^{C}d_c(a_c)\le B.
$$

`d_c(a_c)` is the measured or supplied cost of the action and `B` is the available budget. `ν_c` can give additional weight to a protected tail group. The full derivation, including a worked example, is in [`docs/derivation.md`](docs/derivation.md).

## Integration points

| Hook | Input | Output | Host responsibility |
|---|---|---|---|
| `build_plan` | FIT calibration arrays and budget | `ActionPlan` | choose calibration data and schedule |
| `FusedExecutor.forward` | batch, labels, action plan | logits, per-sample α, trace | optimizer step and checkpointing |
| `balanced_softmax_loss` | logits, labels, counts | loss tensor | data loader and class-count contract |
| `evaluation.py` | frozen run directory | FIT summary | post-freeze evaluation split |

A host adapter should provide a FIT split, a separate evaluation split, a stable class order, a deterministic reference probe and measured native/reference costs. Each arm and seed should write to an isolated output directory. Checkpoints should be frozen before the evaluation entrypoint reads the evaluation split.

## Evaluation entrypoint

The compact evaluation helper aggregates executed FIT artifacts.

```bash
PYTHONPATH=src python -m pv_mfaps_plugin.evaluation \
  --run-dir /path/to/run \
  --split fit \
  --output /path/to/fit_summary.json
```

The summary records Macro-F1, Tail-10 metrics, wall-clock and FLOPs when those fields are present in the host artifacts. A host-specific held-out evaluator should remain in the host project so that its freeze and one-time-read protocol stays close to the data contract.

## Reference measurements from the source project

The following values come from the source PlantSPD and CIFAR-LT execution contracts. They indicate the scale of the complete integration. The standalone package does not re-read those datasets.

| Contract | Reference measurement |
|---|---:|
| PlantSPD, PR-PV-MFAPS versus matched fixed-k | Macro-F1 `0.4841` versus `0.4553`; Tail-10 F1 `0.2976` versus `0.2659` |
| PlantSPD matched execution cost | maximum FLOPs ratio `0.8889`; maximum wall-clock ratio `0.9974` |
| CIFAR-100-LT Stage-TRR-10, 200 epochs | Macro-F1 `0.4221 ± 0.0064`; Tail-10 F1 `0.2389 ± 0.0110` |
| CIFAR-100-LT Fused/Legacy pair | state max delta `4.47 × 10⁻⁸`; encoder forwards `2,580` versus `54,180` in the paired 30-epoch run |

## Repository layout

```text
PV-MFAPS-Plugin/
├── README.md
├── LICENSE
├── CITATION.cff
├── pyproject.toml
├── requirements.txt
├── configs/plugin.yaml
├── docs/
│   ├── architecture_supplied.png
│   └── derivation.md
├── examples/calibration.json
├── scripts/
│   ├── build_action_plan.py
│   ├── equivalence_smoke.py
│   └── smoke.py
├── src/pv_mfaps_plugin/
│   ├── __init__.py
│   ├── allocator_core.py
│   ├── evaluation.py
│   ├── fused.py
│   ├── losses.py
│   └── policy.py
└── tests/test_allocator.py
```

## Failure modes worth checking first

- Increasing `k` does not guarantee a lower total risk. The stability term can become dominant for small-support classes.
- `κ_t` is a feedback diagnostic. It is not the same quantity as accuracy or a generalization error.
- The DP certificate is exact only for the supplied risk table, action grid and quantised cost problem.
- A lower FLOPs count does not guarantee the same reduction in wall-clock time. Data loading, synchronization and kernel launches must be timed separately.
- Legacy and Fused comparisons require independent model states, optimizer states and random seeds.

## Reproducibility

The package supports Python 3.10 or newer, NumPy 1.24 or newer and PyTorch 2.x. The source training contracts used NVIDIA CUDA GPUs, while the allocator smoke runs on CPU. A host project should record seeds, class ordering, calibration files, action grids, budget scale, cost model and source hashes in its own manifest.

This repository contains only the plug-in implementation, its mathematical documentation, minimal examples and executable checks. It does not contain images, checkpoints, server credentials or internal research logs.

## Citation and licence

Citation metadata is provided in [`CITATION.cff`](CITATION.cff). The code is released under the MIT License. Dataset and pretrained-weight terms remain those of their original providers.
