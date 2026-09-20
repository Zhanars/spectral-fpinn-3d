# 3D Spectral Fractional PINN

Code accompanying **"3D Spectral Fractional PINNs for Coupled Nonlinear Klein–Gordon Systems with Caputo Memory"** (Adil, Berdyshev, Birgebayev, Abdiramanov).

## Abstract

We develop a Three-Dimensional Spectral Fractional Physics-Informed Neural Network (3D Spectral fPINN) for forward wave simulation and inverse parameter discovery in coupled systems of nonlinear Klein–Gordon equations with Caputo-type viscoelastic memory. Simulating multi-dimensional coupled fractional wave equations with disparate memory orders α ≠ β via classical numerical discretizations incurs prohibitive memory storage and quadratic temporal complexity O(N_t²). Conversely, conventional Physics-Informed Neural Networks (PINNs) applied to three-dimensional non-local problems suffer from extreme computational graph expansion in automatic differentiation (Autograd) and severe performance bottlenecks when differentiating the 3D spatial Laplacian Δu = u_xx + u_yy + u_zz. To resolve these challenges, the proposed 3D Spectral fPINN integrates three methodological contributions:

1. an **exact multiplicative 3D hard-constraint envelope**, enforcing homogeneous Dirichlet boundary conditions on all six faces of the bounding box and Cauchy initial data (u=0, u_t=0) to machine precision without penalty loss heuristics;
2. a **fast 3D Chebyshev spectral Laplacian** evaluated via tensor contractions across Chebyshev–Gauss–Lobatto (CGL) collocation nodes, entirely bypassing second-order spatial Autograd and reducing per-epoch training time by 76.2% (a 4.2× speedup) relative to a second-order-Autograd baseline on the same grid and network, measured directly;
3. a **vectorized, fully differentiable lower-triangular Toeplitz L1 quadrature operator** evaluating the non-local Caputo memory history in batch mode across all 3D spatial nodes simultaneously.

Three theoretical theorems, proved in detail (with an explicit, algebraically-derived Lipschitz constant for the coupled cubic nonlinearity), establish universal representation under the hard-constraint ansatz, hybrid operator truncation error bounds, and an energy-based residual-to-error stability estimate via the fractional Alikhanov inequality.

Numerical experiments, repeated across three independent random seeds, demonstrate that the 3D solver attains sub-percent-to-low-single-percent relative L² accuracy (1.10% ± 0.09% for u, 0.35% ± 0.04% for v) on a 97,875-node four-dimensional spatiotemporal grid in under three hours on standard CPU hardware. Furthermore, the framework solves the inverse system-identification problem, discovering hidden fractional relaxation orders (α, β) from noisy (1%) 3D sensor data; across three independent random seeds it recovers β to 8.4% ± 0.8% and α to 21.4% ± 8.2% mean relative error.

## Layout

- `1d/` — 1D forward solver (`model.py`, `train.py`), used for Experiment 1 (Table 1: 0.16%/0.46% relative L2 error).
- `3d/` — 3D forward solver (`spectral_utils.py`, `model_3d.py`, `train_3d.py`) and 3D inverse parameter discovery (`inverse_discovery_3d.py`), used for Experiments 2–3 (Table 2: forward; Table 4: inverse). `driver_forward_seed.py` and `driver_inverse_seed.py` are the exact entry points used to produce the paper's multi-seed numbers, e.g.:
  ```bash
  python 3d/driver_forward_seed.py 0   # seed 0, forward (Table 2)
  python 3d/driver_inverse_seed.py 0   # seed 0, inverse (Table 4)
  ```
  Repeat for seeds `1` and `2` to reproduce the full mean ± std tables.
- `ablation/ablation_laplacian_speed.py` — the spectral-vs-Autograd Laplacian timing ablation (Table 3).

## Requirements

```
torch>=2.0
numpy
scipy
matplotlib
```

For the Julia cross-validation code in `julia/`, see `julia/Project.toml` (Julia 1.11+; `julia --project=julia -e 'using Pkg; Pkg.instantiate()'` from the repo root installs the pinned dependency versions from `julia/Manifest.toml`).

## Independent Julia cross-validation

`julia/` contains a from-scratch Julia re-implementation of the forward (1D, 3D) and inverse (3D) solvers, written to cross-validate the PyTorch results in a second language/framework/autodiff stack (Zygote reverse-mode + a centered finite-difference stencil for the PDE's time-second-derivative, in place of nested automatic differentiation, which was tried and found unreliable in this framework combination -- see `julia/3d/Model3D.jl`'s docstring for specifics). Component-level correctness (spectral differentiation matrices, the Kronecker-sum Laplacian, the differentiable Caputo memory operator, and the loss gradient) was verified against the Python implementation and against finite-difference gradient checks (agreement to 1e-5..1e-8 relative error) before any training run.

Results so far:
- **1D forward** (full run, matching Experiment 1's config): relative L² error **0.10% (u) / 0.44% (v)**, vs. the paper's **0.16% / 0.46%** -- same order of magnitude, independently confirming Experiment 1.
- **3D forward** (reduced grid/epoch budget for wall-clock reasons: 11×11×11×21 points, 500 Adam epochs): loss and relative L² error decrease monotonically and land at the same order of magnitude as the Python code's own early/under-trained runs at comparable epoch counts (L² errors around several tens of percent, not yet the paper's sub-percent HQ-configuration numbers, which need a much larger epoch budget than this validation run used).
- **3D inverse** (reduced grid: 7×7×7×13 points, 1500 discovery steps, one seed): **β recovered to 2.3% error**; **α diverged to 114% error** -- reproducing, independently, the α-harder-to-identify-than-β asymmetry and α's occasional divergence documented in the manuscript's Limitations section, rather than contradicting it.

See the module docstrings (particularly `julia/1d/Model1D.jl` and `julia/3d/inverse_discovery_3d.jl`) for the AD-framework issues hit and worked around during this port: Zygote's built-in adjoint for `ForwardDiff.derivative` silently drops gradients through closure-captured parameters (wrong gradient, no error raised); and Optim.jl's L-BFGS through a flatten/unflatten + Zygote round trip hits a ChainRulesCore Tangent/Tuple interop error in this codebase, worked around with extra Adam epochs in both `train_3d.jl` and `inverse_discovery_3d.jl`.

## Notes on scope

This repository contains the final, paper-matching configuration for each experiment. Several intermediate protocol variants for the inverse problem (slower co-adaptation, alternating block-coordinate updates, doubled step budgets, doubled sensor counts, a field-then-parameter bootstrap round, and grid-resolution sweeps) were tried and diagnosed during development; their outcomes are summarized quantitatively in the manuscript's Limitations section, but their driver scripts are not included here to keep this repository to the configuration actually reported. Get in touch with the corresponding author if you would like those as well.

## Citation

If you use this code, please cite the manuscript (details in the published version).
