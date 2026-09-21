# 3D Spectral Fractional PINN

Code accompanying **"3D Spectral Fractional PINNs for Coupled Nonlinear Klein–Gordon Systems with Caputo Memory"** (Adil, Berdyshev, Birgebayev, Abdiramanov).

**Implementation note:** this repository was originally a PyTorch implementation; it has been replaced with an independent from-scratch **Julia** re-implementation (below), written and numerically cross-validated against the original PyTorch code before the Python code was removed. The Julia code is the current, actively-maintained implementation.

## Abstract

We develop a Three-Dimensional Spectral Fractional Physics-Informed Neural Network (3D Spectral fPINN) for forward wave simulation and inverse parameter discovery in coupled systems of nonlinear Klein–Gordon equations with Caputo-type viscoelastic memory. Simulating multi-dimensional coupled fractional wave equations with disparate memory orders α ≠ β via classical numerical discretizations incurs prohibitive memory storage and quadratic temporal complexity O(N_t²). Conversely, conventional Physics-Informed Neural Networks (PINNs) applied to three-dimensional non-local problems suffer from extreme computational graph expansion in automatic differentiation (Autograd) and severe performance bottlenecks when differentiating the 3D spatial Laplacian Δu = u_xx + u_yy + u_zz. To resolve these challenges, the proposed 3D Spectral fPINN integrates three methodological contributions:

1. an **exact multiplicative 3D hard-constraint envelope**, enforcing homogeneous Dirichlet boundary conditions on all six faces of the bounding box and Cauchy initial data (u=0, u_t=0) to machine precision without penalty loss heuristics;
2. a **fast 3D Chebyshev spectral Laplacian** evaluated via tensor contractions across Chebyshev–Gauss–Lobatto (CGL) collocation nodes, entirely bypassing second-order spatial Autograd;
3. a **vectorized, fully differentiable lower-triangular Toeplitz L1 quadrature operator** evaluating the non-local Caputo memory history in batch mode across all 3D spatial nodes simultaneously.

Three theoretical theorems, proved in detail (with an explicit, algebraically-derived Lipschitz constant for the coupled cubic nonlinearity), establish universal representation under the hard-constraint ansatz, hybrid operator truncation error bounds, and an energy-based residual-to-error stability estimate via the fractional Alikhanov inequality.

The framework solves both the forward wave-simulation problem and the inverse system-identification problem (discovering hidden fractional relaxation orders (α, β) from noisy 3D sensor data). See "Results" below for the current Julia-implementation numbers.

## Layout

- `julia/1d/` — 1D forward solver (`Model1D.jl`, `train_1d.jl`), Experiment 1.
- `julia/3d/` — 3D forward solver (`SpectralUtils.jl`, `Model3D.jl`, `train_3d.jl`) and 3D inverse parameter discovery (`inverse_discovery_3d.jl`), Experiments 2–3.
- `julia/ablation/` — spectral-vs-finite-difference Laplacian timing comparison.
- `julia/Project.toml` / `julia/Manifest.toml` — pinned dependencies (Julia 1.11+).

Run from the repo root:
```julia
using Pkg; Pkg.activate("julia"); Pkg.instantiate()
include("julia/1d/train_1d.jl"); train_spectral_fpinn_1d()
include("julia/3d/train_3d.jl"); train_3d_spectral_fpinn()
include("julia/3d/inverse_discovery_3d.jl"); run_parameter_discovery_3d()
```

## Requirements

Julia 1.11+; see `julia/Project.toml` for package dependencies (ForwardDiff, Zygote, Optimisers, Optim, SpecialFunctions). Install with:
```bash
julia --project=julia -e 'using Pkg; Pkg.instantiate()'
```

## Results

AD strategy: Zygote (reverse-mode) for the network-weight/parameter gradient, and a centered finite-difference stencil (not nested automatic differentiation) for the PDE's second time derivative -- see `julia/1d/Model1D.jl`'s docstring for why nested ForwardDiff/Zygote composition was tried and found unreliable in this framework combination. Component-level correctness (spectral differentiation matrices, the Kronecker-sum Laplacian, the differentiable Caputo memory operator, and loss gradients) was verified against finite-difference gradient checks (agreement to 1e-5..1e-8 relative error) before any training run.

- **1D forward** (Experiment 1 config): relative L² error **0.10% (u) / 0.44% (v)**.
- **3D forward** (10×10×10×20 grid, matching the manuscript's protocol): relative L² error **1.29% (u) / 0.60% (v)**. (A first run using an overly aggressive learning-rate decay during the extra refinement phase appeared to plateau at 5.56%/2.34% after only a few hundred refinement epochs; the decay turned out to collapse the step size to near-zero well before the phase ended, freezing the weights rather than converging them. Switching to a bounded cosine decay and continuing training from a checkpoint recovered a substantially better result.)
- **3D inverse** (8×8×8×20 grid, 600 sensors, matching the manuscript's protocol): **alpha discovered=0.822 (true=0.400, error=105.6%)**, **beta discovered=0.742 (true=0.700, error=6.0%)** -- reproduces the manuscript's qualitative finding that alpha is substantially harder to identify from noisy 3D sensor data than beta.

See the module docstrings for the AD-framework issues hit and worked around during development: Zygote's built-in adjoint for `ForwardDiff.derivative` silently drops gradients through closure-captured parameters (wrong gradient, no error raised); and Optim.jl's L-BFGS through a flatten/unflatten + Zygote round trip hits a ChainRulesCore Tangent/Tuple interop error in this codebase, worked around with extra Adam epochs in both `train_3d.jl` and `inverse_discovery_3d.jl`.

## Notes on scope

Several intermediate inverse-problem protocol variants (slower co-adaptation, alternating block-coordinate updates, doubled step budgets, doubled sensor counts, a field-then-parameter bootstrap round, grid-resolution sweeps) were tried and diagnosed during the original PyTorch development; their outcomes are summarized quantitatively in the manuscript's Limitations section. Only the final protocol is implemented here.

## Citation

If you use this code, please cite the manuscript (details in the published version).
