# 3D Spectral Fractional PINN

Code accompanying "3D Spectral Fractional PINNs for Coupled Nonlinear Klein–Gordon Systems with Caputo Memory" (Adil, Berdyshev, Birgebayev, Abdiramanov). A Chebyshev-spectral, hard-constrained Physics-Informed Neural Network for coupled fractional (Caputo-memory) nonlinear Klein–Gordon systems, with forward simulation (1D and 3D) and 3D inverse parameter discovery.

## Layout

- `1d/` — 1D forward solver (`model.py`, `train.py`), used for Experiment 1 (Table 1: 0.16%/0.46% relative L2 error).
- `3d/` — 3D forward solver (`spectral_utils.py`, `model_3d.py`, `train_3d.py`) and 3D inverse parameter discovery (`inverse_discovery_3d.py`), used for Experiments 2–3 (Tables 2–3). `driver_forward_seed.py` and `driver_inverse_seed.py` are the exact entry points used to produce the paper's multi-seed numbers, e.g.:
  ```bash
  python 3d/driver_forward_seed.py 0   # seed 0, forward (Table 2)
  python 3d/driver_inverse_seed.py 0   # seed 0, inverse (Table 3)
  ```
  Repeat for seeds `1` and `2` to reproduce the full mean ± std tables.
- `ablation/ablation_laplacian_speed.py` — the spectral-vs-Autograd Laplacian timing ablation (Table, Section "Ablation").

## Requirements

```
torch>=2.0
numpy
scipy
matplotlib
```

## Notes on scope

This repository contains the final, paper-matching configuration for each experiment. Several intermediate protocol variants for the inverse problem (slower co-adaptation, alternating block-coordinate updates, doubled step budgets, doubled sensor counts, a field-then-parameter bootstrap round, and grid-resolution sweeps) were tried and diagnosed during development; their outcomes are summarized quantitatively in the manuscript's Limitations section, but their driver scripts are not included here to keep this repository to the configuration actually reported. Get in touch with the corresponding author if you would like those as well.

## Citation

If you use this code, please cite the manuscript (details in the published version).
