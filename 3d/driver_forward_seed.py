"""Runs the Experiment-2 (forward) HQ configuration for one explicit seed.
Used to collect independent repeated trials for multi-seed mean+-std reporting
(referee comment M6)."""
import sys
from train_3d import train_3d_spectral_fpinn

seed = int(sys.argv[1])
train_3d_spectral_fpinn(
    epochs=2500, lr=2e-3, Nx=14, Ny=14, Nz=14, Nt=28,
    hidden_dim=128, num_layers=5, lbfgs_max_iter=300,
    fig_suffix=f"_seed{seed}", seed=seed,
)
