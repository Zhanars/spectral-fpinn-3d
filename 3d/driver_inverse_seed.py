"""Final configuration for Experiment 3 (inverse), matching the numbers
reported in the manuscript (Table 3: alpha 21.4%+-8.2%, beta 8.4%+-0.76%
across seeds 0,1,2). Uses the root-cause-fixed protocol: memory-free Stage
A/B field pretraining (make_manufactured_sources_3d_nomemory /
compute_pde_loss_no_memory) so the field cannot become biased toward the
wrong pinned initial guess, then a fully-frozen-network Stage C that
optimizes only (alpha, beta) against the full Caputo-memory PDE residual on
the FULL time grid for a fixed 4000 steps, with no final LBFGS param jump.

Two further ideas were tried and explicitly rejected (see the comment above
Stage C in inverse_discovery_3d.py for the full diagnosis): doubling the
step budget to 8000, and a train/validation split of the time grid with
early stopping by held-out residual. Neither improved on the numbers below;
both are documented, not deployed."""
import sys
from inverse_discovery_3d import run_parameter_discovery_3d

seed = int(sys.argv[1])
run_parameter_discovery_3d(
    num_sensors=600, noise_level=0.01,
    epochs=6000, lr_nn=2e-3, lr_param=1.5e-3,
    Nx=8, Ny=8, Nz=8, Nt=20, hidden_dim=64, num_layers=4,
    warmup_epochs=1000, field_refine_epochs=1000,
    lbfgs_field_iter=100, lbfgs_param_iter=0,
    save_dir=f"results_inverse_3d_seed{seed}",
    fig_suffix=f"_seed{seed}", seed=seed,
    finetune_nn_lr_frac=0.002,
)
