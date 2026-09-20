"""
Ablation: spectral Chebyshev Laplacian vs. second-order spatial Autograd,
on the exact grid/network configuration used in Experiment 2 of the paper
(15x15x15x29 grid, hidden_dim=128, num_layers=5), measuring per-epoch
wall-clock cost of a full forward+backward pass under Adam.

This directly measures the "backward-pass execution time reduced by over 80%"
/ "training accelerated by an order of magnitude" claims in the Abstract and
Introduction, which the original submission asserted without an in-paper
ablation (referee comment M1).
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import time
import torch
torch.set_num_threads(min(8, os.cpu_count() or 8))
import numpy as np
from scipy.special import gamma as scipy_gamma

from model_3d import SpectralFPINN3D
from train_3d import compute_3d_analytical_sources


def compute_loss_autograd_laplacian(model, f1_exact, f2_exact):
    """
    Identical to SpectralFPINN3D.compute_loss, except the 3D spatial Laplacian
    is computed via second-order Autograd (torch.autograd.grad with
    create_graph=True, twice per spatial dimension) instead of the spectral
    Kronecker-sum tensor contraction -- i.e., what a standard (non-spectral)
    3D fPINN would do.
    """
    t_4d = model.t_grid.view(-1, 1, 1, 1).expand(-1, model.Nx + 1, model.Ny + 1, model.Nz + 1).clone()
    x_4d = model.x_cgl.view(1, -1, 1, 1).expand(model.Nt + 1, -1, model.Ny + 1, model.Nz + 1).clone()
    y_4d = model.y_cgl.view(1, 1, -1, 1).expand(model.Nt + 1, model.Nx + 1, -1, model.Nz + 1).clone()
    z_4d = model.z_cgl.view(1, 1, 1, -1).expand(model.Nt + 1, model.Nx + 1, model.Ny + 1, -1).clone()
    t_4d.requires_grad_(True)
    x_4d.requires_grad_(True)
    y_4d.requires_grad_(True)
    z_4d.requires_grad_(True)

    u, v = model.forward(x_4d, y_4d, z_4d, t_4d)
    ones = torch.ones_like(u)

    u_t = torch.autograd.grad(u, t_4d, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    u_tt = torch.autograd.grad(u_t, t_4d, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    v_t = torch.autograd.grad(v, t_4d, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    v_tt = torch.autograd.grad(v_t, t_4d, grad_outputs=ones, create_graph=True, retain_graph=True)[0]

    # Second-order spatial Autograd for the 3D Laplacian (the baseline this
    # paper's spectral Laplacian replaces):
    u_x = torch.autograd.grad(u, x_4d, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_4d, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    u_y = torch.autograd.grad(u, y_4d, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y_4d, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    u_z = torch.autograd.grad(u, z_4d, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    u_zz = torch.autograd.grad(u_z, z_4d, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    lap_u = u_xx + u_yy + u_zz

    v_x = torch.autograd.grad(v, x_4d, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    v_xx = torch.autograd.grad(v_x, x_4d, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    v_y = torch.autograd.grad(v, y_4d, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y_4d, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    v_z = torch.autograd.grad(v, z_4d, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    v_zz = torch.autograd.grad(v_z, z_4d, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    lap_v = v_xx + v_yy + v_zz

    cap_u, cap_v = model.compute_fractional_derivatives(u, v)

    u_int, v_int = u[1:], v[1:]
    u_tt_int, v_tt_int = u_tt[1:], v_tt[1:]
    lap_u_int, lap_v_int = lap_u[1:], lap_v[1:]
    f1_int, f2_int = f1_exact[1:], f2_exact[1:]

    res_u = (u_tt_int - (model.c1 ** 2) * lap_u_int + model.sigma1 * cap_u
             + model.m1 * u_int + model.k1 * (u_int ** 3) + model.eta * u_int * (v_int ** 2) - f1_int)
    res_v = (v_tt_int - (model.c2 ** 2) * lap_v_int + model.sigma2 * cap_v
             + model.m2 * v_int + model.k2 * (v_int ** 3) + model.eta * v_int * (u_int ** 2) - f2_int)

    loss = torch.mean(res_u ** 2) + torch.mean(res_v ** 2)
    return loss


def benchmark(n_epochs, use_autograd, Nx=14, Ny=14, Nz=14, Nt=28, hidden_dim=128, num_layers=5):
    u_exact, v_exact, f1, f2, x_cgl, y_cgl, z_cgl = compute_3d_analytical_sources(
        Nx=Nx, Ny=Ny, Nz=Nz, Nt=Nt
    )
    model = SpectralFPINN3D(Nx=Nx, Ny=Ny, Nz=Nz, Nt=Nt, alpha=0.4, beta=0.7,
                             hidden_dim=hidden_dim, num_layers=num_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

    # Warm-up (excluded from timing): first call compiles/allocates buffers.
    optimizer.zero_grad()
    if use_autograd:
        loss = compute_loss_autograd_laplacian(model, f1, f2)
    else:
        loss, _, _ = model.compute_loss(f1, f2)
    loss.backward()
    optimizer.step()

    t0 = time.time()
    for _ in range(n_epochs):
        optimizer.zero_grad()
        if use_autograd:
            loss = compute_loss_autograd_laplacian(model, f1, f2)
        else:
            loss, _, _ = model.compute_loss(f1, f2)
        loss.backward()
        optimizer.step()
    elapsed = time.time() - t0
    return elapsed / n_epochs


if __name__ == "__main__":
    Nx = Ny = Nz = 14
    Nt = 28
    n_pts = (Nx + 1) * (Ny + 1) * (Nz + 1) * (Nt + 1)
    print(f"Grid: {Nx+1}x{Ny+1}x{Nz+1}x{Nt+1} = {n_pts:,} points, hidden=128, layers=5")
    print("Timing per-epoch cost of a full forward+backward pass under Adam...\n")

    n_epochs_spectral = 20
    t_spectral = benchmark(n_epochs_spectral, use_autograd=False, Nx=Nx, Ny=Ny, Nz=Nz, Nt=Nt)
    print(f"Spectral Laplacian:  {t_spectral*1000:.1f} ms/epoch  ({n_epochs_spectral} epochs)")

    n_epochs_autograd = 10
    t_autograd = benchmark(n_epochs_autograd, use_autograd=True, Nx=Nx, Ny=Ny, Nz=Nz, Nt=Nt)
    print(f"Autograd Laplacian:  {t_autograd*1000:.1f} ms/epoch  ({n_epochs_autograd} epochs)")

    speedup = t_autograd / t_spectral
    reduction = (1 - t_spectral / t_autograd) * 100
    print(f"\nSpeedup: {speedup:.2f}x")
    print(f"Per-epoch time reduction: {reduction:.1f}%")
