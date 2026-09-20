"""
Training, verification, and visualization for 3D Spectral Fractional PINN (4D Spatiotemporal: x, y, z, t).
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import time
import torch
# The 3D collocation tensors here are small (a few thousand points); PyTorch's
# default CPU thread pool (one per physical core) causes severe oversubscription
# overhead on ops this size. Capping the intra-op thread count avoids a >20x
# slowdown observed on many-core machines.
torch.set_num_threads(min(8, os.cpu_count() or 8))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import gamma

from model_3d import SpectralFPINN3D


def compute_3d_analytical_sources(Nx=8, Ny=8, Nz=8, Nt=16,
                                  Lx=1.0, Ly=1.0, Lz=1.0, T=1.0,
                                  alpha=0.4, beta=0.7, c1=1.0, c2=1.0,
                                  m1=1.0, m2=1.0, k1=1.0, k2=1.0,
                                  sigma1=1.0, sigma2=1.0, eta=1.0):
    """
    Computes exact 3D manufactured solutions and RHS sources:
    u_exact = t^3 * sin(pi*x) * sin(pi*y) * sin(pi*z)
    v_exact = t^3 * sin(2*pi*x) * sin(pi*y) * sin(pi*z)
    """
    i = np.arange(Nx + 1)
    j = np.arange(Ny + 1)
    k = np.arange(Nz + 1)
    x_cgl = (Lx / 2.0) * (1.0 - np.cos(i * np.pi / Nx))
    y_cgl = (Ly / 2.0) * (1.0 - np.cos(j * np.pi / Ny))
    z_cgl = (Lz / 2.0) * (1.0 - np.cos(k * np.pi / Nz))
    t_grid = np.linspace(0.0, T, Nt + 1)

    T_4d, X_4d, Y_4d, Z_4d = np.meshgrid(t_grid, x_cgl, y_cgl, z_cgl, indexing='ij')

    sp_u = np.sin(np.pi * X_4d) * np.sin(np.pi * Y_4d) * np.sin(np.pi * Z_4d)
    sp_v = np.sin(2.0 * np.pi * X_4d) * np.sin(np.pi * Y_4d) * np.sin(np.pi * Z_4d)

    u_exact = (T_4d ** 3) * sp_u
    v_exact = (T_4d ** 3) * sp_v

    u_tt = 6.0 * T_4d * sp_u
    v_tt = 6.0 * T_4d * sp_v

    lap_u = -3.0 * (np.pi ** 2) * u_exact
    lap_v = -6.0 * (np.pi ** 2) * v_exact

    cap_u = (6.0 / gamma(4.0 - alpha)) * (T_4d ** (3.0 - alpha)) * sp_u
    cap_v = (6.0 / gamma(4.0 - beta)) * (T_4d ** (3.0 - beta)) * sp_v

    f1 = (u_tt - (c1 ** 2) * lap_u + sigma1 * cap_u
          + m1 * u_exact + k1 * (u_exact ** 3) + eta * u_exact * (v_exact ** 2))

    f2 = (v_tt - (c2 ** 2) * lap_v + sigma2 * cap_v
          + m2 * v_exact + k2 * (v_exact ** 3) + eta * v_exact * (u_exact ** 2))

    return (torch.tensor(u_exact, dtype=torch.float32),
            torch.tensor(v_exact, dtype=torch.float32),
            torch.tensor(f1, dtype=torch.float32),
            torch.tensor(f2, dtype=torch.float32),
            x_cgl, y_cgl, z_cgl)


def train_3d_spectral_fpinn(epochs=350, lr=2e-3, Nx=8, Ny=8, Nz=8, Nt=16,
                             hidden_dim=64, num_layers=4, lbfgs_max_iter=25,
                             fig_suffix="", seed=0):
    print("=" * 75)
    print("3D Spectral Fractional PINN (4D Spatiotemporal Domain: x, y, z, t)")
    print("=" * 75)

    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f"Random seed: {seed}")

    total_collocation_points = (Nx + 1) * (Ny + 1) * (Nz + 1) * (Nt + 1)
    print(f"3D Collocation Grid: {Nx+1} x {Ny+1} x {Nz+1} spatial nodes x {Nt+1} time steps")
    print(f"Total 4D Collocation Nodes: {total_collocation_points:,}")

    u_exact, v_exact, f1, f2, x_cgl, y_cgl, z_cgl = compute_3d_analytical_sources(
        Nx=Nx, Ny=Ny, Nz=Nz, Nt=Nt
    )

    model = SpectralFPINN3D(Nx=Nx, Ny=Ny, Nz=Nz, Nt=Nt,
                            alpha=0.4, beta=0.7, hidden_dim=hidden_dim, num_layers=num_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {'epoch': [], 'loss': [], 'l2_u': [], 'l2_v': []}
    print_every = max(50, epochs // 40)

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        loss, u_pred, v_pred = model.compute_loss(f1, f2)
        loss.backward()
        optimizer.step()

        if epoch % print_every == 0 or epoch == 1:
            with torch.no_grad():
                l2_u = torch.norm(u_pred - u_exact) / torch.norm(u_exact)
                l2_v = torch.norm(v_pred - v_exact) / torch.norm(v_exact)
            history['epoch'].append(epoch)
            history['loss'].append(loss.item())
            history['l2_u'].append(l2_u.item())
            history['l2_v'].append(l2_v.item())
            print(f"Epoch [{epoch:4d}/{epochs:4d}] | Loss: {loss.item():.4e} | Rel L2(u): {l2_u.item():.4e} | Rel L2(v): {l2_v.item():.4e}")

    # L-BFGS refinement
    print("\n--- Second-Order L-BFGS Refinement ---")
    lbfgs = torch.optim.LBFGS(model.parameters(), lr=0.5, max_iter=lbfgs_max_iter, history_size=50, line_search_fn="strong_wolfe")
    
    def closure():
        lbfgs.zero_grad()
        l, _, _ = model.compute_loss(f1, f2)
        l.backward()
        return l
        
    lbfgs.step(closure)
    elapsed = time.time() - start_time

    # Final evaluation
    final_loss, u_pred, v_pred = model.compute_loss(f1, f2)
    with torch.no_grad():
        final_loss_val = final_loss.item()
        final_l2_u = (torch.norm(u_pred - u_exact) / torch.norm(u_exact)).item()
        final_l2_v = (torch.norm(v_pred - v_exact) / torch.norm(v_exact)).item()

    print("=" * 75)
    print(f"3D Training Complete in {elapsed:.2f} seconds!")
    print(f"Final 3D PDE Residual Loss: {final_loss_val:.4e}")
    print(f"Final 3D Relative L2 Error u: {final_l2_u * 100:.2f}%")
    print(f"Final 3D Relative L2 Error v: {final_l2_v * 100:.2f}%")
    print("=" * 75)

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), f"models/model_3d_weights{fig_suffix}.pt")

    os.makedirs("results_3d", exist_ok=True)
    with open(f"results_3d/metrics_3d{fig_suffix}.txt", "w") as fmet:
        fmet.write(f"seed={seed}\n")
        fmet.write(f"final_loss={final_loss_val:.6e}\n")
        fmet.write(f"rel_l2_u={final_l2_u*100:.4f}\n")
        fmet.write(f"rel_l2_v={final_l2_v*100:.4f}\n")
        fmet.write(f"wall_time_s={elapsed:.2f}\n")

    os.makedirs("figures", exist_ok=True)

    # Plot 1: 3D Convergence history
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)
    ax1.semilogy(history['epoch'], history['loss'], 'b-o', lw=1.8, ms=4)
    ax1.set_title("3D Spectral fPINN: Residual Loss", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Loss $\\mathcal{L}_{\\mathrm{PDE}}$", fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.semilogy(history['epoch'], history['l2_u'], 'r-s', lw=1.8, ms=4, label="$L^2$ error $u$")
    ax2.semilogy(history['epoch'], history['l2_v'], 'g-^', lw=1.8, ms=4, label="$L^2$ error $v$")
    ax2.set_title("3D Relative $L^2$ Solution Error", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Relative $L^2$ Error", fontsize=10)
    ax2.legend(frameon=True)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"figures/convergence_3d{fig_suffix}.png")
    plt.close()
    print(f"Saved 3D convergence figure to figures/convergence_3d{fig_suffix}.png")

    # Plot 2: 3D Spatial cross-sections at z = 0.5 (middle index k = Nz//2) and t = T
    k_mid = Nz // 2
    u_exact_slice = u_exact[-1, :, :, k_mid].numpy()
    u_pred_slice = u_pred[-1, :, :, k_mid].detach().numpy()
    err_slice = np.abs(u_pred_slice - u_exact_slice)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), dpi=300)
    im0 = axes[0].contourf(y_cgl, x_cgl, u_exact_slice, levels=30, cmap='viridis')
    axes[0].set_title(f"Exact $u(x, y, z={z_cgl[k_mid]:.2f}, T=1)$", fontsize=10, fontweight='bold')
    axes[0].set_xlabel("$y$")
    axes[0].set_ylabel("$x$")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].contourf(y_cgl, x_cgl, u_pred_slice, levels=30, cmap='viridis')
    axes[1].set_title(f"3D fPINN $\\hat{{u}}(x, y, z={z_cgl[k_mid]:.2f}, T=1)$", fontsize=10, fontweight='bold')
    axes[1].set_xlabel("$y$")
    axes[1].set_ylabel("$x$")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].contourf(y_cgl, x_cgl, err_slice, levels=30, cmap='magma')
    axes[2].set_title("Absolute Pointwise Error $|\\hat{u} - u|$", fontsize=10, fontweight='bold')
    axes[2].set_xlabel("$y$")
    axes[2].set_ylabel("$x$")
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.savefig(f"figures/spatial_3d_slices{fig_suffix}.png")
    plt.close()
    print(f"Saved 3D slice figure to figures/spatial_3d_slices{fig_suffix}.png")

    return final_l2_u, final_l2_v, elapsed


if __name__ == "__main__":
    train_3d_spectral_fpinn(epochs=350)
