"""
Training and Validation script for Spectral fPINN (SPINN).
Solves the coupled nonlinear Klein-Gordon system with Caputo fractional memory.
Validates against the exact manufactured solution from the journal paper:
    u_exact(x, t) = t^3 * sin(pi * x)
    v_exact(x, t) = t^3 * sin(2 * pi * x)
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.special import gamma as scipy_gamma

from model import SpectralFPINN


def make_manufactured_sources(alpha, beta, c1=1.0, c2=1.0, m1=1.0, m2=1.0,
                              k1=1.0, k2=1.0, sigma1=1.0, sigma2=1.0, eta=1.0):
    """
    Returns exact analytical source terms f1(x, t) and f2(x, t) for
    u = t^3 * sin(pi * x), v = t^3 * sin(2 * pi * x).
    """
    gamma_factor_u = 6.0 / scipy_gamma(4.0 - alpha)
    gamma_factor_v = 6.0 / scipy_gamma(4.0 - beta)
    
    def f1_func(x, t):
        # u = t^3 * sin(pi * x)
        sin_pix = torch.sin(np.pi * x)
        sin_2pix = torch.sin(2.0 * np.pi * x)
        
        u = (t ** 3) * sin_pix
        v = (t ** 3) * sin_2pix
        
        u_tt = 6.0 * t * sin_pix
        u_xx = - (np.pi ** 2) * (t ** 3) * sin_pix
        caputo_u = gamma_factor_u * (t ** (3.0 - alpha)) * sin_pix
        
        f1 = (u_tt 
              - (c1 ** 2) * u_xx 
              + sigma1 * caputo_u 
              + m1 * u 
              + k1 * (u ** 3) 
              + eta * u * (v ** 2))
        return f1

    def f2_func(x, t):
        sin_pix = torch.sin(np.pi * x)
        sin_2pix = torch.sin(2.0 * np.pi * x)
        
        u = (t ** 3) * sin_pix
        v = (t ** 3) * sin_2pix
        
        v_tt = 6.0 * t * sin_2pix
        v_xx = - 4.0 * (np.pi ** 2) * (t ** 3) * sin_2pix
        caputo_v = gamma_factor_v * (t ** (3.0 - beta)) * sin_2pix
        
        f2 = (v_tt 
              - (c2 ** 2) * v_xx 
              + sigma2 * caputo_v 
              + m2 * v 
              + k2 * (v ** 3) 
              + eta * v * (u ** 2))
        return f2

    return f1_func, f2_func


def compute_relative_l2_error(pred, exact):
    diff_norm = torch.sqrt(torch.sum((pred - exact) ** 2))
    exact_norm = torch.sqrt(torch.sum(exact ** 2))
    return (diff_norm / (exact_norm + 1e-12)).item()


def train_spectral_fpinn(N=24, N_t=40, T=1.0, L=1.0,
                         alpha=0.4, beta=0.7,
                         hidden_dim=64, num_layers=4,
                         adam_epochs=2000, lbfgs_max_iter=500,
                         lr=2e-3, save_dir="results"):
    
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== Starting Spectral fPINN Training on {device} ===")
    print(f"Config: N={N} (CGL nodes), N_t={N_t} (time steps), T={T}, L={L}")
    print(f"Fractional Orders: alpha={alpha}, beta={beta}")
    
    # Instantiate model
    model = SpectralFPINN(N=N, N_t=N_t, T=T, L=L,
                          hidden_dim=hidden_dim, num_layers=num_layers,
                          alpha=alpha, beta=beta).to(device)
    
    f1_func, f2_func = make_manufactured_sources(alpha=alpha, beta=beta)
    
    # Ground truth exact solutions on the grid
    x_grid = model.x_cgl.unsqueeze(0).expand(N_t + 1, -1)
    t_grid = model.t_grid.unsqueeze(1).expand(-1, N + 1)
    u_exact = (t_grid ** 3) * torch.sin(np.pi * x_grid)
    v_exact = (t_grid ** 3) * torch.sin(2.0 * np.pi * x_grid)
    
    # ------------------------------------------------------------
    # Stage 1: Adam Optimizer
    # ------------------------------------------------------------
    print("\n--- Stage 1: Adam Warm-up Optimization ---")
    optimizer_adam = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer_adam, gamma=0.999)
    
    loss_history = []
    loss_u_history = []
    loss_v_history = []
    l2_u_history = []
    l2_v_history = []
    
    start_time = time.time()
    
    for epoch in range(1, adam_epochs + 1):
        optimizer_adam.zero_grad()
        loss, loss_u, loss_v, (u_pred, v_pred) = model.compute_pde_residuals(f1_func, f2_func)
        loss.backward()
        optimizer_adam.step()
        scheduler.step()
        
        loss_val = loss.item()
        loss_history.append(loss_val)
        loss_u_history.append(loss_u.item())
        loss_v_history.append(loss_v.item())
        
        with torch.no_grad():
            err_u = compute_relative_l2_error(u_pred, u_exact)
            err_v = compute_relative_l2_error(v_pred, v_exact)
            l2_u_history.append(err_u)
            l2_v_history.append(err_v)
            
        if epoch % 200 == 0 or epoch == 1:
            print(f"Adam Epoch {epoch:5d}/{adam_epochs} | Loss: {loss_val:.4e} (u: {loss_u.item():.2e}, v: {loss_v.item():.2e}) | L2 u: {err_u:.4e} | L2 v: {err_v:.4e}")
            
    adam_time = time.time() - start_time
    print(f"Adam finished in {adam_time:.2f}s.")
    
    # ------------------------------------------------------------
    # Stage 2: L-BFGS Optimizer (Fine-Tuning)
    # ------------------------------------------------------------
    if lbfgs_max_iter > 0:
        print("\n--- Stage 2: L-BFGS Fine-Tuning ---")
        optimizer_lbfgs = optim.LBFGS(
            model.parameters(),
            max_iter=lbfgs_max_iter,
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
            history_size=50,
            line_search_fn="strong_wolfe"
        )
        
        step = [0]
        def closure():
            optimizer_lbfgs.zero_grad()
            loss, loss_u, loss_v, (u_pred, v_pred) = model.compute_pde_residuals(f1_func, f2_func)
            loss.backward()
            step[0] += 1
            loss_history.append(loss.item())
            loss_u_history.append(loss_u.item())
            loss_v_history.append(loss_v.item())
            with torch.no_grad():
                err_u = compute_relative_l2_error(u_pred, u_exact)
                err_v = compute_relative_l2_error(v_pred, v_exact)
                l2_u_history.append(err_u)
                l2_v_history.append(err_v)
                
            if step[0] % 50 == 0 or step[0] == 1:
                print(f"L-BFGS Step {step[0]:4d} | Loss: {loss.item():.4e} | L2 u: {err_u:.4e} | L2 v: {err_v:.4e}")
            return loss

        lbfgs_start = time.time()
        optimizer_lbfgs.step(closure)
        lbfgs_time = time.time() - lbfgs_start
        print(f"L-BFGS finished in {lbfgs_time:.2f}s ({step[0]} iterations).")
        
    total_time = time.time() - start_time
    print(f"\nTotal Training Time: {total_time:.2f}s")
    
    # ------------------------------------------------------------
    # Final Evaluation & Visualization
    # ------------------------------------------------------------
    with torch.no_grad():
        u_final, v_final, _, _ = model.evaluate_grid()
        final_l2_u = compute_relative_l2_error(u_final, u_exact)
        final_l2_v = compute_relative_l2_error(v_final, v_exact)
        max_err_u = torch.max(torch.abs(u_final - u_exact)).item()
        max_err_v = torch.max(torch.abs(v_final - v_exact)).item()
        
    print(f"\n=== Final Accuracy Summary ===")
    print(f"u-component: Relative L2 Error = {final_l2_u:.4e}, Max Absolute Error = {max_err_u:.4e}")
    print(f"v-component: Relative L2 Error = {final_l2_v:.4e}, Max Absolute Error = {max_err_v:.4e}")
    
    # Save Model Weights
    torch.save(model.state_dict(), os.path.join(save_dir, "spectral_fpinn_model.pth"))
    
    # ------------------------------------------------------------
    # Generate Plots
    # ------------------------------------------------------------
    print("\nGenerating diagnostic and solution plots...")
    
    # 1. Training Convergence Curve
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.semilogy(loss_history, label="Total PDE Loss", color="black")
    plt.semilogy(loss_u_history, label="Loss u", color="blue", alpha=0.7)
    plt.semilogy(loss_v_history, label="Loss v", color="red", alpha=0.7)
    plt.xlabel("Iteration / Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("Spectral fPINN Loss History")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.semilogy(l2_u_history, label="Rel L2 Error u", color="blue")
    plt.semilogy(l2_v_history, label="Rel L2 Error v", color="red")
    plt.xlabel("Iteration / Epoch")
    plt.ylabel("Relative L2 Error")
    plt.title("Error Convergence History")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "convergence_history.png"), dpi=300)
    plt.close()
    
    # 2. Final Time Slice (t = T = 1.0)
    x_np = model.x_cgl.cpu().numpy()
    u_pred_T = u_final[-1].cpu().numpy()
    v_pred_T = v_final[-1].cpu().numpy()
    u_exact_T = u_exact[-1].cpu().numpy()
    v_exact_T = v_exact[-1].cpu().numpy()
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x_np, u_exact_T, 'k-', lw=2, label="Exact $u(x, 1)$")
    plt.plot(x_np, u_pred_T, 'b--', lw=2, marker='o', markersize=4, label=r"Spectral fPINN $\hat{u}$")
    plt.xlabel("Spatial coordinate $x$")
    plt.ylabel("Displacement $u$")
    plt.title(f"$u(x, T=1)$ [$\\alpha={alpha}$, Rel L2={final_l2_u:.2e}]")
    plt.grid(True, ls="--", alpha=0.5)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(x_np, v_exact_T, 'k-', lw=2, label="Exact $v(x, 1)$")
    plt.plot(x_np, v_pred_T, 'r--', lw=2, marker='s', markersize=4, label=r"Spectral fPINN $\hat{v}$")
    plt.xlabel("Spatial coordinate $x$")
    plt.ylabel("Displacement $v$")
    plt.title(f"$v(x, T=1)$ [$\\beta={beta}$, Rel L2={final_l2_v:.2e}]")
    plt.grid(True, ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "solution_slice_T.png"), dpi=300)
    plt.close()
    
    # 3. 2D Spatiotemporal Contour Maps
    t_np = model.t_grid.cpu().numpy()
    X_mesh, T_mesh = np.meshgrid(x_np, t_np)
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    
    # u component
    c0 = axes[0, 0].contourf(X_mesh, T_mesh, u_exact.cpu().numpy(), 40, cmap="viridis")
    axes[0, 0].set_title("Exact $u(x, t)$")
    plt.colorbar(c0, ax=axes[0, 0])
    
    c1 = axes[0, 1].contourf(X_mesh, T_mesh, u_final.cpu().numpy(), 40, cmap="viridis")
    axes[0, 1].set_title(r"Predicted $\hat{u}(x, t)$")
    plt.colorbar(c1, ax=axes[0, 1])
    
    c2 = axes[0, 2].contourf(X_mesh, T_mesh, np.abs(u_final.cpu().numpy() - u_exact.cpu().numpy()), 40, cmap="inferno")
    axes[0, 2].set_title(f"Absolute Error $|u - \\hat{{u}}|$ (Max={max_err_u:.2e})")
    plt.colorbar(c2, ax=axes[0, 2])
    
    # v component
    c3 = axes[1, 0].contourf(X_mesh, T_mesh, v_exact.cpu().numpy(), 40, cmap="plasma")
    axes[1, 0].set_title(r"Exact $v(x, t)$")
    plt.colorbar(c3, ax=axes[1, 0])
    
    c4 = axes[1, 1].contourf(X_mesh, T_mesh, v_final.cpu().numpy(), 40, cmap="plasma")
    axes[1, 1].set_title(r"Predicted $\hat{v}(x, t)$")
    plt.colorbar(c4, ax=axes[1, 1])
    
    c5 = axes[1, 2].contourf(X_mesh, T_mesh, np.abs(v_final.cpu().numpy() - v_exact.cpu().numpy()), 40, cmap="inferno")
    axes[1, 2].set_title(f"Absolute Error $|v - \\hat{{v}}|$ (Max={max_err_v:.2e})")
    plt.colorbar(c5, ax=axes[1, 2])
    
    for ax in axes.flat:
        ax.set_xlabel("Space $x$")
        ax.set_ylabel("Time $t$")
        
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "spatiotemporal_contours.png"), dpi=300)
    plt.close()
    
    print(f"All figures saved to '{save_dir}/'.")
    
    # Save metrics summary to file
    with open(os.path.join(save_dir, "metrics.txt"), "w") as f:
        f.write("=== Spectral fPINN Validation Metrics ===\n")
        f.write(f"Spatial Grid N: {N} (CGL points)\n")
        f.write(f"Temporal Grid N_t: {N_t} (dt = {T/N_t:.4f})\n")
        f.write(f"Orders: alpha = {alpha}, beta = {beta}\n")
        f.write(f"Training Time: {total_time:.2f} s\n")
        f.write(f"Final Total Loss: {loss_history[-1]:.6e}\n")
        f.write(f"u-component Rel L2 Error: {final_l2_u:.6e}\n")
        f.write(f"u-component Max Abs Error: {max_err_u:.6e}\n")
        f.write(f"v-component Rel L2 Error: {final_l2_v:.6e}\n")
        f.write(f"v-component Max Abs Error: {max_err_v:.6e}\n")

    return final_l2_u, final_l2_v, total_time


if __name__ == "__main__":
    # Test run with N=20, N_t=32, 1000 Adam + 200 L-BFGS
    train_spectral_fpinn(
        N=20,
        N_t=32,
        T=1.0,
        L=1.0,
        alpha=0.4,
        beta=0.7,
        hidden_dim=64,
        num_layers=4,
        adam_epochs=1200,
        lbfgs_max_iter=250,
        lr=2e-3,
        save_dir="results"
    )
