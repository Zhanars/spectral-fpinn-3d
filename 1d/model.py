"""
Spectral Fractional Physics-Informed Neural Network (Spectral fPINN / SPINN)
For Coupled Nonlinear Klein-Gordon Equations with Caputo Fractional Memory.

Key Components:
1. Chebyshev-Gauss-Lobatto (CGL) grid and spectral D2 matrix.
2. Vectorized, fully differentiable L1 Caputo memory operator.
3. MLP with exact hard initial and boundary constraints:
   u(x,t) = x * (1 - x) * t^2 * MLP_u(x,t)
   v(x,t) = x * (1 - x) * t^2 * MLP_v(x,t)
4. Fast spatial spectral differentiation (D2 @ u) + temporal Autograd (u_tt).
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
from scipy.special import gamma as scipy_gamma


def build_chebyshev_grid_and_d2(N, L=1.0):
    """
    Constructs shifted Chebyshev-Gauss-Lobatto (CGL) points on [0, L]
    and the second-order spectral differentiation matrix D2.
    """
    j = np.arange(N + 1)
    # Standard Chebyshev nodes on [-1, 1]
    xi = np.cos(np.pi * j / N)
    x_cgl = 0.5 * L * (1.0 - xi)  # mapped to [0, L]
    
    c = np.ones(N + 1)
    c[0] = 2.0
    c[N] = 2.0
    c *= (-1.0) ** j
    
    X = np.outer(xi, np.ones(N + 1))
    dX = X - X.T + np.eye(N + 1)
    
    D = np.outer(c, 1.0 / c) / dX
    D -= np.diag(np.sum(D, axis=1))
    
    # Mapping x = 0.5 * L * (1 - xi) implies d/dx = (-2.0/L) * d/dxi
    D_x = (-2.0 / L) * D
    D2 = D_x @ D_x  # exact second derivative matrix on [0, L]
    
    return torch.tensor(x_cgl, dtype=torch.float32), torch.tensor(D2, dtype=torch.float32)


def build_l1_toeplitz_matrix(N_t, gamma, dt):
    """
    Constructs the lower-triangular Toeplitz weight matrix for the L1 scheme
    of the Caputo fractional derivative of order gamma in (0, 1).
    
    (D_t^gamma U)_n = prefactor * sum_{k=0}^{n-1} w_k * (U_{n-k} - U_{n-k-1})
    where w_k = (k+1)^(1-gamma) - k^(1-gamma).
    """
    k = np.arange(N_t)
    w = (k + 1.0) ** (1.0 - gamma) - (k) ** (1.0 - gamma)
    
    # Lower triangular matrix W: row n has weights w[n-1], w[n-2], ..., w[0]
    W = np.zeros((N_t, N_t), dtype=np.float32)
    for n in range(N_t):
        W[n, :n+1] = w[:n+1][::-1]
        
    prefactor = (dt ** (-gamma)) / scipy_gamma(2.0 - gamma)
    W = prefactor * W
    return torch.tensor(W, dtype=torch.float32)


class SineActivation(nn.Module):
    """Sine activation with omega factor (ideal for wave/periodic phenomena)."""
    def __init__(self, omega=1.0):
        super().__init__()
        self.omega = omega

    def forward(self, x):
        return torch.sin(self.omega * x)


class SpectralFPINN(nn.Module):
    """
    Spectral Fractional PINN model for coupled nonlinear Klein-Gordon system.
    """
    def __init__(self, N=24, N_t=64, T=1.0, L=1.0, hidden_dim=64, num_layers=4,
                 alpha=0.4, beta=0.7, c1=1.0, c2=1.0, m1=1.0, m2=1.0,
                 k1=1.0, k2=1.0, sigma1=1.0, sigma2=1.0, eta=1.0):
        super().__init__()
        self.N = N
        self.N_t = N_t
        self.T = T
        self.L = L
        self.dt = T / N_t
        
        # Physical parameters
        self.alpha = alpha
        self.beta = beta
        self.c1 = c1
        self.c2 = c2
        self.m1 = m1
        self.m2 = m2
        self.k1 = k1
        self.k2 = k2
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.eta = eta
        
        # Grid and spectral matrix
        x_cgl, D2 = build_chebyshev_grid_and_d2(N, L)
        self.register_buffer('x_cgl', x_cgl)  # shape (N+1,)
        self.register_buffer('D2', D2)        # shape (N+1, N+1)
        
        # Time steps t_0=0, t_1=dt, ..., t_{N_t}=T
        t_grid = torch.linspace(0.0, T, N_t + 1, dtype=torch.float32)
        self.register_buffer('t_grid', t_grid)  # shape (N_t+1,)
        
        # Precomputed L1 matrices for memory acceleration
        W_alpha = build_l1_toeplitz_matrix(N_t, alpha, self.dt)
        W_beta = build_l1_toeplitz_matrix(N_t, beta, self.dt)
        self.register_buffer('W_alpha', W_alpha)  # shape (N_t, N_t)
        self.register_buffer('W_beta', W_beta)    # shape (N_t, N_t)
        
        # Fourier feature embedding dimension (x, t, sin(pi*x), cos(pi*x), sin(2*pi*x), cos(2*pi*x))
        in_dim = 6
        layers = [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dim, 2))
        self.mlp = nn.Sequential(*layers)
        
        # Xavier initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward_raw(self, x, t):
        """Raw MLP forward pass with spatial Fourier feature embedding."""
        sin_pix = torch.sin(np.pi * x)
        cos_pix = torch.cos(np.pi * x)
        sin_2pix = torch.sin(2.0 * np.pi * x)
        cos_2pix = torch.cos(2.0 * np.pi * x)
        xt = torch.stack([x, t, sin_pix, cos_pix, sin_2pix, cos_2pix], dim=-1)
        return self.mlp(xt)

    def forward(self, x, t):
        """
        Forward pass with Hard Constraints:
        u(x, t) = x * (L - x) * t^2 * raw_u(x, t)
        v(x, t) = x * (L - x) * t^2 * raw_v(x, t)
        
        Guarantees:
        u(0, t) = u(L, t) = 0
        u(x, 0) = 0
        u_t(x, 0) = 0
        """
        raw = self.forward_raw(x, t)
        factor = x * (self.L - x) * (t ** 2)
        u = factor * raw[..., 0]
        v = factor * raw[..., 1]
        return u, v

    def evaluate_grid(self):
        """
        Evaluates u, v over the full tensor-product grid (N_t+1, N+1).
        Returns:
            u: tensor of shape (N_t+1, N+1)
            v: tensor of shape (N_t+1, N+1)
            t_mesh: tensor of shape (N_t+1, N+1) requiring gradients for Autograd
            x_mesh: tensor of shape (N_t+1, N+1)
        """
        # Create 2D meshgrid
        # t_mesh shape: (N_t+1, N+1), x_mesh shape: (N_t+1, N+1)
        t_mesh = self.t_grid.unsqueeze(1).expand(-1, self.N + 1).clone()
        x_mesh = self.x_cgl.unsqueeze(0).expand(self.N_t + 1, -1).clone()
        t_mesh.requires_grad_(True)
        
        u, v = self.forward(x_mesh, t_mesh)
        return u, v, t_mesh, x_mesh

    def compute_fractional_derivatives(self, u, v):
        """
        Computes Caputo fractional derivatives ^C D_t^alpha u and ^C D_t^beta v
        at time steps t_1, ..., t_{N_t} using the precomputed L1 Toeplitz matrices.
        
        u: shape (N_t+1, N+1)
        v: shape (N_t+1, N+1)
        Returns:
            Du: shape (N_t, N+1)
            Dv: shape (N_t, N+1)
        """
        # Increments: delta_u shape (N_t, N+1)
        delta_u = u[1:] - u[:-1]
        delta_v = v[1:] - v[:-1]
        
        # Matrix multiplication over time dimension: (N_t, N_t) @ (N_t, N+1) -> (N_t, N+1)
        Du = torch.matmul(self.W_alpha, delta_u)
        Dv = torch.matmul(self.W_beta, delta_v)
        return Du, Dv

    def compute_spatial_derivatives(self, u, v):
        """
        Computes second spatial derivatives using Chebyshev spectral matrix D2:
        u_xx = u @ D2^T
        
        u: shape (N_t+1, N+1)
        v: shape (N_t+1, N+1)
        Returns:
            u_xx: shape (N_t+1, N+1)
            v_xx: shape (N_t+1, N+1)
        """
        # Matrix multiplication across spatial dimension
        u_xx = torch.matmul(u, self.D2.T)
        v_xx = torch.matmul(v, self.D2.T)
        return u_xx, v_xx

    def compute_pde_residuals(self, f1_func, f2_func):
        """
        Computes the physics residuals Res_u and Res_v across the internal grid.
        f1_func, f2_func: callables (x, t) returning analytical source terms.
        """
        u, v, t_mesh, x_mesh = self.evaluate_grid()
        
        # 1. Temporal derivatives via Autograd
        # First time derivative: u_t, v_t
        u_t = torch.autograd.grad(
            u, t_mesh,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]
        v_t = torch.autograd.grad(
            v, t_mesh,
            grad_outputs=torch.ones_like(v),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Second time derivative: u_tt, v_tt
        u_tt = torch.autograd.grad(
            u_t, t_mesh,
            grad_outputs=torch.ones_like(u_t),
            create_graph=True,
            retain_graph=True
        )[0]
        v_tt = torch.autograd.grad(
            v_t, t_mesh,
            grad_outputs=torch.ones_like(v_t),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # 2. Spatial derivatives via Chebyshev matrix D2
        u_xx, v_xx = self.compute_spatial_derivatives(u, v)
        
        # 3. Fractional Caputo derivatives via L1 operator (for t_1 ... t_Nt)
        caputo_u, caputo_v = self.compute_fractional_derivatives(u, v)
        
        # We evaluate the residual on steps n = 1 ... N_t (interior of time domain)
        u_sub = u[1:]
        v_sub = v[1:]
        u_tt_sub = u_tt[1:]
        v_tt_sub = v_tt[1:]
        u_xx_sub = u_xx[1:]
        v_xx_sub = v_xx[1:]
        x_sub = x_mesh[1:]
        t_sub = t_mesh[1:]
        
        # Exact source functions
        f1_vals = f1_func(x_sub, t_sub)
        f2_vals = f2_func(x_sub, t_sub)
        
        # PDE residual equations:
        # res_u = u_tt - c1^2 u_xx + sigma1 * ^C D_t^alpha u + m1 u + k1 u^3 + eta u v^2 - f1
        res_u = (u_tt_sub 
                 - (self.c1 ** 2) * u_xx_sub 
                 + self.sigma1 * caputo_u 
                 + self.m1 * u_sub 
                 + self.k1 * (u_sub ** 3) 
                 + self.eta * u_sub * (v_sub ** 2) 
                 - f1_vals)
        
        # res_v = v_tt - c2^2 v_xx + sigma2 * ^C D_t^beta v + m2 v + k2 v^3 + eta v u^2 - f2
        res_v = (v_tt_sub 
                 - (self.c2 ** 2) * v_xx_sub 
                 + self.sigma2 * caputo_v 
                 + self.m2 * v_sub 
                 + self.k2 * (v_sub ** 3) 
                 + self.eta * v_sub * (u_sub ** 2) 
                 - f2_vals)
        
        loss_u = torch.mean(res_u ** 2)
        loss_v = torch.mean(res_v ** 2)
        total_loss = loss_u + loss_v
        
        return total_loss, loss_u, loss_v, (u, v)
