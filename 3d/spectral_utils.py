"""
Dimension-agnostic spectral building blocks shared by the 3D Spectral fPINN:
Chebyshev-Gauss-Lobatto (CGL) grid + D2 differentiation matrix, and the
L1 Toeplitz quadrature matrix for the Caputo fractional derivative.
"""

import numpy as np
import torch
from scipy.special import gamma as scipy_gamma


def build_chebyshev_grid_and_d2(N, L=1.0):
    """
    Constructs shifted Chebyshev-Gauss-Lobatto (CGL) points on [0, L]
    and the second-order spectral differentiation matrix D2.
    """
    j = np.arange(N + 1)
    xi = np.cos(np.pi * j / N)
    x_cgl = 0.5 * L * (1.0 - xi)

    c = np.ones(N + 1)
    c[0] = 2.0
    c[N] = 2.0
    c *= (-1.0) ** j

    X = np.outer(xi, np.ones(N + 1))
    dX = X - X.T + np.eye(N + 1)

    D = np.outer(c, 1.0 / c) / dX
    D -= np.diag(np.sum(D, axis=1))

    D_x = (-2.0 / L) * D
    D2 = D_x @ D_x

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

    W = np.zeros((N_t, N_t), dtype=np.float32)
    for n in range(N_t):
        W[n, :n + 1] = w[:n + 1][::-1]

    prefactor = (dt ** (-gamma)) / scipy_gamma(2.0 - gamma)
    W = prefactor * W
    return torch.tensor(W, dtype=torch.float32)


def build_differentiable_l1_matrix(N_t, gamma_tensor, dt):
    """
    Same L1 Toeplitz weight matrix as build_l1_toeplitz_matrix, but built from
    native PyTorch ops on a trainable gamma_tensor so Autograd can flow through
    the fractional order itself (used for inverse parameter discovery).
    """
    k = torch.arange(N_t, dtype=torch.float32, device=gamma_tensor.device)
    w = (k + 1.0) ** (1.0 - gamma_tensor) - (k) ** (1.0 - gamma_tensor)

    rows = []
    for n in range(N_t):
        row = torch.zeros(N_t, device=gamma_tensor.device)
        row[:n + 1] = torch.flip(w[:n + 1], dims=[0])
        rows.append(row)
    W = torch.stack(rows)

    prefactor = (dt ** (-gamma_tensor)) / torch.exp(torch.special.gammaln(2.0 - gamma_tensor))
    return prefactor * W
