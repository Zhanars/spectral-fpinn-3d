"""
Dimension-agnostic spectral building blocks shared by the 3D Spectral fPINN:
Chebyshev-Gauss-Lobatto (CGL) grid + D2 differentiation matrix, and the
L1 Toeplitz quadrature matrix for the Caputo fractional derivative.

Direct Julia port of spectral_utils.py. Uses Float64 throughout (the Python
version used Float32); this is a deliberate, noted difference for the
cross-validation exercise, not an attempt at bit-exact reproduction.
"""
module SpectralUtils

using LinearAlgebra
using SpecialFunctions: gamma, loggamma

export build_chebyshev_grid_and_d2, build_l1_toeplitz_matrix, build_differentiable_l1_matrix

"""
Constructs shifted Chebyshev-Gauss-Lobatto (CGL) points on [0, L]
and the second-order spectral differentiation matrix D2.
Direct port of spectral_utils.build_chebyshev_grid_and_d2.
"""
function build_chebyshev_grid_and_d2(N::Int, L::Float64=1.0)
    j = collect(0:N)
    xi = cos.(π .* j ./ N)
    x_cgl = 0.5 * L .* (1.0 .- xi)

    c = ones(Float64, N + 1)
    c[1] = 2.0
    c[end] = 2.0
    c .*= (-1.0) .^ j

    X = xi * ones(1, N + 1)
    dX = X - X' + Matrix{Float64}(I, N + 1, N + 1)

    D = (c * (1.0 ./ c)') ./ dX
    D = D - Diagonal(vec(sum(D, dims=2)))

    Dx = (-2.0 / L) .* D
    D2 = Dx * Dx

    return x_cgl, D2
end

"""
Lower-triangular Toeplitz L1-scheme weight matrix for the Caputo derivative
of order gamma in (0,1), with a FIXED (non-differentiable) gamma. Used for
the forward problem. Direct port of spectral_utils.build_l1_toeplitz_matrix.
"""
function build_l1_toeplitz_matrix(Nt::Int, gamma_val::Float64, dt::Float64)
    w = [(Float64(i))^(1.0 - gamma_val) - (Float64(i - 1))^(1.0 - gamma_val) for i in 1:Nt]
    W = zeros(Float64, Nt, Nt)
    for n in 1:Nt
        for m in 1:n
            W[n, m] = w[n - m + 1]
        end
    end
    prefactor = dt^(-gamma_val) / gamma(2.0 - gamma_val)
    return prefactor .* W
end

"""
Same L1 Toeplitz weight matrix, but built via a non-mutating comprehension
from a scalar `gamma_val` that may be a Dual (ForwardDiff) or tracked
(Zygote) number, so gradients flow through the fractional order itself.
Direct port of spectral_utils.build_differentiable_l1_matrix.
"""
function build_differentiable_l1_matrix(Nt::Int, gamma_val, dt::Float64)
    w = [(Float64(i))^(1.0 - gamma_val) - (Float64(i - 1))^(1.0 - gamma_val) for i in 1:Nt]
    W = [m <= n ? w[n - m + 1] : zero(gamma_val) for n in 1:Nt, m in 1:Nt]
    prefactor = dt^(-gamma_val) / exp(loggamma(2.0 - gamma_val))
    return prefactor .* W
end

end # module
