"""
1D Spectral Fractional PINN, direct Julia port of _archive_1d_2d/code/model.py.

AD strategy note (deliberate, disclosed difference from the PyTorch original,
which uses exact `torch.autograd.grad(..., create_graph=True)` for u_tt):
composing Julia's forward-mode (ForwardDiff, ideal for the single scalar time
coordinate) with its reverse-mode (Zygote, ideal for the many-parameter MLP)
for this specific "second time-derivative inside a trained network" pattern
hit two separate, confirmed-bad interactions during development: (1) Zygote's
built-in adjoint rule for `ForwardDiff.derivative` silently ignores gradients
through the closure's captured parameters (produced a wrong gradient, ~1.4%
off from a finite-difference check, with no error raised); (2) manually
nesting `ForwardDiff.Dual` numbers for a second-order derivative, combined
with an outer `ForwardDiff.gradient` or `Zygote.gradient` pass over the
parameters, hits Dual-tag-ordering errors Julia cannot resolve automatically.
Fully-nested Zygote (reverse-over-reverse) is correct but prohibitively slow
to compile even for a tiny network.

Resolution used here: u_tt is computed via a plain centered finite-difference
stencil in t (step h=1e-4, i.e. O(h^2) ~ 1e-8 truncation error, negligible
next to the ~1e-3-1e-2 relative L2 errors this solver targets), which is just
three ordinary forward evaluations -- no nested Dual types, no tag-ordering
ambiguity. The outer training gradient then uses plain reverse-mode Zygote.
This combination was verified against a finite-difference check of the FULL
loss gradient (not just u_tt) to 1.7e-5 relative error before use.
"""
module Model1D

include(joinpath(@__DIR__, "..", "3d", "SpectralUtils.jl"))
using .SpectralUtils: build_chebyshev_grid_and_d2, build_l1_toeplitz_matrix

using Random
using LinearAlgebra

const TIME_FD_STEP = 1e-4

export SpectralFPINN1D, init_params, forward, forward_raw, evaluate_grid,
       compute_fractional_derivatives, compute_spatial_derivatives, compute_pde_loss,
       flatten_params, unflatten_params

struct SpectralFPINN1D
    N::Int
    Nt::Int
    T::Float64
    L::Float64
    dt::Float64
    alpha::Float64
    beta::Float64
    c1::Float64
    c2::Float64
    m1::Float64
    m2::Float64
    k1::Float64
    k2::Float64
    sigma1::Float64
    sigma2::Float64
    eta::Float64
    x_cgl::Vector{Float64}
    D2::Matrix{Float64}
    t_grid::Vector{Float64}
    W_alpha::Matrix{Float64}
    W_beta::Matrix{Float64}
    hidden_dim::Int
    num_layers::Int
end

function SpectralFPINN1D(; N=24, Nt=64, T=1.0, L=1.0, hidden_dim=64, num_layers=4,
                          alpha=0.4, beta=0.7, c1=1.0, c2=1.0, m1=1.0, m2=1.0,
                          k1=1.0, k2=1.0, sigma1=1.0, sigma2=1.0, eta=1.0)
    x_cgl, D2 = build_chebyshev_grid_and_d2(N, L)
    dt = T / Nt
    t_grid = collect(range(0.0, T, length=Nt + 1))
    W_alpha = build_l1_toeplitz_matrix(Nt, alpha, dt)
    W_beta = build_l1_toeplitz_matrix(Nt, beta, dt)
    return SpectralFPINN1D(N, Nt, T, L, dt, alpha, beta, c1, c2, m1, m2, k1, k2,
                            sigma1, sigma2, eta, x_cgl, D2, t_grid, W_alpha, W_beta,
                            hidden_dim, num_layers)
end

# ---- Parameters: a Vector of (W, b) tuples, one per Linear layer ----
# Xavier-normal weight init (PyTorch nn.init.xavier_normal_ default: gain=1,
# std = sqrt(2/(fan_in+fan_out))), zero bias, matching model._init_weights().
function init_params(model::SpectralFPINN1D; rng=Random.default_rng())
    in_dim = 6  # x, t, sin(pi x), cos(pi x), sin(2 pi x), cos(2 pi x)
    h = model.hidden_dim
    dims = vcat([in_dim], fill(h, model.num_layers - 1), [2])
    params = Vector{Tuple{Matrix{Float64},Vector{Float64}}}()
    for l in 1:(length(dims) - 1)
        fan_in, fan_out = dims[l], dims[l + 1]
        std = sqrt(2.0 / (fan_in + fan_out))
        W = randn(rng, fan_out, fan_in) .* std
        b = zeros(fan_out)
        push!(params, (W, b))
    end
    return params
end

function mlp_apply(params, xt)
    h = xt
    for l in 1:(length(params) - 1)
        W, b = params[l]
        h = tanh.(W * h .+ b)
    end
    W, b = params[end]
    return W * h .+ b
end

function forward_raw(params, x, t)
    xt = [x, t, sin(π * x), cos(π * x), sin(2π * x), cos(2π * x)]
    return mlp_apply(params, xt)  # length-2 vector [raw_u, raw_v]
end

function forward(params, x, t, L)
    raw = forward_raw(params, x, t)
    factor = x * (L - x) * t^2
    return factor * raw[1], factor * raw[2]
end

"""u, v, u_tt, v_tt at a single (x,t) point via a centered finite-difference
stencil over t (see module docstring for why, in place of nested autograd)."""
function point_with_time_derivs(params, x, t, L, h=TIME_FD_STEP)
    u0, v0 = forward(params, x, t, L)
    up, vp = forward(params, x, t + h, L)
    um, vm = forward(params, x, t - h, L)
    u_tt = (up - 2u0 + um) / h^2
    v_tt = (vp - 2v0 + vm) / h^2
    return u0, v0, u_tt, v_tt
end

"""Evaluates u, v over the full (Nt+1) x (N+1) grid (no time derivatives)."""
function evaluate_grid(params, model::SpectralFPINN1D)
    Nt1, N1 = model.Nt + 1, model.N + 1
    U = [forward(params, model.x_cgl[j], model.t_grid[i], model.L)[1] for i in 1:Nt1, j in 1:N1]
    V = [forward(params, model.x_cgl[j], model.t_grid[i], model.L)[2] for i in 1:Nt1, j in 1:N1]
    return U, V
end

function compute_fractional_derivatives(model::SpectralFPINN1D, U, V)
    dU = U[2:end, :] - U[1:end-1, :]
    dV = V[2:end, :] - V[1:end-1, :]
    Du = model.W_alpha * dU
    Dv = model.W_beta * dV
    return Du, Dv
end

function compute_spatial_derivatives(model::SpectralFPINN1D, U, V)
    Uxx = U * model.D2'
    Vxx = V * model.D2'
    return Uxx, Vxx
end

"""
Full PDE residual loss, mirroring model.compute_pde_residuals. f1_func, f2_func
take (x, t) and return the scalar analytical source value.
"""
function compute_pde_loss(params, model::SpectralFPINN1D, f1_func, f2_func)
    Nt1, N1 = model.Nt + 1, model.N + 1
    pts = [point_with_time_derivs(params, model.x_cgl[j], model.t_grid[i], model.L) for i in 1:Nt1, j in 1:N1]
    U = getindex.(pts, 1)
    V = getindex.(pts, 2)
    Utt = getindex.(pts, 3)
    Vtt = getindex.(pts, 4)

    Uxx, Vxx = compute_spatial_derivatives(model, U, V)
    CapU, CapV = compute_fractional_derivatives(model, U, V)

    U_sub = U[2:end, :]
    V_sub = V[2:end, :]
    Utt_sub = Utt[2:end, :]
    Vtt_sub = Vtt[2:end, :]
    Uxx_sub = Uxx[2:end, :]
    Vxx_sub = Vxx[2:end, :]

    F1 = [f1_func(model.x_cgl[j], model.t_grid[i]) for i in 2:Nt1, j in 1:N1]
    F2 = [f2_func(model.x_cgl[j], model.t_grid[i]) for i in 2:Nt1, j in 1:N1]

    res_u = Utt_sub .- (model.c1^2) .* Uxx_sub .+ model.sigma1 .* CapU .+
            model.m1 .* U_sub .+ model.k1 .* (U_sub .^ 3) .+
            model.eta .* U_sub .* (V_sub .^ 2) .- F1
    res_v = Vtt_sub .- (model.c2^2) .* Vxx_sub .+ model.sigma2 .* CapV .+
            model.m2 .* V_sub .+ model.k2 .* (V_sub .^ 3) .+
            model.eta .* V_sub .* (U_sub .^ 2) .- F2

    loss_u = sum(res_u .^ 2) / length(res_u)
    loss_v = sum(res_v .^ 2) / length(res_v)
    return loss_u + loss_v, loss_u, loss_v, U, V
end

"""Flattens the Vector{(W,b)} parameter structure into one Float64 vector
plus the shape metadata needed to reconstruct it (for Optim.jl's L-BFGS,
which wants a flat vector interface)."""
function flatten_params(params)
    v = Float64[]
    shapes = Tuple{Tuple{Int,Int},Int}[]
    for (W, b) in params
        append!(v, vec(W))
        append!(v, b)
        push!(shapes, (size(W), length(b)))
    end
    return v, shapes
end

"""Non-mutating (Zygote-differentiable) inverse of flatten_params: builds the
result via a comprehension rather than push!, since Zygote cannot
differentiate through in-place array mutation."""
function unflatten_params(v, shapes)
    sizes = [wshape[1] * wshape[2] + blen for (wshape, blen) in shapes]
    offsets = cumsum(vcat([0], sizes))
    return [
        let wshape = shapes[i][1], blen = shapes[i][2], n = wshape[1] * wshape[2], off = offsets[i]
            (reshape(v[off+1:off+n], wshape), v[off+n+1:off+n+blen])
        end
        for i in 1:length(shapes)
    ]
end

end # module
