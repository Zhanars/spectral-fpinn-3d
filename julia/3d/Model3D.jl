"""
3D Spectral Fractional PINN, direct Julia port of model_3d.py.

AD strategy: same as Model1D.jl (see its docstring for the full story of why
u_tt uses a centered finite-difference stencil in t rather than nested
autograd). Here everything is vectorized over the flattened 4D grid via
matrix operations (permutedims + reshape + matmul for the Kronecker-sum
Laplacian and Caputo Toeplitz contraction, mirroring PyTorch's einsum calls),
rather than per-point loops, for performance on the larger 3D grids.
"""
module Model3D

include(joinpath(@__DIR__, "SpectralUtils.jl"))
using .SpectralUtils: build_chebyshev_grid_and_d2, build_l1_toeplitz_matrix

using Random
using LinearAlgebra

export SpectralFPINN3D, init_params, forward_batch, build_grid4d,
       contract_dim, compute_loss, flatten_params, unflatten_params

const TIME_FD_STEP = 1e-4

struct SpectralFPINN3D
    Nx::Int; Ny::Int; Nz::Int; Nt::Int
    Lx::Float64; Ly::Float64; Lz::Float64; T::Float64
    dt::Float64
    alpha::Float64; beta::Float64
    c1::Float64; c2::Float64; m1::Float64; m2::Float64
    k1::Float64; k2::Float64; sigma1::Float64; sigma2::Float64; eta::Float64
    x_cgl::Vector{Float64}; y_cgl::Vector{Float64}; z_cgl::Vector{Float64}
    D2_x::Matrix{Float64}; D2_y::Matrix{Float64}; D2_z::Matrix{Float64}
    t_grid::Vector{Float64}
    W_alpha::Matrix{Float64}; W_beta::Matrix{Float64}
    hidden_dim::Int; num_layers::Int
end

function SpectralFPINN3D(; Nx=8, Ny=8, Nz=8, Nt=16, Lx=1.0, Ly=1.0, Lz=1.0, T=1.0,
                          alpha=0.4, beta=0.7, c1=1.0, c2=1.0, m1=1.0, m2=1.0,
                          k1=1.0, k2=1.0, sigma1=1.0, sigma2=1.0, eta=1.0,
                          hidden_dim=96, num_layers=4)
    x_cgl, D2_x = build_chebyshev_grid_and_d2(Nx, Lx)
    y_cgl, D2_y = build_chebyshev_grid_and_d2(Ny, Ly)
    z_cgl, D2_z = build_chebyshev_grid_and_d2(Nz, Lz)
    dt = T / Nt
    t_grid = collect(range(0.0, T, length=Nt + 1))
    W_alpha = build_l1_toeplitz_matrix(Nt, alpha, dt)
    W_beta = build_l1_toeplitz_matrix(Nt, beta, dt)
    return SpectralFPINN3D(Nx, Ny, Nz, Nt, Lx, Ly, Lz, T, dt, alpha, beta,
                            c1, c2, m1, m2, k1, k2, sigma1, sigma2, eta,
                            x_cgl, y_cgl, z_cgl, D2_x, D2_y, D2_z, t_grid,
                            W_alpha, W_beta, hidden_dim, num_layers)
end

function init_params(model::SpectralFPINN3D; rng=Random.default_rng())
    in_dim = 10  # x,y,z,t, sin/cos(pi x), sin/cos(pi y), sin/cos(pi z)
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

# Same mlp_apply generalizes to batched (features x Npoints) input via Julia's
# generic matrix multiplication + column-broadcast bias addition.
function mlp_apply(params, xt)
    h = xt
    for l in 1:(length(params) - 1)
        W, b = params[l]
        h = tanh.(W * h .+ b)
    end
    W, b = params[end]
    return W * h .+ b
end

"""Batched forward pass. X,Y,Z,T are equal-length Vectors (flattened grid
points). Returns (U, V) as Vectors of the same length."""
function forward_batch(params, X, Y, Z, T, Lx, Ly, Lz)
    feats = vcat(X', Y', Z', T',
                 sin.(π .* X)', cos.(π .* X)',
                 sin.(π .* Y)', cos.(π .* Y)',
                 sin.(π .* Z)', cos.(π .* Z)')  # (10, Npoints)
    raw = mlp_apply(params, feats)  # (2, Npoints)
    envelope = X .* (Lx .- X) .* Y .* (Ly .- Y) .* Z .* (Lz .- Z) .* (T .^ 2)
    U = envelope .* raw[1, :]
    V = envelope .* raw[2, :]
    return U, V
end

"""4D coordinate grids (Nt+1, Nx+1, Ny+1, Nz+1), matching
np.meshgrid(t,x,y,z,indexing='ij') ordering used throughout the Python code."""
function build_grid4d(model::SpectralFPINN3D)
    Nt1, Nx1, Ny1, Nz1 = model.Nt + 1, model.Nx + 1, model.Ny + 1, model.Nz + 1
    T4 = [model.t_grid[i] for i in 1:Nt1, j in 1:Nx1, k in 1:Ny1, l in 1:Nz1]
    X4 = [model.x_cgl[j] for i in 1:Nt1, j in 1:Nx1, k in 1:Ny1, l in 1:Nz1]
    Y4 = [model.y_cgl[k] for i in 1:Nt1, j in 1:Nx1, k in 1:Ny1, l in 1:Nz1]
    Z4 = [model.z_cgl[l] for i in 1:Nt1, j in 1:Nx1, k in 1:Ny1, l in 1:Nz1]
    return T4, X4, Y4, Z4
end

"""Contracts matrix D (n x n) against array U's `dim`-th axis: mirrors
torch.einsum('ia,...a...->...i...', D, U) for whichever axis `a` sits at.
Implemented via permutedims+reshape+matmul so Zygote differentiates it with
its standard (well-supported) matrix-multiplication adjoint."""
function contract_dim(D::AbstractMatrix, U::AbstractArray{T,4}, dim::Int) where T
    perm = (dim, filter(!=(dim), 1:4)...)
    Up = permutedims(U, perm)
    n = size(Up, 1)
    rest = size(Up)[2:end]
    R2 = D * reshape(Up, n, :)
    R = reshape(R2, n, rest...)
    invperm = invperm_of(perm)
    return permutedims(R, invperm)
end

invperm_of(perm) = ntuple(i -> findfirst(==(i), perm), length(perm))

"""
Full 3D PDE residual loss, mirroring SpectralFPINN3D.compute_loss.
F1, F2 are precomputed analytical source arrays of shape (Nt+1,Nx+1,Ny+1,Nz+1)
(only indices 2:end are used, matching Python's f1_exact[1:]).
"""
function compute_loss(params, model::SpectralFPINN3D, F1, F2, h=TIME_FD_STEP)
    T4, X4, Y4, Z4 = build_grid4d(model)
    sz = size(T4)
    Xf, Yf, Zf, Tf = vec(X4), vec(Y4), vec(Z4), vec(T4)

    U0f, V0f = forward_batch(params, Xf, Yf, Zf, Tf, model.Lx, model.Ly, model.Lz)
    Upf, Vpf = forward_batch(params, Xf, Yf, Zf, Tf .+ h, model.Lx, model.Ly, model.Lz)
    Umf, Vmf = forward_batch(params, Xf, Yf, Zf, Tf .- h, model.Lx, model.Ly, model.Lz)

    U0 = reshape(U0f, sz); V0 = reshape(V0f, sz)
    Utt = reshape((Upf .- 2 .* U0f .+ Umf) ./ h^2, sz)
    Vtt = reshape((Vpf .- 2 .* V0f .+ Vmf) ./ h^2, sz)

    lap_u = contract_dim(model.D2_x, U0, 2) .+ contract_dim(model.D2_y, U0, 3) .+ contract_dim(model.D2_z, U0, 4)
    lap_v = contract_dim(model.D2_x, V0, 2) .+ contract_dim(model.D2_y, V0, 3) .+ contract_dim(model.D2_z, V0, 4)

    dU = U0[2:end, :, :, :] .- U0[1:end-1, :, :, :]
    dV = V0[2:end, :, :, :] .- V0[1:end-1, :, :, :]
    cap_u = contract_dim(model.W_alpha, dU, 1)
    cap_v = contract_dim(model.W_beta, dV, 1)

    U_int = U0[2:end, :, :, :]; V_int = V0[2:end, :, :, :]
    Utt_int = Utt[2:end, :, :, :]; Vtt_int = Vtt[2:end, :, :, :]
    lap_u_int = lap_u[2:end, :, :, :]; lap_v_int = lap_v[2:end, :, :, :]
    F1_int = F1[2:end, :, :, :]; F2_int = F2[2:end, :, :, :]

    res_u = Utt_int .- (model.c1^2) .* lap_u_int .+ model.sigma1 .* cap_u .+
            model.m1 .* U_int .+ model.k1 .* (U_int .^ 3) .+ model.eta .* U_int .* (V_int .^ 2) .- F1_int
    res_v = Vtt_int .- (model.c2^2) .* lap_v_int .+ model.sigma2 .* cap_v .+
            model.m2 .* V_int .+ model.k2 .* (V_int .^ 3) .+ model.eta .* V_int .* (U_int .^ 2) .- F2_int

    loss = sum(res_u .^ 2) / length(res_u) + sum(res_v .^ 2) / length(res_v)
    return loss, U0, V0
end

function flatten_params(params)
    v = Float64[]
    shapes = Tuple{Tuple{Int,Int},Int}[]
    for (W, b) in params
        append!(v, vec(W)); append!(v, b)
        push!(shapes, (size(W), length(b)))
    end
    return v, shapes
end

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
