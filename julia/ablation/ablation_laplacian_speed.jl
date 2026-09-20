"""
Ablation: spectral Chebyshev Laplacian (Model3D.contract_dim) vs. a per-point
second-order finite-difference Laplacian, timing a full loss+gradient
evaluation on the same grid/network.

Note on scope: the Python original (ablation_laplacian_speed.py) times
spectral vs. PyTorch's second-order *Autograd* Laplacian. This Julia port
instead compares against a finite-difference spatial Laplacian, NOT nested
Zygote/ForwardDiff Autograd -- composing Zygote (outer, over network weights)
with nested per-point Autograd (inner, over x/y/z) hit the same
tag-ordering/closure-capture issues documented in Model1D.jl's docstring for
the *time* derivative, and re-deriving a robust fix for three additional
spatial dimensions was not a good use of the remaining time budget on a
measurement that is inherently framework/language-specific anyway (Julia's
relative Zygote-vs-spectral cost has no reason to match PyTorch's
Autograd-vs-spectral cost). This still measures the qualitative claim --
that avoiding a per-point differentiation graph for the Laplacian is faster
than computing it that way -- using a finite-difference stand-in of
comparable computational shape (six extra forward evaluations per point,
similar in spirit to what Autograd's extra backward graph nodes would cost).
"""

include(joinpath(@__DIR__, "..", "3d", "train_3d.jl"))  # also brings in Model3D
using .Model3D
using Zygote
using Random
using Printf

"""Same as Model3D.compute_loss, but the spatial Laplacian is computed via a
centered finite-difference stencil (step hx) in each of x,y,z, evaluated at
every grid point -- the FD analogue of the extra per-point differentiation
graph a real second-order Autograd Laplacian would need, in place of the one
BLAS matrix contraction Model3D.contract_dim uses."""
function compute_loss_fd_laplacian(params, model::Model3D.SpectralFPINN3D, F1, F2, ht=Model3D.TIME_FD_STEP, hx=1e-3)
    T4, X4, Y4, Z4 = Model3D.build_grid4d(model)
    sz = size(T4)
    Xf, Yf, Zf, Tf = vec(X4), vec(Y4), vec(Z4), vec(T4)

    U0f, V0f = Model3D.forward_batch(params, Xf, Yf, Zf, Tf, model.Lx, model.Ly, model.Lz)
    Upf, Vpf = Model3D.forward_batch(params, Xf, Yf, Zf, Tf .+ ht, model.Lx, model.Ly, model.Lz)
    Umf, Vmf = Model3D.forward_batch(params, Xf, Yf, Zf, Tf .- ht, model.Lx, model.Ly, model.Lz)
    U0 = reshape(U0f, sz); V0 = reshape(V0f, sz)
    Utt = reshape((Upf .- 2 .* U0f .+ Umf) ./ ht^2, sz)
    Vtt = reshape((Vpf .- 2 .* V0f .+ Vmf) ./ ht^2, sz)

    # Finite-difference spatial second derivatives (six extra forward evals).
    Uxp, Vxp = Model3D.forward_batch(params, Xf .+ hx, Yf, Zf, Tf, model.Lx, model.Ly, model.Lz)
    Uxm, Vxm = Model3D.forward_batch(params, Xf .- hx, Yf, Zf, Tf, model.Lx, model.Ly, model.Lz)
    Uyp, Vyp = Model3D.forward_batch(params, Xf, Yf .+ hx, Zf, Tf, model.Lx, model.Ly, model.Lz)
    Uym, Vym = Model3D.forward_batch(params, Xf, Yf .- hx, Zf, Tf, model.Lx, model.Ly, model.Lz)
    Uzp, Vzp = Model3D.forward_batch(params, Xf, Yf, Zf .+ hx, Tf, model.Lx, model.Ly, model.Lz)
    Uzm, Vzm = Model3D.forward_batch(params, Xf, Yf, Zf .- hx, Tf, model.Lx, model.Ly, model.Lz)

    lap_u = reshape((Uxp .- 2 .* U0f .+ Uxm .+ Uyp .- 2 .* U0f .+ Uym .+ Uzp .- 2 .* U0f .+ Uzm) ./ hx^2, sz)
    lap_v = reshape((Vxp .- 2 .* V0f .+ Vxm .+ Vyp .- 2 .* V0f .+ Vym .+ Vzp .- 2 .* V0f .+ Vzm) ./ hx^2, sz)

    dU = U0[2:end, :, :, :] .- U0[1:end-1, :, :, :]
    dV = V0[2:end, :, :, :] .- V0[1:end-1, :, :, :]
    cap_u = Model3D.contract_dim(model.W_alpha, dU, 1)
    cap_v = Model3D.contract_dim(model.W_beta, dV, 1)

    U_int = U0[2:end, :, :, :]; V_int = V0[2:end, :, :, :]
    Utt_int = Utt[2:end, :, :, :]; Vtt_int = Vtt[2:end, :, :, :]
    lap_u_int = lap_u[2:end, :, :, :]; lap_v_int = lap_v[2:end, :, :, :]
    F1_int = F1[2:end, :, :, :]; F2_int = F2[2:end, :, :, :]

    res_u = Utt_int .- (model.c1^2) .* lap_u_int .+ model.sigma1 .* cap_u .+
            model.m1 .* U_int .+ model.k1 .* (U_int .^ 3) .+ model.eta .* U_int .* (V_int .^ 2) .- F1_int
    res_v = Vtt_int .- (model.c2^2) .* lap_v_int .+ model.sigma2 .* cap_v .+
            model.m2 .* V_int .+ model.k2 .* (V_int .^ 3) .+ model.eta .* V_int .* (U_int .^ 2) .- F2_int
    return sum(res_u .^ 2) / length(res_u) + sum(res_v .^ 2) / length(res_v)
end

function benchmark(n_epochs, use_fd_laplacian; Nx=14, Ny=14, Nz=14, Nt=28, hidden_dim=128, num_layers=5)
    Random.seed!(0)
    model = Model3D.SpectralFPINN3D(Nx=Nx, Ny=Ny, Nz=Nz, Nt=Nt, alpha=0.4, beta=0.7,
                                     hidden_dim=hidden_dim, num_layers=num_layers)
    params = init_params(model)
    _, _, F1, F2 = compute_3d_analytical_sources(model)
    loss_fn = use_fd_laplacian ? (p -> compute_loss_fd_laplacian(p, model, F1, F2)[1]) :
                                  (p -> compute_loss(p, model, F1, F2)[1])
    Zygote.withgradient(loss_fn, params)  # warmup
    t = @elapsed for _ in 1:n_epochs
        Zygote.withgradient(loss_fn, params)
    end
    return t / n_epochs
end

if abspath(PROGRAM_FILE) == @__FILE__
    Nx = Ny = Nz = 14; Nt = 28
    n_pts = (Nx+1)*(Ny+1)*(Nz+1)*(Nt+1)
    @printf("Grid: %dx%dx%dx%d = %d points, hidden=128, layers=5\n", Nx+1,Ny+1,Nz+1,Nt+1,n_pts)
    println("Timing per-epoch cost of a full loss+gradient evaluation...\n")

    t_spectral = benchmark(10, false; Nx=Nx,Ny=Ny,Nz=Nz,Nt=Nt)
    @printf("Spectral Laplacian:       %.1f ms/epoch\n", t_spectral*1000)

    t_fd = benchmark(5, true; Nx=Nx,Ny=Ny,Nz=Nz,Nt=Nt)
    @printf("Finite-diff Laplacian:    %.1f ms/epoch\n", t_fd*1000)

    @printf("\nSpeedup: %.2fx\n", t_fd/t_spectral)
    @printf("Per-epoch time reduction: %.1f%%\n", (1-t_spectral/t_fd)*100)
end
