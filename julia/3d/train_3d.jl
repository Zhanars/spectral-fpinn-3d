"""
Training script for the 3D Spectral fPINN, direct Julia port of train_3d.py
(Experiment 2's manufactured-solution benchmark).
"""

include(joinpath(@__DIR__, "Model3D.jl"))
using .Model3D
using Zygote
using Optimisers
using Optim
using Random
using Printf
using SpecialFunctions: gamma as gammafn

"""Manufactured solution + analytical RHS sources, direct port of
train_3d.compute_3d_analytical_sources. Returns U_exact, V_exact, F1, F2, each
(Nt+1,Nx+1,Ny+1,Nz+1), plus the grids."""
function compute_3d_analytical_sources(model::Model3D.SpectralFPINN3D)
    T4, X4, Y4, Z4 = Model3D.build_grid4d(model)
    sp_u = sin.(π .* X4) .* sin.(π .* Y4) .* sin.(π .* Z4)
    sp_v = sin.(2π .* X4) .* sin.(π .* Y4) .* sin.(π .* Z4)

    U_exact = (T4 .^ 3) .* sp_u
    V_exact = (T4 .^ 3) .* sp_v

    Utt = 6.0 .* T4 .* sp_u
    Vtt = 6.0 .* T4 .* sp_v

    lap_u = -3.0 * π^2 .* U_exact
    lap_v = -6.0 * π^2 .* V_exact

    cap_u = (6.0 / gammafn(4.0 - model.alpha)) .* (T4 .^ (3.0 - model.alpha)) .* sp_u
    cap_v = (6.0 / gammafn(4.0 - model.beta)) .* (T4 .^ (3.0 - model.beta)) .* sp_v

    F1 = Utt .- (model.c1^2) .* lap_u .+ model.sigma1 .* cap_u .+
         model.m1 .* U_exact .+ model.k1 .* (U_exact .^ 3) .+ model.eta .* U_exact .* (V_exact .^ 2)
    F2 = Vtt .- (model.c2^2) .* lap_v .+ model.sigma2 .* cap_v .+
         model.m2 .* V_exact .+ model.k2 .* (V_exact .^ 3) .+ model.eta .* V_exact .* (U_exact .^ 2)

    return U_exact, V_exact, F1, F2
end

relative_l2_error(pred, exact) = sqrt(sum((pred .- exact) .^ 2)) / sqrt(sum(exact .^ 2))

function train_3d_spectral_fpinn(; epochs=350, lr=2e-3, Nx=8, Ny=8, Nz=8, Nt=16,
                                  hidden_dim=64, num_layers=4, lbfgs_max_iter=25, seed=0)
    println("="^75)
    println("3D Spectral Fractional PINN (Julia port)")
    println("="^75)
    Random.seed!(seed)

    total_pts = (Nx + 1) * (Ny + 1) * (Nz + 1) * (Nt + 1)
    @printf("Grid: %dx%dx%dx%d = %d points\n", Nx+1, Ny+1, Nz+1, Nt+1, total_pts)

    model = Model3D.SpectralFPINN3D(Nx=Nx, Ny=Ny, Nz=Nz, Nt=Nt, alpha=0.4, beta=0.7,
                                     hidden_dim=hidden_dim, num_layers=num_layers)
    params = init_params(model)
    U_exact, V_exact, F1, F2 = compute_3d_analytical_sources(model)

    loss_fn(p) = compute_loss(p, model, F1, F2)[1]

    println("--- Adam ---")
    t0 = time()
    opt_state = Optimisers.setup(Optimisers.Adam(lr), params)
    print_every = max(50, epochs ÷ 40)
    for epoch in 1:epochs
        loss, grad = Zygote.withgradient(loss_fn, params)
        opt_state, params = Optimisers.update!(opt_state, params, grad[1])
        if epoch % print_every == 0 || epoch == 1
            _, U, V = compute_loss(params, model, F1, F2)
            eu, ev = relative_l2_error(U, U_exact), relative_l2_error(V, V_exact)
            @printf("Epoch %4d/%d | loss=%.4e | L2 u=%.4e L2 v=%.4e\n", epoch, epochs, loss, eu, ev)
        end
    end
    @printf("Adam done in %.2fs\n", time() - t0)

    if lbfgs_max_iter > 0
        println("--- L-BFGS ---")
        theta0, shapes = flatten_params(params)
        f(theta) = compute_loss(unflatten_params(theta, shapes), model, F1, F2)[1]
        g!(G, theta) = copyto!(G, Zygote.gradient(f, theta)[1])
        t1 = time()
        res = Optim.optimize(f, g!, theta0, Optim.LBFGS(m=50),
                              Optim.Options(iterations=lbfgs_max_iter, g_tol=1e-7, show_trace=false))
        params = unflatten_params(Optim.minimizer(res), shapes)
        @printf("L-BFGS done in %.2fs (%d iters)\n", time() - t1, Optim.iterations(res))
    end

    elapsed = time() - t0
    final_loss, U, V = compute_loss(params, model, F1, F2)
    final_l2_u, final_l2_v = relative_l2_error(U, U_exact), relative_l2_error(V, V_exact)
    println("="^75)
    @printf("3D Training Complete in %.2f seconds\n", elapsed)
    @printf("Final Loss: %.4e | Rel L2 u: %.2f%% | Rel L2 v: %.2f%%\n", final_loss, final_l2_u*100, final_l2_v*100)
    println("="^75)
    return final_l2_u, final_l2_v, elapsed
end

if abspath(PROGRAM_FILE) == @__FILE__
    train_3d_spectral_fpinn(epochs=350)
end
