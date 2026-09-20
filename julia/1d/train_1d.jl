"""
Training script for the 1D Spectral fPINN, direct Julia port of
_archive_1d_2d/code/train.py (Experiment 1's manufactured-solution benchmark).
"""

include(joinpath(@__DIR__, "Model1D.jl"))
using .Model1D
using Zygote
using Optimisers
using Optim
using SpecialFunctions: gamma as gammafn
using Random
using Printf

function make_manufactured_sources(alpha, beta; c1=1.0, c2=1.0, m1=1.0, m2=1.0,
                                    k1=1.0, k2=1.0, sigma1=1.0, sigma2=1.0, eta=1.0)
    gamma_factor_u = 6.0 / gammafn(4.0 - alpha)
    gamma_factor_v = 6.0 / gammafn(4.0 - beta)

    function f1_func(x, t)
        sin_pix = sin(π * x)
        sin_2pix = sin(2.0 * π * x)
        u = t^3 * sin_pix
        v = t^3 * sin_2pix
        u_tt = 6.0 * t * sin_pix
        u_xx = -(π^2) * (t^3) * sin_pix
        caputo_u = gamma_factor_u * (t^(3.0 - alpha)) * sin_pix
        return u_tt - (c1^2) * u_xx + sigma1 * caputo_u + m1 * u + k1 * (u^3) + eta * u * (v^2)
    end

    function f2_func(x, t)
        sin_pix = sin(π * x)
        sin_2pix = sin(2.0 * π * x)
        u = t^3 * sin_pix
        v = t^3 * sin_2pix
        v_tt = 6.0 * t * sin_2pix
        v_xx = -4.0 * (π^2) * (t^3) * sin_2pix
        caputo_v = gamma_factor_v * (t^(3.0 - beta)) * sin_2pix
        return v_tt - (c2^2) * v_xx + sigma2 * caputo_v + m2 * v + k2 * (v^3) + eta * v * (u^2)
    end

    return f1_func, f2_func
end

relative_l2_error(pred, exact) = sqrt(sum((pred .- exact) .^ 2)) / (sqrt(sum(exact .^ 2)) + 1e-12)

function train_spectral_fpinn_1d(; N=24, Nt=40, T=1.0, L=1.0, alpha=0.4, beta=0.7,
                                  hidden_dim=64, num_layers=4, adam_epochs=2000,
                                  lbfgs_max_iter=500, lr=2e-3, seed=0)
    Random.seed!(seed)
    model = SpectralFPINN1D(N=N, Nt=Nt, T=T, L=L, hidden_dim=hidden_dim,
                             num_layers=num_layers, alpha=alpha, beta=beta)
    params = init_params(model)

    f1_func, f2_func = make_manufactured_sources(alpha, beta)

    Nt1, N1 = Nt + 1, N + 1
    U_exact = [(model.t_grid[i]^3) * sin(π * model.x_cgl[j]) for i in 1:Nt1, j in 1:N1]
    V_exact = [(model.t_grid[i]^3) * sin(2π * model.x_cgl[j]) for i in 1:Nt1, j in 1:N1]

    loss_fn(p) = compute_pde_loss(p, model, f1_func, f2_func)[1]

    @printf("=== 1D Spectral fPINN (Julia) === N=%d Nt=%d alpha=%.2f beta=%.2f\n", N, Nt, alpha, beta)

    println("--- Stage 1: Adam ---")
    t0 = time()
    opt_state = Optimisers.setup(Optimisers.Adam(lr), params)
    for epoch in 1:adam_epochs
        loss, grad = Zygote.withgradient(loss_fn, params)
        opt_state, params = Optimisers.update!(opt_state, params, grad[1])
        Optimisers.adjust!(opt_state, lr * 0.999^epoch)
        if epoch % 200 == 0 || epoch == 1
            U, V = evaluate_grid(params, model)
            eu, ev = relative_l2_error(U, U_exact), relative_l2_error(V, V_exact)
            @printf("Adam %5d/%d | loss=%.4e | L2 u=%.4e L2 v=%.4e\n", epoch, adam_epochs, loss, eu, ev)
        end
    end
    adam_time = time() - t0
    @printf("Adam done in %.2fs\n", adam_time)

    if lbfgs_max_iter > 0
        println("--- Stage 2: L-BFGS ---")
        theta0, shapes = flatten_params(params)
        f(theta) = loss_fn(unflatten_params(theta, shapes))
        g!(G, theta) = copyto!(G, Zygote.gradient(f, theta)[1])
        lbfgs_t0 = time()
        res = Optim.optimize(f, g!, theta0, Optim.LBFGS(m=50),
                              Optim.Options(iterations=lbfgs_max_iter, g_tol=1e-7, show_trace=false))
        params = unflatten_params(Optim.minimizer(res), shapes)
        @printf("L-BFGS done in %.2fs (%d iterations, converged=%s)\n",
                time() - lbfgs_t0, Optim.iterations(res), Optim.converged(res))
    end

    total_time = time() - t0
    U_final, V_final = evaluate_grid(params, model)
    final_l2_u = relative_l2_error(U_final, U_exact)
    final_l2_v = relative_l2_error(V_final, V_exact)
    max_err_u = maximum(abs.(U_final .- U_exact))
    max_err_v = maximum(abs.(V_final .- V_exact))

    @printf("\n=== Final ===\nu: RelL2=%.4e MaxAbs=%.4e\nv: RelL2=%.4e MaxAbs=%.4e\nTotal time=%.2fs\n",
            final_l2_u, max_err_u, final_l2_v, max_err_v, total_time)

    return final_l2_u, final_l2_v, total_time
end

if abspath(PROGRAM_FILE) == @__FILE__
    train_spectral_fpinn_1d(N=20, Nt=32, alpha=0.4, beta=0.7, hidden_dim=64,
                             num_layers=4, adam_epochs=1200, lbfgs_max_iter=250, lr=2e-3)
end
