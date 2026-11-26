@testitem "Trace estimation" begin
    using NonconvexInteriorPoint, NonconvexSemidefinite, LinearAlgebra, Random

    n = 40
    A = Symmetric(
        Matrix(
            qr(randn(n, n)).Q *
            Diagonal(range(1.0, 2.0, length=n)) *
            Matrix(qr(randn(n, n)).Q)',
        ),
    )
    A = Symmetric(A + n * I) # SPD, spectrum [n+1, n+2]
    f! = (y, x) -> mul!(y, A, x)
    rng = MersenneTwister(3)

    # Hutchinson trace estimate
    tr_est = NonconvexInteriorPoint.hutchinson_trace(f!, n, 2000, rng)
    @test tr_est ≈ tr(A) rtol = 0.05

    # Hutch++ trace estimate
    tr_est2 = NonconvexInteriorPoint.hutchpp_trace(f!, n, 40, rng)
    @test tr_est2 ≈ tr(A) rtol = 0.05

    # Stochastic Lanczos quadrature log-determinant estimate (lives in
    # NonconvexSemidefinite; feature-gated so the tests also pass against
    # the released NonconvexSemidefinite, which predates the estimator
    # suite)
    if isdefined(NonconvexSemidefinite, :slq_logdet)
        ld_est = NonconvexSemidefinite.slq_logdet(
            f!, n; lanczos_deg=60, nprobes=20, rng=rng
        )
        ld_true = logdet(A)
        @test ld_est ≈ ld_true rtol = 0.02
    end

    # More probes tighten the estimate: with this many probes the error is
    # bounded by a few standard deviations (σ ≈ 2·||A||_F/√n per probe, so
    # ≈ 8.5 at 50 probes and ≈ 1.3 at 2000 probes on this matrix).
    err_h = abs(NonconvexInteriorPoint.hutchinson_trace(f!, n, 50, rng) - tr(A))
    err_h_more = abs(NonconvexInteriorPoint.hutchinson_trace(f!, n, 2000, rng) - tr(A))
    @test err_h_more <= 8.0
    @test err_h <= 60.0
end

@testitem "Options and capabilities" begin
    using NonconvexInteriorPoint

    # defaults
    o = IPMOptions()
    @test o.linear_solver == :auto
    @test o.mu_init == 0.1
    @test o.mu_min == 1e-11
    @test o.tau == 0.995

    # validation
    @test_throws ArgumentError IPMOptions(linear_solver=:bogus)
    @test_throws ArgumentError IPMOptions(max_iter=-1)

    # evaluation selection from labels
    sel = IPMEvaluationSelection(matrixfree=:big, trace=[:huge, :big])
    @test :big in sel.matrixfree
    @test :huge in sel.trace
    @test :big in sel.trace

    # per-label mode resolution
    opts = IPMOptions(evaluation=sel)
    @test NonconvexInteriorPoint._mode(Set([:big]), opts) == :trace
    @test NonconvexInteriorPoint._mode(Set([:small]), opts) == :dense
    opts2 = IPMOptions(evaluation=IPMEvaluationSelection(matrixfree_all=true))
    @test NonconvexInteriorPoint._mode(Set{Symbol}(), opts2) == :matrixfree

    # capability declaration covers the supported features
    caps = NonconvexCore.solver_capabilities(IPMAlg())
    @test caps.linear_eq === true
    @test caps.nonlinear_ineq === true
    @test caps.sets_sd === true
    @test caps.sets_simplex === true
    @test caps.box === true
    @test caps.integer === false

    # integer models are rejected by the pre-check
    m = Model(x -> x[1]^2)
    addvar!(m, [0.0], [10.0]; integer=true)
    @test_throws ArgumentError optimize(m, IPMAlg(), [1.0])
end

@testitem "Linear constraint detection" begin
    using NonconvexInteriorPoint

    # structural LinearFunction detection (add_linear_constraint!)
    m = Model(x -> sum(x .^ 2))
    addvar!(m, [-2.0, -2.0], [2.0, 2.0])
    add_linear_constraint!(m, [1.0 2.0], LessThan([1.0]))
    vm, _, _ = NonconvexCore.tovecmodel(m, [0.0, 0.0])
    ipm = NonconvexInteriorPoint.build_ipm_model(vm, IPMOptions())
    @test ipm.linAineq !== nothing
    @test size(ipm.linAineq) == (1, 2)
    @test NonconvexInteriorPoint.n_ineq_nl(ipm) == 0
    @test NonconvexInteriorPoint.n_ineq_lin(ipm) == 1

    # Jacobian-constancy probe: a plain linear function goes native
    m = Model(x -> sum(x .^ 2))
    addvar!(m, [-2.0, -2.0], [2.0, 2.0])
    add_ineq_constraint!(m, x -> x[1] + 2x[2] - 1.0)
    vm, _, _ = NonconvexCore.tovecmodel(m, [0.0, 0.0])
    ipm = NonconvexInteriorPoint.build_ipm_model(vm, IPMOptions())
    @test ipm.linAineq !== nothing
    @test NonconvexInteriorPoint.n_ineq_nl(ipm) == 0

    # nonlinear constraints stay nonlinear
    m = Model(x -> sum(x .^ 2))
    addvar!(m, [-2.0, -2.0], [2.0, 2.0])
    add_ineq_constraint!(m, x -> x[1]^2 + 2x[2] - 1.0)
    vm, _, _ = NonconvexCore.tovecmodel(m, [0.0, 0.0])
    ipm = NonconvexInteriorPoint.build_ipm_model(vm, IPMOptions())
    @test ipm.linAineq === nothing
    @test NonconvexInteriorPoint.n_ineq_nl(ipm) == 1

    # equality linear function via a set
    m = Model(x -> sum(x .^ 2))
    addvar!(m, [-2.0, -2.0], [2.0, 2.0])
    add_linear_constraint!(m, [1.0 2.0], EqualTo([0.5]))
    vm, _, _ = NonconvexCore.tovecmodel(m, [0.0, 0.0])
    ipm = NonconvexInteriorPoint.build_ipm_model(vm, IPMOptions())
    @test ipm.linAeq !== nothing
    @test NonconvexInteriorPoint.n_eq_nl(ipm) == 0
end

@testitem "Constraint relaxation via labels" begin
    using NonconvexInteriorPoint

    # An equality constraint carrying the :relax label is moved into a
    # quadratic penalty instead of being enforced.
    m = Model(x -> (x[1] - 1.0)^2 + (x[2] - 2.0)^2)
    addvar!(m, [-10.0, -10.0], [10.0, 10.0])
    add_eq_constraint!(m, x -> x[1] + x[2] - 10.0; flags=Set([:relax]))
    r = optimize(
        m,
        IPMAlg(),
        [0.0, 0.0];
        options=IPMOptions(relaxed_labels=[:relax], relaxed_penalty_rho=1.0),
    )
    @test r.status == :converged
    # the relaxed constraint is not satisfied exactly
    @test abs(r.minimizer[1] + r.minimizer[2] - 10.0) > 1.0
    # the penalized optimum: min (x1-1)² + (x2-2)² + 0.5 (x1+x2-10)²
    @test r.minimizer ≈ [2.75, 3.75] atol = 1e-2
end

@testitem "Adjoint-based barrier gradients" begin
    using NonconvexInteriorPoint
    using NonconvexCore: tovecmodel

    # A vector inequality constraint: the barrier gradient must be assembled
    # from adjoints (J'(μ/s)) without materializing the Jacobian.
    f = x -> [x[1]^2 + x[2] - 1.0, x[1] * x[2] - 0.2]
    m = Model(x -> x[1]^2 + x[2]^2)
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    add_ineq_constraint!(m, f)
    vm, _, _ = tovecmodel(m, [0.5, 0.1])
    ipm = NonconvexInteriorPoint.build_ipm_model(vm, IPMOptions())

    # feasible probe point: the strict adjoint path is taken
    x = [0.5, 0.1]
    mu = 0.1
    g = zeros(2)
    NonconvexInteriorPoint.barrier_gradient!(g, ipm, x, mu, IPMOptions())
    # analytic: ∇f + J'(μ/s) + bound-barrier terms, s = -f(x)
    v = f(x)
    s = -v
    w = mu ./ s
    J = [2x[1] 1.0; x[2] x[1]]
    g_true = [2x[1], 2x[2]] + J' * w
    g_true .-= mu ./ (x .- (-5.0))
    g_true .+= mu ./ (5.0 .- x)
    @test g ≈ g_true atol = 1e-8

    # the full solve with the vector inequality constraint
    r = optimize(m, IPMAlg(), [2.0, 2.0])
    @test r.status == :converged
    @test all(f(r.minimizer) .<= 1e-6)
    @test r.minimizer ≈ [0.0, 0.0] atol = 1e-5
end
