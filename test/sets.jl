@testitem "Set constraints" begin
    using NonconvexInteriorPoint

    # Order set (converted to a slacked inequality): min x²+y² s.t. x+y ≥ 1
    m = Model(x -> x[1]^2 + x[2]^2)
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    add_set_constraint!(m, x -> [x[1] + x[2]], GreaterThan([1.0]))
    r = optimize(m, IPMAlg(), [2.0, 2.0])
    @test r.status == :converged
    @test r.minimizer ≈ [0.5, 0.5] atol = 1e-5

    # LessThan set
    m = Model(x -> x[1]^2 + x[2]^2)
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    add_set_constraint!(m, x -> [-x[1] - x[2]], LessThan([-1.0]))
    r = optimize(m, IPMAlg(), [2.0, 2.0])
    @test r.status == :converged
    @test r.minimizer ≈ [0.5, 0.5] atol = 1e-5

    # Simplex constraint: min ||x||² over the 3-simplex → (1/3, 1/3, 1/3)
    m = Model(x -> sum(x .^ 2))
    addvar!(m, [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])
    add_set_constraint!(m, x -> x, Simplex(3))
    r = optimize(m, IPMAlg(), [0.5, 0.3, 0.2])
    @test r.status == :converged
    @test r.minimizer ≈ fill(1 / 3, 3) atol = 1e-4

    # Product of order sets
    m = Model(x -> sum(x .^ 2))
    addvar!(m, [-2.0, -2.0], [2.0, 2.0])
    add_set_constraint!(
        m, x -> (x[1], x[2]), SetProduct((GreaterThan(0.5), GreaterThan(0.5)))
    )
    r = optimize(m, IPMAlg(), [1.0, 1.0])
    @test r.status == :converged
    @test r.minimizer ≈ [0.5, 0.5] atol = 1e-5
end

@testitem "Semidefinite constraints" begin
    using NonconvexInteriorPoint, NonconvexSemidefinite, LinearAlgebra, Random

    # Interior-optimum log-determinant problem: the PSD barrier is active
    # but the optimum is strictly interior, keeping the problem well
    # conditioned for the primal barrier method.
    #
    #   min -log det X(x) + 0.5 ||x - xref||²,  X(x) = [x1 x2; x2 x3] ⪰ 0
    #
    # At the solution the gradient of the original objective vanishes.
    function make_model(; flags=Set{Symbol}())
        m = Model(
            x -> -logdet([x[1] x[2]; x[2] x[3]]) + 0.5 * sum((x .- [2.0, 0.5, 2.0]) .^ 2)
        )
        addvar!(m, [0.1, -Inf, 0.1], [Inf, Inf, Inf])
        add_set_constraint!(m, x -> [x[1] x[2]; x[2] x[3]], PSDCone(2); flags=flags)
        return m
    end

    # dense oracle
    m = make_model()
    r = optimize(m, IPMAlg(), [2.5, 0.5, 2.5])
    @test r.status == :converged
    X = [r.minimizer[1] r.minimizer[2]; r.minimizer[2] r.minimizer[3]]
    @test all(eigvals(Symmetric(X)) .> 0) # strictly feasible
    Xinv = inv(X)
    g = [
        -Xinv[1, 1] + (r.minimizer[1] - 2.0),
        -2Xinv[1, 2] + (r.minimizer[2] - 0.5),
        -Xinv[2, 2] + (r.minimizer[3] - 2.0),
    ]
    @test maximum(abs, g) <= 1e-3

    # matrix-free oracle (label-selected)
    m = make_model(; flags=Set([:matrixfree]))
    r = optimize(m, IPMAlg(), [2.5, 0.5, 2.5])
    @test r.status == :converged
    X = [r.minimizer[1] r.minimizer[2]; r.minimizer[2] r.minimizer[3]]
    @test all(eigvals(Symmetric(X)) .> 0)

    # trace-estimated oracle (label-selected, randomized)
    m = make_model(; flags=Set([:trace]))
    r = optimize(
        m,
        IPMAlg(),
        [2.5, 0.5, 2.5];
        options=IPMOptions(
            trace_nprobes=20, trace_lanczos_deg=20, rng=Random.MersenneTwister(7)
        ),
    )
    @test r.status == :converged
    X = [r.minimizer[1] r.minimizer[2]; r.minimizer[2] r.minimizer[3]]
    @test all(eigvals(Symmetric(X)) .> 0)
    Xinv = inv(X)
    g = [
        -Xinv[1, 1] + (r.minimizer[1] - 2.0),
        -2Xinv[1, 2] + (r.minimizer[2] - 0.5),
        -Xinv[2, 2] + (r.minimizer[3] - 2.0),
    ]
    @test maximum(abs, g) <= 1e-2

    # Unsupported barrier modes throw an informative error
    @test_throws ArgumentError NonconvexInteriorPoint._psd_barrier(
        x -> [x[1] x[2]; x[2] x[3]],
        NonconvexCore.PSDCone(2),
        [2.5, 0.5, 2.5],
        Val(:bogus),
        IPMOptions(),
    )
end
