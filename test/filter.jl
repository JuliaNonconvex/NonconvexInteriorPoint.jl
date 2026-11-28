@testitem "Filter line search" begin
    using NonconvexInteriorPoint
    using NonconvexCore

    # A nonconvex-feasible-region problem: the filter must find the
    # unconstrained optimum (the feasible interior), not stall at the
    # constraint boundary.
    f = x -> [x[1]^2 + x[2] - 1.0, x[1] * x[2] - 0.2]
    m = Model(x -> x[1]^2 + x[2]^2)
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    add_ineq_constraint!(m, f)
    r = optimize(m, IPMAlg(), [2.0, 2.0]; options=IPMOptions(line_search=:filter))
    @test r.status == :converged
    @test r.minimizer ≈ [0.0, 0.0] atol = 1e-5

    # The filter agrees with the merit line search on a plain convex QP.
    m2 = Model(x -> (x[1] - 1.0)^2 + (x[2] - 2.0)^2)
    addvar!(m2, [-1.0, -1.0], [1.0, 3.0])
    r2 = optimize(m2, IPMAlg(), [0.0, 0.0]; options=IPMOptions(line_search=:filter))
    @test r2.status == :converged
    @test r2.minimizer ≈ [1.0, 2.0] atol = 1e-5

    # Equality-constrained problem (exercises the θ-measure and the
    # restoration path's gradient/Hessian plumbing).
    m3 = Model(x -> (x[1] - 1.0)^2 + (x[2] - 2.0)^2)
    addvar!(m3, [-10.0, -10.0], [10.0, 10.0])
    add_eq_constraint!(m3, x -> [x[1] + x[2] - 1.0])
    r3 = optimize(m3, IPMAlg(), [0.0, 0.0]; options=IPMOptions(line_search=:filter))
    @test r3.status == :converged
    @test r3.minimizer ≈ [0.0, 1.0] atol = 1e-5

    # Invalid line-search names are rejected.
    @test_throws ArgumentError IPMOptions(line_search=:bogus)
end
