@testitem "Feasibility stage" begin
    using NonconvexInteriorPoint

    # The feasibility stage runs before the barrier iterations whenever the
    # initial point violates inequalities, bounds or equalities, so
    # infeasible starts converge to the feasible optimum.
    function qp(; start)
        m = Model(x -> x[1]^2 + x[2]^2)
        addvar!(m, [-5.0, -5.0], [5.0, 5.0])
        add_ineq_constraint!(m, x -> 1.0 - x[1] - x[2]) # x1 + x2 ≥ 1
        return optimize(m, IPMAlg(), start)
    end

    # mildly infeasible start
    r = qp(; start=[0.0, 0.0])
    @test r.status == :converged
    @test r.minimizer ≈ [0.5, 0.5] atol = 1e-4

    # deeply infeasible start
    r = qp(; start=[10.0, -10.0])
    @test r.status == :converged
    @test r.minimizer ≈ [0.5, 0.5] atol = 1e-4

    # equality + bound violations at the start (non-degenerate: the
    # equality leaves an interior optimum [1.25, 1.25])
    m = Model(x -> sum(x .^ 2))
    addvar!(m, [1.0, 1.0], [5.0, 5.0])
    add_eq_constraint!(m, x -> x[1] + x[2] - 2.5)
    r = optimize(m, IPMAlg(), [0.0, 0.0])
    @test r.status == :converged
    @test r.minimizer ≈ [1.25, 1.25] atol = 1e-4

    # a feasible start skips the stage and behaves identically
    r = qp(; start=[1.0, 1.0])
    @test r.status == :converged
    @test r.minimizer ≈ [0.5, 0.5] atol = 1e-4

    # disabling the stage still converges (the smoothly extended barrier
    # restores feasibility on its own)
    m = Model(x -> x[1]^2 + x[2]^2)
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    add_ineq_constraint!(m, x -> 1.0 - x[1] - x[2])
    r = optimize(m, IPMAlg(), [0.0, 0.0]; options=IPMOptions(feasibility_max_iter=0))
    @test r.status == :converged
    @test r.minimizer ≈ [0.5, 0.5] atol = 1e-4

    # the slacked path also benefits from the stage (inequality-infeasible
    # start; the slacks then handle the inequality in the usual way)
    m = Model(x -> x[1]^2 + x[2]^2)
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    add_ineq_constraint!(m, x -> 1.0 - x[1] - x[2])
    r = optimize(m, IPMAlg(), [0.0, 0.0]; options=IPMOptions(slacks=true))
    @test r.status == :converged
    @test r.minimizer ≈ [0.5, 0.5] atol = 1e-4
end
