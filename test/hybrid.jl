@testitem "Hybrid barrier/slacked selection" begin
    using NonconvexInteriorPoint
    using NonconvexCore

    # Per-constraint label selection (Core models): the labeled inequality
    # gets explicit slack/dual variables, the others stay in the barrier.
    f1 = x -> [x[1]^2 + x[2] - 1.0]
    f2 = x -> [x[1] * x[2] - 0.2]
    m = Model(x -> x[1]^2 + x[2]^2)
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    add_ineq_constraint!(m, f1, 0.0; flags=Set([:slackme]))
    add_ineq_constraint!(m, f2)
    add_eq_constraint!(m, x -> [x[1] + 2.0 * x[2] - 0.4])
    r_bar = optimize(m, IPMAlg(), [1.0, 1.0]; options=IPMOptions())
    r_sl = optimize(m, IPMAlg(), [1.0, 1.0]; options=IPMOptions(slacks=true))
    r_hy = optimize(
        m,
        IPMAlg(),
        [1.0, 1.0];
        options=IPMOptions(barrier=BarrierSelection(labels=:slackme)),
    )
    @test r_bar.status == :converged
    @test r_sl.status == :converged
    @test r_hy.status == :converged
    @test r_hy.minimizer ≈ r_bar.minimizer atol = 1e-5
    @test r_hy.minimizer ≈ r_sl.minimizer atol = 1e-5

    # Row-index selection (one row of a vector inequality).
    f3 = x -> [x[1]^2 + x[2] - 1.0, x[1] * x[2] - 0.2]
    m2 = Model(x -> x[1]^2 + x[2]^2)
    addvar!(m2, [-5.0, -5.0], [5.0, 5.0])
    add_ineq_constraint!(m2, f3)
    add_eq_constraint!(m2, x -> [x[1] + 2.0 * x[2] - 0.4])
    r2 = optimize(m2, IPMAlg(), [1.0, 1.0]; options=IPMOptions(slacked_ineq_rows=[1]))
    @test r2.status == :converged
    @test r2.minimizer ≈ [0.08, 0.16] atol = 1e-4
    r3 = optimize(m2, IPMAlg(), [1.0, 1.0]; options=IPMOptions(slacked_ineq_rows=[1, 2]))
    @test r3.status == :converged
    @test r3.minimizer ≈ [0.08, 0.16] atol = 1e-4
end
