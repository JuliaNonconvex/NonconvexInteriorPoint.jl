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

    # JuMP-converted models: the whole inequality block carries a single
    # label set through the conversion kwargs; selecting it slacks the
    # block (matching slacks=true here, since every inequality is in it).
    using JuMP
    jm = JuMP.Model()
    @variable(jm, -5.0 <= x[1:2] <= 5.0, start = 1.0)
    @NLobjective(jm, Min, x[1]^2 + x[2]^2)
    @NLconstraint(jm, x[1]^2 + x[2] - 1.0 <= 0)
    @NLconstraint(jm, x[1] * x[2] - 0.2 <= 0)
    @NLconstraint(jm, x[1] + 2.0 * x[2] - 0.4 == 0)
    dm = NonconvexCore.DictModel(jm; ineq_flags=:slackme)
    rj_hy = optimize(dm, IPMAlg(); options=IPMOptions(barrier=BarrierSelection(labels=:slackme)))
    rj_sl = optimize(dm, IPMAlg(); options=IPMOptions(slacks=true))
    @test rj_hy.status == :converged
    @test rj_sl.status == :converged
    @test collect(values(rj_hy.minimizer)) ≈ collect(values(rj_sl.minimizer)) atol = 1e-4
end
