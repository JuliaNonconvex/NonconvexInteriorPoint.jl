@testitem "Mixed JuMP/native models" begin
    using NonconvexInteriorPoint
    using NonconvexCore
    using JuMP

    # A JuMP model with a nonlinear constraint, then NATIVE constraints
    # added to the converted DictModel: the problem mixes the two worlds.
    # The JuMP rows must keep the evaluator fast path, the native rows
    # use AD.
    jm = JuMP.Model()
    @variable(jm, -5.0 <= x[1:2] <= 5.0, start=1.0)
    @objective(jm, Min, x[1]^2 + x[2]^2)
    @constraint(jm, x[1]^2 + x[2] - 1.0 <= 0)  # JuMP side
    dm = NonconvexCore.DictModel(jm)
    add_ineq_constraint!(dm, x -> [x[:x][1] * x[:x][2] - 0.2])  # native side
    add_eq_constraint!(dm, x -> [x[:x][1] + 2.0 * x[:x][2] - 0.4])  # native eq

    vm, _, _ = NonconvexCore.tovecmodel(dm)
    ipm = NonconvexInteriorPoint.build_ipm_model(vm, IPMOptions())
    @test ipm.jump !== nothing # the evaluator fast path stays active
    @test ipm.jump_obj
    @test ipm.n_jump_ineq == 1

    r_bar = optimize(dm, IPMAlg())
    r_sl = optimize(dm, IPMAlg(); options=IPMOptions(slacks=true))
    @test r_bar.status == :converged
    @test r_sl.status == :converged
    @test collect(values(r_bar.minimizer))[1] ≈ [0.08, 0.16] atol = 1e-4
    @test collect(values(r_sl.minimizer))[1] ≈ [0.08, 0.16] atol = 1e-4

    # native objective + JuMP constraints (the other mix direction)
    jm2 = JuMP.Model()
    @variable(jm2, -5.0 <= y[1:2] <= 5.0, start=1.0)
    @constraint(jm2, y[1]^2 + y[2] - 1.0 <= 0)
    @constraint(jm2, y[1] * y[2] - 0.2 <= 0)
    dm2 = NonconvexCore.DictModel(jm2)
    set_objective!(dm2, x -> (x[:y][1] - 0.1)^2 + (x[:y][2] - 0.2)^2)
    vm2, _, _ = NonconvexCore.tovecmodel(dm2)
    ipm2 = NonconvexInteriorPoint.build_ipm_model(vm2, IPMOptions())
    @test ipm2.jump !== nothing
    @test !ipm2.jump_obj
    r2 = optimize(dm2, IPMAlg())
    r3 = optimize(dm2, IPMAlg(); options=IPMOptions(slacks=true))
    @test r2.status == :converged
    @test r3.status == :converged
    @test collect(values(r2.minimizer))[1] ≈ [0.1, 0.2] atol = 1e-4
    @test collect(values(r3.minimizer))[1] ≈ [0.1, 0.2] atol = 1e-4
end
