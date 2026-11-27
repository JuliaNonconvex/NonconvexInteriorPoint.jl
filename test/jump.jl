@testitem "JuMP models (MOI evaluator derivatives)" begin
    using NonconvexInteriorPoint, NonconvexCore, NonconvexIpopt, JuMP
    using LinearAlgebra, Test

    # A JuMP model converted through the shared MOI evaluator: the first-
    # and second-order derivatives come from the evaluator (sparse
    # reverse-mode), not from generic AD over the expression tree.
    function make_model(; start=[1.0, 1.5, 1.5])
        jm = JuMP.Model()
        JuMP.@variable(jm, -5 <= x[1:2] <= 5, start=start[1])
        JuMP.@variable(jm, 1 <= y <= 5, start=start[2])
        JuMP.@objective(jm, Min, (x[1] - 1)^2 + (x[2] - 2)^2 + (y - 1.5)^2)
        JuMP.@constraint(jm, x[1]^2 + x[2] - y <= 3.0)
        JuMP.@constraint(jm, x[1] + x[2] + y == 4.0)
        JuMP.@NLconstraint(jm, x[1] * x[2] - y <= 0.5)
        dm = NonconvexCore.DictModel(jm)
        return dm
    end

    # the converted model carries the shared evaluator
    vm, _, _ = NonconvexCore.tovecmodel(make_model())
    @test NonconvexCore.jump_evaluator(vm) !== nothing

    # the true solution: the equality pins x1+x2+y = 4 and the inequalities
    # are inactive → the projection of (1, 2, 1.5) onto the equality
    sol = [5 / 6, 11 / 6, 4 / 3]
    ks = collect(keys(NonconvexCore.getmin(make_model())))
    flat(d) = vcat((vec(collect(d[k])) for k in ks)...)

    # IPMAlg (primal barrier, default) from a feasible start
    dm = make_model()
    r = optimize(dm, IPMAlg(), NonconvexCore.getinit(dm))
    @test r.status == :converged
    @test flat(r.minimizer) ≈ sol atol = 1e-4

    # Ipopt through NonconvexCore agrees
    dm = make_model()
    r = optimize(
        dm, IpoptAlg(), NonconvexCore.getinit(dm); options=IpoptOptions(print_level=0)
    )
    @test flat(r.minimizer) ≈ sol atol = 1e-4
end
