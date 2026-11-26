@testitem "Nonlinear programs" begin
    using NonconvexInteriorPoint

    # Convex QP with an active inequality: min x²+y² s.t. x+y ≥ 1 → (0.5, 0.5)
    m = Model(x -> x[1]^2 + x[2]^2)
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    add_ineq_constraint!(m, x -> 1.0 - x[1] - x[2])
    r = optimize(m, IPMAlg(), [2.0, 2.0])
    @test r.status == :converged
    @test r.minimizer ≈ [0.5, 0.5] atol = 1e-5
    @test r.minimum ≈ 0.5 atol = 1e-5

    # Same via the native linear-constraint path
    m = Model(x -> x[1]^2 + x[2]^2)
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    add_linear_constraint!(m, [1.0 1.0], GreaterThan([1.0]))
    r = optimize(m, IPMAlg(), [2.0, 2.0])
    @test r.status == :converged
    @test r.minimizer ≈ [0.5, 0.5] atol = 1e-5

    # Equality constraint: min x² s.t. x = 3
    m = Model(x -> x[1]^2)
    addvar!(m, [-10.0], [10.0])
    add_eq_constraint!(m, x -> x[1] - 3.0)
    r = optimize(m, IPMAlg(), [0.0])
    @test r.status == :converged
    @test r.minimizer ≈ [3.0] atol = 1e-5
    @test r.minimum ≈ 9.0 atol = 1e-5

    # Two-variable equality-constrained least squares
    m = Model(x -> (x[1] - 1)^2 + (x[2] - 1)^2)
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    add_eq_constraint!(m, x -> x[1] + 2x[2] - 3.0)
    r = optimize(m, IPMAlg(), [0.0, 0.0])
    @test r.status == :converged
    @test r.minimizer ≈ [1.0, 1.0] atol = 1e-5

    # Nonconvex objective (Himmelblau): local minimum at (3, 2)
    m = Model(x -> (x[1]^2 + x[2] - 11)^2 + (x[1] + x[2]^2 - 7)^2)
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    r = optimize(m, IPMAlg(), [2.5, 2.0])
    @test r.status == :converged
    @test r.minimizer ≈ [3.0, 2.0] atol = 1e-4
    @test r.minimum ≈ 0.0 atol = 1e-6

    # Nonlinear inequality constraint (projection onto a disk)
    m = Model(x -> (x[1] - 1.0)^2 + (x[2] - 2.0)^2)
    addvar!(m, [-10.0, -10.0], [10.0, 10.0])
    add_ineq_constraint!(m, x -> x[1]^2 + x[2]^2 - 3.0)
    r = optimize(m, IPMAlg(), [0.0, 0.0])
    @test r.status == :converged
    @test r.minimizer ≈ [0.7745966692414834, 1.5491933384829668] atol = 1e-4

    # Krylov (matrix-free KKT) solver on the same problem
    r = optimize(m, IPMAlg(), [0.0, 0.0]; options=IPMOptions(linear_solver=:krylov))
    @test r.status == :converged
    @test r.minimizer ≈ [0.7745966692414834, 1.5491933384829668] atol = 1e-4

    # Box-constrained problem with the lower bound active
    m = Model(x -> (x[1] - 2.0)^2 + (x[2] + 1.0)^2)
    addvar!(m, [0.0, -Inf], [Inf, Inf])
    r = optimize(m, IPMAlg(), [5.0, 5.0])
    @test r.status == :converged
    @test r.minimizer ≈ [2.0, -1.0] atol = 1e-4

    # Iteration count and function-call counters are populated
    @test r.iterations > 0
    @test r.fcalls > 0
end
