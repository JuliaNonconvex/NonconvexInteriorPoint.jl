@testitem "Unconstrained convex QP" begin
    using NonconvexInteriorPoint

    m = Model(x -> (x[1] - 1.0)^2 + (x[2] - 2.0)^2)
    addvar!(m, [-10.0, -10.0], [10.0, 10.0])
    r = optimize(m, IPMAlg(), [0.0, 0.0])
    @test r.status == :converged
    @test r.minimizer ≈ [1.0, 2.0] atol = 1e-4
    @test r.minimum ≈ 0.0 atol = 1e-6
end
