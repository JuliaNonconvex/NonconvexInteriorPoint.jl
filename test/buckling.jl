@testitem "Buckling-constrained truss (global semidefinite constraint)" begin
    using NonconvexInteriorPoint, NonconvexSemidefinite, NonconvexCore
    using NonconvexAugLagLab, NonconvexIpopt
    using LinearAlgebra, Test, Random

    # Statically indeterminate 3-bar truss with a global buckling constraint:
    #
    #   A = (0,0), B = (1,0), D = (0.5,-1) fixed, tip C = (0.5,1) free,
    #   bars A-C, B-C, D-C, reference load P = (0,-1) at C, E = 1.
    #
    #   min Σ lᵢ xᵢ  s.t.  S(x) = K(x) + c̄·G(x, u(x)) ⪰ 0,  x ≥ x_min
    #
    # where K is the (per-unit-area) elastic stiffness, u = K⁻¹P the
    # displacement, G the geometric stiffness assembled from the axial
    # forces (compression < 0) and c̄ = 10 the required load factor. The
    # statical indeterminacy makes G depend on the design, so the global
    # buckling LMI is genuinely nonconvex in x.
    const TRUSS_NODES = [0.0 0.0; 1.0 0.0; 0.5 -1.0; 0.5 1.0]
    const TRUSS_BARS = [(1, 4), (2, 4), (3, 4)]
    const TRUSS_DOFS = Dict(1 => [0, 0], 2 => [0, 0], 3 => [0, 0], 4 => [1, 2])
    const TRUSS_P = [0.0, -1.0]
    const TRUSS_L = [norm(TRUSS_NODES[b, :] - TRUSS_NODES[a, :]) for (a, b) in TRUSS_BARS]
    const TRUSS_DIRS = [
        (TRUSS_NODES[b, :] - TRUSS_NODES[a, :]) ./ TRUSS_L[i] for
        (i, (a, b)) in enumerate(TRUSS_BARS)
    ]
    const TRUSS_CBAR = 10.0

    function buckling_state(x)
        # functional (mutation-free) assembly: the barrier gradients
        # differentiate through this closure
        K = foldl(
            +,
            [
                x[i] .* ([c[1]^2 c[1] * c[2]; c[1] * c[2] c[2]^2] ./ TRUSS_L[i]) for
                (i, c) in enumerate(TRUSS_DIRS)
            ],
        )
        u = K \ TRUSS_P
        G = foldl(
            +,
            [
                ((x[i] * dot(u, TRUSS_DIRS[i]) / TRUSS_L[i]) / TRUSS_L[i]) .*
                [c[2]^2 -c[1] * c[2]; -c[1] * c[2] c[1]^2] for
                (i, c) in enumerate(TRUSS_DIRS)
            ],
        )
        return K + TRUSS_CBAR .* G, u
    end

    function make_model(; flags=Set{Symbol}())
        m = Model(x -> sum(TRUSS_L .* x))
        addvar!(m, fill(0.01, 3), fill(50.0, 3))
        add_set_constraint!(m, x -> buckling_state(x)[1], PSDCone(2); flags=flags)
        return m
    end

    # The cheapest feasible design has the buckling constraint active:
    # x ≈ (22.4, 22.4, 0.01) with volume ≈ 50 (found by grid search; the
    # unconstrained volume minimum x → 0 violates the LMI by the sway
    # buckling mode).
    @test minimum(eigvals(Symmetric(buckling_state([22.4, 22.4, 0.01])[1]))) >= -1e-6
    @test minimum(eigvals(Symmetric(buckling_state([10.0, 10.0, 0.01])[1]))) < 0

    # dense oracle
    m = make_model()
    r = optimize(m, IPMAlg(), [25.0, 25.0, 5.0])
    @test r.status == :converged
    S, _ = buckling_state(r.minimizer)
    @test minimum(eigvals(Symmetric(S))) >= -1e-6
    @test minimum(eigvals(Symmetric(S))) <= 1e-3 # the buckling LMI is active
    @test r.minimum ≈ 50.0 rtol = 5e-2

    # matrix-free oracle (label-selected; feature-gated: requires the PSD
    # barrier oracles added to NonconvexSemidefinite after v0.4.6)
    if isdefined(NonconvexSemidefinite, :PSDBarrierOracle)
        m = make_model(; flags=Set([:matrixfree]))
        r = optimize(
            m,
            IPMAlg(),
            [25.0, 25.0, 5.0];
            options=IPMOptions(
                evaluation=NonconvexInteriorPoint.IPMEvaluationSelection(
                    matrixfree_all=true
                ),
            ),
        )
        @test r.status == :converged
        S, _ = buckling_state(r.minimizer)
        @test minimum(eigvals(Symmetric(S))) >= -1e-5
    end

    # AugLagAlg comparison: same problem, same optimal volume
    m = make_model()
    r = NonconvexCore.optimize(m, AugLagAlg(), [25.0, 25.0, 5.0])
    S, _ = buckling_state(r.minimizer)
    @test minimum(eigvals(Symmetric(S))) >= -1e-5
    @test sum(TRUSS_L .* r.minimizer) ≈ 50.0 rtol = 1e-1
end
