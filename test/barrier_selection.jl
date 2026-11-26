@testitem "Barrier selection and types" begin
    using NonconvexInteriorPoint

    # Barrier types for the scalar barriers: :inv and :quad give the same
    # optimum as :log on a well-behaved QP.
    for btype in (:log, :inv, :quad)
        m = Model(x -> x[1]^2 + x[2]^2)
        addvar!(m, [-5.0, -5.0], [5.0, 5.0])
        add_ineq_constraint!(m, x -> 1.0 - x[1] - x[2])
        r = optimize(
            m,
            IPMAlg(),
            [2.0, 2.0];
            options=IPMOptions(
                barrier=NonconvexInteriorPoint.BarrierSelection(ineq_type=btype)
            ),
        )
        @test r.status == :converged
        @test r.minimizer ≈ [0.5, 0.5] atol = 5e-4
    end

    # Invalid barrier types are rejected.
    @test_throws ArgumentError NonconvexInteriorPoint.BarrierSelection(ineq_type=:bogus)
    @test_throws ArgumentError NonconvexInteriorPoint.BarrierSelection(sd_type=:bogus)

    # Slacked formulation (slacks = true): the inequality is enforced through
    # explicit slack variables.
    m = Model(x -> x[1]^2 + x[2]^2)
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    add_ineq_constraint!(m, x -> 1.0 - x[1] - x[2])
    r = optimize(m, IPMAlg(), [2.0, 2.0]; options=IPMOptions(slacks=true))
    @test r.status == :converged
    @test r.minimizer ≈ [0.5, 0.5] atol = 1e-4

    # Barrier exclusion (ineq = false): the inequality is slacked per
    # constraint (the per-category counterpart of slacks = true).
    m = Model(x -> x[1]^2 + x[2]^2)
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    add_ineq_constraint!(m, x -> 1.0 - x[1] - x[2])
    r = optimize(
        m,
        IPMAlg(),
        [2.0, 2.0];
        options=IPMOptions(barrier=NonconvexInteriorPoint.BarrierSelection(ineq=false)),
    )
    @test r.status == :converged
    @test r.minimizer ≈ [0.5, 0.5] atol = 1e-4

    # Label-based exclusion: the :hard constraint is slacked, the unlabeled
    # one is barriered. Check the build-level masks directly (the mixed
    # barrier/slacked endgame is covered by the separate slacks-mode and
    # barrier-exclusion solves above).
    m = Model(x -> x[1]^2 + x[2]^2)
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    add_ineq_constraint!(m, x -> x[1]^2 - x[2] - 0.5) # nonlinear, barriered
    add_ineq_constraint!(m, x -> x[1] - 0.8; flags=Set([:hard])) # linear, slacked
    add_ineq_constraint!(m, x -> x[1]^3 - x[2] + 0.2; flags=Set([:hard])) # nonlinear, slacked
    vm, _, _ = NonconvexCore.tovecmodel(m, [0.5, 0.5])
    ipm = NonconvexInteriorPoint.build_ipm_model(
        vm, IPMOptions(barrier=NonconvexInteriorPoint.BarrierSelection(labels=[:hard]))
    )
    @test ipm.nlineq_barrier == [true, false]
    @test ipm.lin_ineq_barrier == [false]

    # slacks = true excludes every inequality from the barrier
    ipm2 = NonconvexInteriorPoint.build_ipm_model(vm, IPMOptions(slacks=true))
    @test all(!, ipm2.nlineq_barrier)
    @test all(!, ipm2.lin_ineq_barrier)

    # ineq = false does the same per category
    ipm3 = NonconvexInteriorPoint.build_ipm_model(
        vm, IPMOptions(barrier=NonconvexInteriorPoint.BarrierSelection(ineq=false))
    )
    @test all(!, ipm3.nlineq_barrier)
    @test all(!, ipm3.lin_ineq_barrier)

    # Box exclusion: bounds are not enforced (the unconstrained optimum is
    # reached).
    m = Model(x -> x[1]^2 + x[2]^2)
    addvar!(m, [1.0, 1.0], [5.0, 5.0])
    r = optimize(
        m,
        IPMAlg(),
        [2.0, 2.0];
        options=IPMOptions(barrier=NonconvexInteriorPoint.BarrierSelection(box=false)),
    )
    @test r.status == :converged
    @test r.minimizer ≈ [0.0, 0.0] atol = 1e-4
end

@testitem "Inner optimizer" begin
    using NonconvexInteriorPoint, NonconvexIpopt

    # An inner optimizer (Ipopt) solves the barrier subproblems; the outer
    # barrier loop drives μ.
    m = Model(x -> x[1]^2 + x[2]^2)
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    add_ineq_constraint!(m, x -> 1.0 - x[1] - x[2])
    r = optimize(
        m,
        IPMAlg(),
        [2.0, 2.0];
        options=IPMOptions(inner=IpoptAlg(), inner_options=IpoptOptions(print_level=0)),
    )
    @test r.status in (:converged, :max_iter)
    @test r.minimizer ≈ [0.5, 0.5] atol = 1e-3

    # Nonconvex problem with the inner optimizer.
    m = Model(x -> (x[1]^2 + x[2] - 11)^2 + (x[1] + x[2]^2 - 7)^2)
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    r = optimize(
        m,
        IPMAlg(),
        [2.5, 2.0];
        options=IPMOptions(inner=IpoptAlg(), inner_options=IpoptOptions(print_level=0)),
    )
    @test r.status in (:converged, :max_iter)
    @test r.minimizer ≈ [3.0, 2.0] atol = 1e-3

    # The built-in inner solver (default) still solves the same problems.
    m = Model(x -> x[1]^2 + x[2]^2)
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    add_ineq_constraint!(m, x -> 1.0 - x[1] - x[2])
    r = optimize(m, IPMAlg(), [2.0, 2.0]; options=IPMOptions(inner_max_iter=50))
    @test r.status == :converged
    @test r.minimizer ≈ [0.5, 0.5] atol = 1e-4
end

@testitem "Semidefinite barrier types" begin
    using NonconvexInteriorPoint, NonconvexSemidefinite, LinearAlgebra, Random

    function make_model(; flags=Set{Symbol}())
        m = Model(
            x -> -logdet([x[1] x[2]; x[2] x[3]]) + 0.5 * sum((x .- [2.0, 0.5, 2.0]) .^ 2)
        )
        addvar!(m, [0.1, -Inf, 0.1], [Inf, Inf, Inf])
        add_set_constraint!(m, x -> [x[1] x[2]; x[2] x[3]], PSDCone(2); flags=flags)
        return m
    end

    # sd barrier types: :inverse (tr(X⁻¹)) and :inverse_square (tr(X⁻²)/2)
    # converge to an interior optimum. Feature-gated: they require the PSD
    # barrier oracles added to NonconvexSemidefinite after v0.4.6.
    if isdefined(NonconvexSemidefinite, :PSDBarrierOracle)
        for sdtype in (:inverse, :inverse_square)
            m = make_model()
            r = optimize(
                m,
                IPMAlg(),
                [2.5, 0.5, 2.5];
                options=IPMOptions(
                    barrier=NonconvexInteriorPoint.BarrierSelection(sd_type=sdtype)
                ),
            )
            @test r.status == :converged
            X = [r.minimizer[1] r.minimizer[2]; r.minimizer[2] r.minimizer[3]]
            @test all(eigvals(Symmetric(X)) .> 0)
        end
    end

    # logdet (default) still converges.
    m = make_model()
    r = optimize(m, IPMAlg(), [2.5, 0.5, 2.5])
    @test r.status == :converged
    X = [r.minimizer[1] r.minimizer[2]; r.minimizer[2] r.minimizer[3]]
    @test all(eigvals(Symmetric(X)) .> 0)
end
