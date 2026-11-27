@testitem "Internal fallbacks and conversions" begin
    using NonconvexInteriorPoint, NonconvexCore, NonconvexSemidefinite, LinearAlgebra
    const N = NonconvexInteriorPoint

    # a set with no barrier oracle / conversion support
    struct UnsupportedSet <: NonconvexCore.AbstractSet end

    opts = IPMOptions()

    # --- barrier.jl: the generic (non-PSDCone) barrier protocol ----------
    # the model converts every supported set to classical constraints, so
    # the generic fallbacks only ever see unsupported sets — they throw,
    # except Reals (empty barrier)
    c = x -> [x[1] + x[2]]
    x = [1.0, 1.0]
    set = GreaterThan(0.5)
    @test_throws ArgumentError N.barrier_value_c(c, set, x, Val(:dense), opts)
    @test_throws ArgumentError N.barrier_grad_x!(zeros(2), c, set, x, Val(:dense), opts)
    @test_throws ArgumentError N.barrier_hess_x!(zeros(2, 2), c, set, x, Val(:dense), opts)
    # the Hessian-product fallback silently degrades to zeros (the hvp
    # fallback absorbs the failed differentiation)
    y0 = zeros(2)
    N.barrier_hess_x_prod!(y0, c, set, x, [0.3, -0.2], Val(:dense), opts)
    @test y0 == zeros(2)
    @test_throws ArgumentError N.barrier_strictly_feasible(c, set, x)

    # Reals has an empty barrier
    @test N.barrier_value(Reals(1), [1.0]) == 0.0
    @test_throws ArgumentError N.barrier_value(UnsupportedSet(), [1.0])
    @test_throws ArgumentError N.barrier_value_c(c, UnsupportedSet(), x, Val(:dense), opts)

    # _set_value with a LinearFunction-backed PSD constraint
    lf = NonconvexCore.LinearFunction([1.0 0.0; 0.0 1.0; 0.0 1.0; 1.0 0.0], zeros(4))
    Xv = N._set_value(lf, PSDCone(2), [2.0, 1.0])
    @test Xv == [2.0 1.0; 1.0 2.0]

    # --- barrier.jl: the core dense PSD fallbacks (bypassing the ext) ----
    cpsd = z -> [z[1] z[2]; z[2] z[3]]
    xpsd = [4.0, 1.0, 3.0]
    Xpsd = [4.0 1.0; 1.0 3.0]
    f_core =
        (c, set, x, m, o) ->
            invoke(N._psd_barrier, Tuple{Any,PSDCone,Any,Val,IPMOptions}, c, set, x, m, o)
    @test f_core(cpsd, PSDCone(2), xpsd, Val(:dense), opts) ≈ -logdet(Xpsd)
    gcore = zeros(3)
    invoke(
        N._psd_barrier_grad!,
        Tuple{Vector{Float64},Any,PSDCone,Any,Val,IPMOptions},
        gcore,
        cpsd,
        PSDCone(2),
        xpsd,
        Val(:dense),
        opts,
    )
    Xinv = inv(Xpsd)
    @test gcore ≈ [-Xinv[1, 1], -2Xinv[1, 2], -Xinv[2, 2]] rtol = 1e-10
    Hcore = zeros(3, 3)
    invoke(
        N._psd_barrier_hess!,
        Tuple{Matrix{Float64},Any,PSDCone,Any,Val,IPMOptions},
        Hcore,
        cpsd,
        PSDCone(2),
        xpsd,
        Val(:dense),
        opts,
    )
    @test Hcore ≈ Hcore' # symmetric
    ycore = zeros(3)
    invoke(
        N._psd_barrier_hessprod!,
        Tuple{Vector{Float64},Any,PSDCone,Any,Vector{Float64},Val,IPMOptions},
        ycore,
        cpsd,
        PSDCone(2),
        xpsd,
        [0.3, -0.7, 0.4],
        Val(:dense),
        opts,
    )
    @test ycore ≈ Hcore * [0.3, -0.7, 0.4] rtol = 1e-10
    # the core non-dense modes throw, and so do the core type-aware
    # fallbacks for non-log-determinant types
    # the core throw branches (the extension specializes these modes, so
    # reach the core methods through invoke)
    @test_throws ArgumentError invoke(
        N._psd_barrier,
        Tuple{Any,PSDCone,Any,Val,IPMOptions},
        cpsd,
        PSDCone(2),
        xpsd,
        Val(:matrixfree),
        opts,
    )
    @test_throws ArgumentError invoke(
        N._sd_barrier,
        Tuple{Any,PSDCone,Any,Val,Symbol,IPMOptions},
        cpsd,
        PSDCone(2),
        xpsd,
        Val(:dense),
        :inverse,
        opts,
    )
    @test_throws ArgumentError N._sd_barrier(
        cpsd, PSDCone(2), xpsd, Val(:bogus), :logdet, opts
    )
    @test_throws ArgumentError invoke(
        N._sd_barrier_grad!,
        Tuple{Vector{Float64},Any,PSDCone,Any,Val,Symbol,IPMOptions},
        zeros(3),
        cpsd,
        PSDCone(2),
        xpsd,
        Val(:dense),
        :inverse,
        opts,
    )
    @test_throws ArgumentError invoke(
        N._sd_barrier_hess!,
        Tuple{Matrix{Float64},Any,PSDCone,Any,Val,Symbol,IPMOptions},
        zeros(3, 3),
        cpsd,
        PSDCone(2),
        xpsd,
        Val(:dense),
        :inverse,
        opts,
    )
    @test_throws ArgumentError invoke(
        N._sd_barrier_hessprod!,
        Tuple{Vector{Float64},Any,PSDCone,Any,Vector{Float64},Val,Symbol,IPMOptions},
        zeros(3),
        cpsd,
        PSDCone(2),
        xpsd,
        [0.1, 0.2, 0.3],
        Val(:dense),
        :inverse,
        opts,
    )

    # the core type-aware passthroughs (mode = :bogus hits the core) and
    # the PSDCone dispatch helpers
    @test_throws ArgumentError N._sd_barrier_grad!(
        zeros(3), cpsd, PSDCone(2), xpsd, Val(:bogus), :logdet, opts
    )
    @test_throws ArgumentError N._sd_barrier_hess!(
        zeros(3, 3), cpsd, PSDCone(2), xpsd, Val(:bogus), :logdet, opts
    )
    @test_throws ArgumentError N._sd_barrier_hessprod!(
        zeros(3), cpsd, PSDCone(2), xpsd, [0.1, 0.2, 0.3], Val(:bogus), :logdet, opts
    )
    @test_throws ArgumentError N.barrier_grad_x!(
        zeros(3), cpsd, PSDCone(2), xpsd, Val(:bogus), opts
    )
    @test_throws ArgumentError N.barrier_hess_x!(
        zeros(3, 3), cpsd, PSDCone(2), xpsd, Val(:bogus), opts
    )
    @test_throws ArgumentError N.barrier_hess_x_prod!(
        zeros(3), cpsd, PSDCone(2), xpsd, [0.1, 0.2, 0.3], Val(:bogus), opts
    )
    @test_throws ArgumentError N.sd_barrier_hess_x!(
        zeros(3, 3), cpsd, PSDCone(2), xpsd, Val(:bogus), :logdet, opts
    )
    @test_throws ArgumentError N.barrier_value_c(cpsd, PSDCone(2), xpsd, Val(:bogus), opts)

    # --- ad.jl: the Hessian-product fallbacks ----------------------------
    # a mutating function breaks both the prepared hvp and the dense
    # Hessian: the fallbacks return zeros
    v = [1.0, 2.0, 3.0]
    @test N.hvp(x -> (y=copy(x); y[1]=0.0; sum(y)), [1.0, 2.0], v[1:2]) ≈ zeros(2)
    # direct Jacobian/Jacobian-transpose products
    g2 = x -> [x[1]^2, x[1] * x[2]]
    @test N.jacprod(g2, [2.0, 3.0], [0.5, -0.5]) ≈ [0.5, -1.0]
    @test N.jtprod(g2, [2.0, 3.0], [0.2, 0.3]) ≈ [1.7, 0.6]

    # --- model.jl: the linear and functional set conversions -------------
    function build(opts=IPMOptions(), flags=Set{Symbol}(); fset)
        m = Model(x -> sum(abs2, x))
        addvar!(m, [-2.0, -2.0], [2.0, 2.0])
        add_set_constraint!(m, fset[1], fset[2]; flags=flags)
        vm, _, _ = NonconvexCore.tovecmodel(m, [0.5, 0.5])
        return N.build_ipm_model(vm, opts)
    end
    A = [1.0 0.0; 0.0 1.0]
    # linear set conversions (LinearFunction + sets)
    for set in (
        GreaterThan([0.0, 0.0]),
        LessThan([1.0, 1.0]),
        EqualTo([0.5, 0.5]),
        Interval([0.0, 0.0], [1.0, 1.0]),
        Zeros(2),
        Nonnegatives(2),
        Nonpositives(2),
        SetProduct((GreaterThan(0.0), LessThan(1.0))),
    )
        m = build(IPMOptions(); fset=(NonconvexCore.LinearFunction(A, zeros(2)), set))
        @test m isa N.IPMModel
    end
    # Simplex via the linear path
    m = build(IPMOptions(); fset=(NonconvexCore.LinearFunction(A, zeros(2)), Simplex(2)))
    @test m isa N.IPMModel
    # functional set conversions (plain functions + sets)
    f2 = x -> [x[1] + x[2], x[1] - x[2]]
    for set in (
        GreaterThan([0.0, 0.0]),
        LessThan([1.0, 1.0]),
        EqualTo([0.5, 0.5]),
        Interval([0.0, 0.0], [1.0, 1.0]),
        Zeros(2),
        Nonnegatives(2),
        Nonpositives(2),
        SetProduct((GreaterThan(0.0), LessThan(1.0))),
        Simplex(2),
    )
        m = build(IPMOptions(); fset=(f2, set))
        @test m isa N.IPMModel
    end
    # Reals drops the constraint
    m = build(IPMOptions(); fset=(f2, Reals(2)))
    @test isempty(m.nleqs) && isempty(m.nlineqs) && isempty(m.setcons)
    # unsupported sets throw
    @test_throws ArgumentError build(IPMOptions(); fset=(f2, UnsupportedSet()))

    # relaxed inequalities (label-based relaxation at build)
    m = Model(x -> sum(abs2, x))
    addvar!(m, [-2.0, -2.0], [2.0, 2.0])
    add_ineq_constraint!(m, x -> 1.0 - x[1] - x[2]; flags=Set([:relax]))
    vm, _, _ = NonconvexCore.tovecmodel(m, [0.5, 0.5])
    ipm = N.build_ipm_model(vm, IPMOptions(relaxed_labels=[:relax]))
    @test length(ipm.relaxed_ineqs) == 1
    @test isempty(ipm.nlineqs)

    # _aug_objective: domain errors become Inf, relaxed constraints add
    # quadratic penalties
    m = Model(x -> x[1] > 0.0 ? 0.0 : log(x[1])) # DomainError for x[1] <= 0
    addvar!(m, [-2.0, -2.0], [2.0, 2.0])
    vm, _, _ = NonconvexCore.tovecmodel(m, [1.0, 1.0])
    ipm = N.build_ipm_model(vm, IPMOptions())
    @test N._aug_objective(ipm, [-1.0, 0.0]) == Inf
    @test N._aug_objective(ipm, [1.0, 0.0]) == 0.0
    # per-row barrier masks helper
    rows = N.ineq_barrier_rows(ipm)
    @test rows isa Vector{Bool}
    # Lagrangian closure (objective + multipliers on the constraints)
    m = Model(x -> sum(abs2, x))
    addvar!(m, [-2.0, -2.0], [2.0, 2.0])
    add_eq_constraint!(m, x -> x[1] - x[2])
    add_ineq_constraint!(m, x -> 1.0 - x[1])
    vm, _, _ = NonconvexCore.tovecmodel(m, [0.5, 0.5])
    ipm = N.build_ipm_model(vm, IPMOptions())
    L = N._lagrangian_closure(ipm, [0.3], [0.2])
    @test L([1.0, 1.0]) ≈ 2.0 + 0.3 * 0.0 + 0.2 * 0.0

    # --- slacked.jl: the internal slacked machinery with mixed masks -----
    m = Model(x -> sum(abs2, x))
    addvar!(m, [-2.0, -2.0], [2.0, 2.0])
    add_ineq_constraint!(m, x -> 1.0 - x[1] - x[2]) # barriered
    add_ineq_constraint!(m, x -> x[1] - 1.5; flags=Set([:hard])) # slacked
    vm, _, _ = NonconvexCore.tovecmodel(m, [0.5, 0.5])
    ipm = N.build_ipm_model(
        vm, IPMOptions(barrier=N.BarrierSelection(labels=[:hard]), linear_constraints=false)
    )
    @test any(ipm.nlineq_barrier) && any(!, ipm.nlineq_barrier)
    s = [0.8]
    x = [0.7, 0.7]
    mu = 0.1
    # slacked barrier value (feasible regime)
    ipm.strict[] = true
    @test isfinite(N.slacked_barrier_value(ipm, x, s, mu, IPMOptions()))
    # infeasible regime (extended barriers)
    ipm.strict[] = false
    v_ext = N.slacked_barrier_value(ipm, [-1.0, -1.0], s, mu, IPMOptions())
    @test isfinite(v_ext)
    # the slacked barrier gradient (infeasible-start fallback included)
    g = zeros(2)
    N.slacked_barrier_gradient!(g, ipm, x, s, mu, IPMOptions())
    @test all(isfinite, g)
    # constraint evaluation with the mixed masks
    cE = zeros(0)
    cI = zeros(1)
    N.eval_slacked_cons!(cE, cI, ipm, x)
    @test cI[1] ≈ 0.7 - 1.5
    # the slacked KKT operator
    st = N.SlackedState(ipm, x, s, mu)
    op = N.SlackedKKTOp(ipm, st, IPMOptions(), 1e-10, 1e-8)
    u = randn(4)
    y = zeros(4)
    mul!(y, op, u)
    @test all(isfinite, y)
    # the slacked Lagrangian
    Ls = N._slacked_lagrangian(ipm, [0.1], [0.2])
    @test Ls([1.0, 1.0]) ≈ 2.0 + 0.2 * (-0.5)

    # --- kkt.jl: scalar barrier validation -------------------------------
    @test_throws ArgumentError N._scalar_barrier(1.0, 0.1, :bogus)
    @test_throws ArgumentError N._scalar_barrier_deriv(1.0, 0.1, :bogus)
    @test N._scalar_barrier(1.0, 0.1, :log) ≈ -0.1 * log(1.0)
    @test N._scalar_barrier(1.0, 2.0, :inv) ≈ 2.0 / 1.0
    @test N._scalar_barrier(2.0, 1.0, :quad) ≈ 1.0 / (2 * 4.0)

    # --- kkt.jl: the extended barrier for inner optimizers ---------------
    m = Model(x -> sum(abs2, x))
    addvar!(m, [-2.0, -2.0], [2.0, 2.0])
    add_ineq_constraint!(m, x -> 1.0 - x[1] - x[2])
    vm, _, _ = NonconvexCore.tovecmodel(m, [0.5, 0.5])
    ipm = N.build_ipm_model(vm, IPMOptions())
    # feasible point: log branch; near-boundary: linear branch; violated:
    # hard quadratic penalty
    @test isfinite(N.barrier_objective_ext(ipm, [0.4, 0.4], 0.1, IPMOptions()))
    @test isfinite(N.barrier_objective_ext(ipm, [0.5, 0.5], 0.1, IPMOptions()))
    @test isfinite(N.barrier_objective_ext(ipm, [2.0, 2.0], 0.1, IPMOptions()))
    # the penalty closure (wraps penalty_merit)
    st = N.IPMState(ipm, [0.4, 0.4], 0.1)
    @test N._penalty_closure(ipm, st, IPMOptions())([0.4, 0.4]) isa Real

    # --- kkt.jl: the Krylov KKT operator with native linear equalities ---
    m = Model(x -> sum(abs2, x))
    addvar!(m, [-2.0, -2.0], [2.0, 2.0])
    add_linear_constraint!(m, [1.0 1.0], EqualTo([1.0]))
    add_ineq_constraint!(m, x -> 1.0 - x[1] - x[2]; flags=Set([:mf]))
    vm, _, _ = NonconvexCore.tovecmodel(m, [0.4, 0.4])
    ipm = N.build_ipm_model(
        vm, IPMOptions(evaluation=N.IPMEvaluationSelection(matrixfree_all=true))
    )
    st = N.IPMState(ipm, [0.4, 0.4], 0.1)
    st.lamE .= [0.3]
    op = N.KKTLinearOp(ipm, st, IPMOptions(), 1e-10, 1e-8)
    u = randn(3)
    y = zeros(3)
    mul!(y, op, u)
    @test all(isfinite, y)
    # diagonal preconditioner
    P = N.KKTDiagPrecond([1.0, 2.0, 3.0])
    pv = [3.0, 6.0, 9.0]
    ldiv!(P, pv)
    @test pv == [3.0, 3.0, 3.0]
    @test size(P) == (3, 3)

    # --- slacked.jl: mask helpers and the extended regime ----------------
    m = Model(x -> sum(abs2, x))
    addvar!(m, [-2.0, -2.0], [2.0, 2.0])
    add_ineq_constraint!(m, x -> 1.0 - x[1] - x[2])
    add_ineq_constraint!(m, x -> x[1] - 1.5; flags=Set([:hard]))
    vm, _, _ = NonconvexCore.tovecmodel(m, [0.5, 0.5])
    ipm = N.build_ipm_model(
        vm, IPMOptions(barrier=N.BarrierSelection(labels=[:hard]), linear_constraints=false)
    )
    @test N._nl_slacked(ipm) == Bool[false, true]
    @test N._lin_slacked(ipm) == Bool[]
    # violated inequality in the extended regime (strict = false)
    ipm.strict[] = false
    v_ext = N.slacked_barrier_value(ipm, [2.0, 2.0], [0.8], 0.1, IPMOptions())
    @test isfinite(v_ext)
    g = zeros(2)
    N.slacked_barrier_gradient!(g, ipm, [2.0, 2.0], [0.8], 0.1, IPMOptions())
    @test all(isfinite, g)
    # violated bounds in the extended regime
    ipm2 = N.build_ipm_model(vm, IPMOptions(slacks=true, linear_constraints=false))
    ipm2.strict[] = false
    @test isfinite(
        N.slacked_barrier_value(ipm2, [-3.0, -3.0], [1.0, 1.0], 0.1, IPMOptions())
    )
    # slacked value with a PSD set constraint
    m = Model(x -> sum(abs2, x))
    addvar!(m, [0.1, -Inf, 0.1], [Inf, Inf, Inf])
    add_set_constraint!(m, x -> [x[1] x[2]; x[2] x[3]], PSDCone(2))
    vm, _, _ = NonconvexCore.tovecmodel(m, [2.0, 0.5, 2.0])
    ipm3 = N.build_ipm_model(vm, IPMOptions(slacks=true))
    @test isfinite(
        N.slacked_barrier_value(ipm3, [2.0, 0.5, 2.0], Float64[], 0.1, IPMOptions())
    )
end
