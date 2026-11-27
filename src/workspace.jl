# Workspace and the interior-point main loop (primal barrier form).

"""
    IPMAlg

A barrier interior-point method for smooth, generally nonconvex nonlinear
programs: box bounds, inequalities and set constraints (including `PSDCone`)
enter a log-barrier objective (slacks derived exactly, keeping the iterates
on the central path), while equality constraints are enforced through the
KKT system with an inertia-correction loop. Evaluation strategies are
selectable per constraint via NonconvexCore labels
([`IPMEvaluationSelection`](@ref)). Options are set with
[`IPMOptions`](@ref).
"""
struct IPMAlg <: AbstractOptimizer end

function NonconvexCore.solver_capabilities(::IPMAlg)
    return ConstraintCapabilities(;
        linear_eq=true,
        nonlinear_eq=true,
        linear_ineq=true,
        nonlinear_ineq=true,
        linear_sd=true,
        nonlinear_sd=true,
        sets_eq=true,
        sets_ineq=true,
        sets_sd=true,
        sets_simplex=true,
        sets_product=true,
        sets_chance=false,
        variable_sets_sd=true,
        variable_sets_simplex=true,
        box=true,
        continuous=true,
        binary=false,
        integer=false,
        infinite_bounds=true,
    )
end

"""
    IPMWorkspace

Workspace of [`IPMAlg`](@ref): holds the original `VecModel`, the flattened
[`IPMModel`](@ref), the initial point, and the [`IPMOptions`](@ref).
"""
mutable struct IPMWorkspace{M,IP} <: Workspace
    model::M
    ipm::IP
    x0::Vector{Float64}
    options::IPMOptions
end

function Workspace(
    model::VecModel,
    optimizer::IPMAlg,
    x0::AbstractVector;
    options::IPMOptions=IPMOptions(),
    kwargs...,
)
    return IPMWorkspace(model, build_ipm_model(model, options), Float64.(x0), options)
end
function Workspace(model::VecModel, optimizer::IPMAlg; kwargs...)
    return Workspace(model, optimizer, getinit(model); kwargs...)
end

function optimize!(workspace::IPMWorkspace)
    m = workspace.ipm
    opts = workspace.options
    if opts.slacks || any_ineq_slacked(m)
        return _ipm_solve_slacked!(m, opts, workspace.x0)
    end
    if opts.inner === nothing
        return _ipm_solve!(m, opts; x0=workspace.x0)
    end
    return _ipm_solve_inner!(m, opts, workspace.x0)
end

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

# Resolve the linear-solver mode (:direct or :krylov) from the options and
# the presence of matrix-free/trace-estimated constraints.
function _solver_mode(m::IPMModel, opts::IPMOptions)
    opts.linear_solver === :krylov && return :krylov
    opts.linear_solver === :direct && return :direct
    # :auto
    modes = vcat(m.nleq_modes, m.nlineq_modes, Symbol[s[3] for s in m.setcons])
    return any(mode -> mode !== :dense, modes) ? :krylov : :direct
end

function _init_x!(x::Vector{Float64}, m::IPMModel, opts::IPMOptions)
    r = opts.bound_relax_factor
    @inbounds for i in 1:m.n
        if isfinite(m.xl[i])
            d = max(r * (1 + abs(m.xl[i])), r)
            x[i] = max(x[i], m.xl[i] + d)
        end
        if isfinite(m.xu[i])
            d = max(r * (1 + abs(m.xu[i])), r)
            x[i] = min(x[i], m.xu[i] - d)
        end
    end
    return x
end

# Transposed-equality-Jacobian-times-multipliers (for the stationarity
# residual g + J_E' λE).
function _jteq_λ!(r::AbstractVector, m::IPMModel, x::AbstractVector, lamE::AbstractVector)
    if m.jump !== nothing
        return _jump_jteq_λ!(r, m, x, lamE)
    end
    offset = 1
    for (k, c) in enumerate(m.nleqs)
        d = getdim(c)
        jtprod!(r, c, x, view(lamE, offset:(offset + d - 1)))
        offset += d
    end
    if m.linAeq !== nothing
        mul!(r, m.linAeq', view(lamE, offset:(offset + size(m.linAeq, 1) - 1)), 1.0, 1.0)
        offset += size(m.linAeq, 1)
    end
    return nothing
end

# Set-constraint violation contribution to the feasibility measure.
function _set_violation(m::IPMModel, x::AbstractVector)
    viol = 0.0
    for (k, (c, set, mode)) in enumerate(m.setcons)
        if set isa PSDCone
            X = Symmetric(_set_value(c, set, x))
            viol = max(viol, max(0.0, -minimum(eigvals(X))))
        else
            v = barrier_value(set, _set_value(c, set, x))
            isfinite(v) || (viol = max(viol, 1.0))
        end
    end
    return viol
end

# One Newton iteration on the barrier problem at st.mu (the built-in inner
# solver): evaluates the barrier, solves the KKT system, and takes a
# backtracking step. Returns `(:ok, kkt_inf, primal_inf)` or
# `(:fail, kkt_inf, primal_inf)`.
function _newton_barrier_step!(
    m::IPMModel, st::IPMState, opts::IPMOptions, mode::Symbol, xt::Vector{Float64}
)
    # --- evaluate ---
    st.f = _aug_objective(m, st.x)
    m.counter[] += 1 + length(m.relaxed_eqs) + length(m.relaxed_ineqs)
    barrier_gradient!(st.gf, m, st.x, st.mu, opts)
    cE, cI = _slacks(m, st.x)
    copyto!(st.cE, cE)
    copyto!(st.cI, cI)
    mode === :direct && assemble_jacobian!(st, m)

    # switch to strict barriers once the iterate is feasible w.r.t. the
    # inequalities and bounds
    if !m.strict[]
        infeas_viol = isempty(st.cI) ? 0.0 : maximum(v -> max(v, 0.0), st.cI)
        if infeas_viol <= 0.0
            m.strict[] = true
        end
    end

    # --- residuals ---
    primal_inf = max(
        isempty(st.cE) ? 0.0 : maximum(abs, st.cE),
        isempty(st.cI) ? 0.0 : maximum(v -> max(v, 0.0), st.cI),
        _set_violation(m, st.x),
    )
    rstat = copy(st.gf)
    _jteq_λ!(rstat, m, st.x, st.lamE)
    dual_inf = isempty(rstat) ? 0.0 : maximum(abs, rstat)

    # --- KKT solve ---
    fill_rhs!(st, m, st.gf)
    ok = if mode === :direct
        solve_kkt_dense!(st, m, opts)
    else
        (solve_kkt_krylov!(st, m, opts) || solve_kkt_dense!(st, m, opts))
    end
    ok || return (:fail, dual_inf, primal_inf)
    dx = _dx(st)
    dlam = _dlam(st)

    # --- backtracking line search on the exact-penalty merit ---
    α = _bound_step(st.x, dx, m, opts)
    ν = max(1.0, maximum(abs, st.lamE; init=0.0) + 10.0)
    φ0 = penalty_merit(m, st, st.x, opts)
    dφ = dot(st.gf, dx) - ν * sum(abs, st.cE)
    accepted = false
    # A (near) pure-dual step: the primal move is negligible, so the
    # exact-penalty merit is flat along dx and its |cE| kink makes the
    # Armijo test reject any step that nudges cE off zero by more than a
    # rounding unit (stalling the equality-multiplier update). Accept such
    # steps directly when the merit stays finite and essentially
    # non-increasing — this still rejects steps that would cross a
    # constraint boundary (the strict barrier makes the merit infinite).
    if maximum(abs, dx) <= 1e-8 * (1 + maximum(abs, st.x))
        xt .= st.x .+ α .* dx
        φ = penalty_merit(m, st, xt, opts)
        if isfinite(φ) && φ <= φ0 + 1e-3 * max(abs(φ0), 1.0)
            accepted = true
        end
    end
    while !accepted && α > 1e-14
        xt .= st.x .+ α .* dx
        φ = penalty_merit(m, st, xt, opts)
        if isfinite(φ) && φ <= φ0 + 1e-4 * α * dφ
            accepted = true
            break
        end
        α *= 0.5
    end
    accepted || return (:fail, dual_inf, primal_inf)
    st.x .= st.x .+ α .* dx
    st.lamE .+= α .* dlam
    return (:ok, dual_inf, primal_inf)
end

# Residual/convergence/μ bookkeeping shared by the outer loops.
function _outer_update!(m::IPMModel, st::IPMState, opts::IPMOptions, iter::Int, t0::Float64)
    primal_inf = max(
        isempty(st.cE) ? 0.0 : maximum(abs, st.cE),
        isempty(st.cI) ? 0.0 : maximum(v -> max(v, 0.0), st.cI),
        _set_violation(m, st.x),
    )
    rstat = copy(st.gf)
    _jteq_λ!(rstat, m, st.x, st.lamE)
    dual_inf = isempty(rstat) ? 0.0 : maximum(abs, rstat)
    scale = max(1.0, isempty(st.gf) ? 0.0 : maximum(abs, st.gf))
    if st.mu <= 1e-9 && primal_inf <= opts.tol && dual_inf <= max(100 * opts.tol, 1e-4)
        return (:converged, dual_inf, primal_inf)
    end
    if st.mu <= opts.mu_min && primal_inf <= opts.tol && dual_inf <= opts.tol * scale
        return (:converged, dual_inf, primal_inf)
    end
    # μ update (quality-function rule)
    E_mu = max(dual_inf, primal_inf)
    if E_mu <= opts.theta_mu * st.mu || E_mu <= 1e-3 || primal_inf <= 100 * opts.tol
        if E_mu <= opts.theta_mu * st.mu
            st.mu = max(opts.mu_min, st.mu^opts.mu_superlinear_decrease_power)
        else
            st.mu = max(opts.mu_min, opts.mu_linear_decrease_factor * st.mu)
        end
    end
    opts.verbose && println(
        "iter $iter: mu = $(st.mu), primal_inf = $primal_inf, dual = $dual_inf, f = $(st.f)",
    )
    if time() - t0 > opts.max_time
        return (:max_iter, dual_inf, primal_inf)
    end
    return (:continue, dual_inf, primal_inf)
end

# Sub-model for a custom inner optimizer: the (always smoothly extended)
# barrier objective, the enforced equality constraints and the bounds.
function _barrier_submodel(m::IPMModel, opts::IPMOptions, mu_ref::Base.RefValue{Float64})
    sub = Model(x -> barrier_objective_ext(m, x, mu_ref[], opts))
    addvar!(sub, m.xl, m.xu)
    for c in m.nleqs
        add_eq_constraint!(sub, c)
    end
    if m.linAeq !== nothing
        add_linear_constraint!(sub, m.linAeq, EqualTo(m.linbeq))
    end
    return sub
end

# Outer μ loop with a custom inner optimizer for the barrier subproblems.
# Inner-optimizer statuses that mean "no usable point was produced"; any
# other status (including iteration limits) is treated as a usable point
# whose quality is then verified by the primal-feasibility checks.
const _INNER_FAILED_STATUSES = (
    :diverged,
    :infeasible,
    :Diverging_Iterates,
    :Infeasible_Problem_Detected,
    :Invalid_Number_Detected,
    :Search_Direction_Becomes_Too_Small,
    :Not_Enough_Degrees_Of_Freedom,
    :Infeasible_Interior_Point,
    :Unrecoverable_Exception,
)

function _ipm_solve_inner!(m::IPMModel, opts::IPMOptions, x0::Vector{Float64})
    t0 = time()
    x = _init_x!(copy(x0), m, opts)
    for (k, (c, set, em)) in enumerate(m.setcons)
        barrier_strictly_feasible(c, set, x) || throw(
            ArgumentError(
                "initial point is not strictly feasible for set constraint $k " *
                "($(typeof(set))); IPMAlg requires a strictly feasible start for PSD constraints",
            ),
        )
    end
    # feasibility stage for infeasible starts
    if !_strictly_feasible_zone(m, x)
        opts.verbose && println("feasibility stage: restoring feasibility")
        _feasibility_stage!(m, x, opts)
    end
    mu = opts.mu_init
    st = IPMState(m, x, mu)
    mu_ref = Ref(st.mu)
    sub = _barrier_submodel(m, opts, mu_ref)
    for iter in 1:opts.max_iter
        # solve the barrier problem at st.mu with the inner optimizer
        mu_ref[] = st.mu
        r = if opts.inner_options === nothing
            NonconvexCore.optimize(sub, opts.inner, st.x)
        else
            NonconvexCore.optimize(sub, opts.inner, st.x; options=opts.inner_options)
        end
        copyto!(st.x, r.minimizer)
        st.lamE .= 0.0
        # residuals
        st.f = _aug_objective(m, st.x)
        m.counter[] += 1 + length(m.relaxed_eqs) + length(m.relaxed_ineqs)
        barrier_gradient!(st.gf, m, st.x, st.mu, opts)
        cE, cI = _slacks(m, st.x)
        copyto!(st.cE, cE)
        copyto!(st.cI, cI)
        # Practical convergence of the inner-optimizer path: the barrier
        # subproblem solved at mu <= mu_min is O(mu_min)-close to the
        # original solution, and its dual residual (lambda = mu/s) cannot
        # be verified numerically in the endgame (the extended barrier
        # used by the inner optimizer has a kink at the epsilon transition,
        # and resolving s ~ mu near the active boundary is below the inner
        # solver's working precision). Trust the inner optimizer's own
        # success when the returned point is primal-feasible.
        if st.mu <= opts.mu_min &&
            st.mu <= 1e-9 &&
            !(r.status in _INNER_FAILED_STATUSES) &&
            isfinite(r.minimum)
            primal_inf = max(
                isempty(st.cE) ? 0.0 : maximum(abs, st.cE),
                isempty(st.cI) ? 0.0 : maximum(v -> max(v, 0.0), st.cI),
                _set_violation(m, st.x),
            )
            primal_inf <= opts.tol && return _final_result(m, st, :converged, iter)
        end
        status, _, _ = _outer_update!(m, st, opts, iter, t0)
        status === :continue || return _final_result(m, st, status, iter)
    end
    return _final_result(m, st, :max_iter, opts.max_iter)
end

# Strict-feasibility test for the scalar parts of the model (the same
# threshold the barrier uses to switch from its smooth extension to the
# strict barriers): every inequality below -ε and every finite bound at a
# distance above ε.
function _strictly_feasible_zone(m::IPMModel, x::AbstractVector)
    ε = 1e-6
    for c in m.nlineqs
        for ci in _vec(c(x))
            ci <= -ε || return false
        end
    end
    if m.linAineq !== nothing
        for ci in m.linAineq * x .- m.linbineq
            ci <= -ε || return false
        end
    end
    @inbounds for i in 1:m.n
        isfinite(m.xl[i]) && (x[i] - m.xl[i] <= ε) && return false
        isfinite(m.xu[i]) && (m.xu[i] - x[i] <= ε) && return false
    end
    return true
end

# C¹ measure of the constraint violation at x: ½Σ max(c_I + ε, 0)² + ½Σ c_E²
# + ½Σ bound-violation² (c_I in violation form, feasible ⇔ c_I ≤ 0). The
# margin ε = 1e-2 keeps the landing point well inside the strict-barrier
# zone: the barrier's initial multiplier μ/s (≈ μ/ε) stays moderate, so the
# first barrier iterations are not pinned to the boundary (a landing
# exactly on the boundary degenerates the barrier Hessian and stalls the
# solver).
function _violation_objective(m::IPMModel, x::AbstractVector)
    ε = 1e-2
    v = 0.0
    for c in m.nlineqs
        for ci in _vec(c(x))
            ci + ε > 0 && (v += (ci + ε)^2 / 2)
        end
    end
    if m.linAineq !== nothing
        for ci in m.linAineq * x .- m.linbineq
            ci + ε > 0 && (v += (ci + ε)^2 / 2)
        end
    end
    for c in m.nleqs
        v += sum(abs2, _vec(c(x))) / 2
    end
    if m.linAeq !== nothing
        v += sum(abs2, m.linAeq * x .- m.linbeq) / 2
    end
    @inbounds for i in 1:m.n
        if isfinite(m.xl[i]) && x[i] < m.xl[i] + ε
            v += (m.xl[i] + ε - x[i])^2 / 2
        end
        if isfinite(m.xu[i]) && x[i] > m.xu[i] - ε
            v += (x[i] - (m.xu[i] - ε))^2 / 2
        end
    end
    return v
end

# Regularized Hessian operator of the violation objective (for MINRES):
# (H + δI)u with H·u from AD Hessian-vector products.
mutable struct ViolationHessianOp
    f::Any
    x::Vector{Float64}
    δ::Float64
    n::Int
end
Base.size(op::ViolationHessianOp) = (op.n, op.n)
Base.size(op::ViolationHessianOp, d::Int) = size(op)[d]
Base.eltype(::ViolationHessianOp) = Float64
function LinearAlgebra.mul!(y, op::ViolationHessianOp, u)
    fill!(y, 0.0)
    if op.f isa IPMModel
        # JuMP-converted model: the violation Hessian product comes from the
        # evaluator (the closure over the evaluator cannot be AD-traced)
        _jump_violation_hessprod!(y, op.f, op.x, u)
    else
        hvp!(y, op.f, op.x, u)
    end
    y .+= op.δ .* u
    return y
end

# Feasibility stage: run before the barrier iterations when the initial
# point violates the inequalities, bounds or equalities. Minimizes the
# constraint violation with a regularized Newton method (MINRES on H + δI,
# Armijo backtracking on the violation objective) until the violation is
# within the solver tolerance, or gives up after `feasibility_max_iter`
# iterations (the C¹-extended barrier then continues from wherever the
# stage left off).
function _feasibility_stage!(m::IPMModel, x::Vector{Float64}, opts::IPMOptions)
    opts.feasibility_max_iter > 0 || return nothing
    _strictly_feasible_zone(m, x) && return nothing
    f! = z -> _violation_objective(m, z)
    xt = similar(x)
    g = similar(x)
    dx = similar(x)
    restarts = 0
    for iter in 1:opts.feasibility_max_iter
        m.counter[] += 1 + length(m.nlineqs) + length(m.nleqs)
        if m.jump !== nothing
            _jump_violation_gradient!(g, m, x)
        else
            _, gr = value_gradient(f!, x)
            gr === nothing && return nothing
            copyto!(g, gr)
        end
        # regularized Newton direction: (H + δI) dx = -g
        δ = 1e-10
        solved = false
        if m.jump !== nothing
            # the violation Hessian product of a JuMP-converted model comes
            # from the evaluator (no AD through the closure)
            jop = ViolationHessianOp(m, x, δ, m.n)
            sol, stats = Krylov.minres(
                jop, .-g; rtol=1e-8, atol=1e-12, itmax=max(2 * m.n, 100), verbose=0
            )
            if stats.solved
                copyto!(dx, sol)
                solved = true
            end
        end
        for _ in 1:(solved ? 0 : 6)
            op = ViolationHessianOp(f!, x, δ, m.n)
            sol, stats = Krylov.minres(
                op, .-g; rtol=1e-8, atol=1e-12, itmax=max(2 * m.n, 100), verbose=0
            )
            if stats.solved
                copyto!(dx, sol)
                solved = true
                break
            end
            δ *= 100.0
        end
        solved || return nothing
        # Armijo backtracking on the violation objective
        φ0 = f!(x)
        α = 1.0
        dφ = dot(g, dx)
        accepted = false
        while α > 1e-14
            xt .= x .+ α .* dx
            φ = f!(xt)
            m.counter[] += 1 + length(m.nlineqs) + length(m.nleqs)
            if φ <= φ0 + 1e-4 * α * dφ
                accepted = true
                break
            end
            α *= 0.5
        end
        if !accepted
            # Stationary point of the violation measure (e.g. a symmetric
            # saddle of a flat start) with residual violation: break the
            # symmetry with a deterministic perturbation and keep going.
            restarts += 1
            restarts > 5 && return nothing
            @inbounds for i in 1:m.n
                x[i] += 1e-3 * (1 + abs(x[i])) * sin(1.7 * i + 2.3 * restarts)
            end
            continue
        end
        x .+= α .* dx
        f!(x) <= 1e-12 && return nothing # all margins (1e-2) met
    end
    return nothing
end

function _ipm_solve!(m::IPMModel, opts::IPMOptions; x0::Vector{Float64}=m.x0)
    n = m.n
    mode = _solver_mode(m, opts)
    if mode === :direct
        for em in vcat(m.nleq_modes, m.nlineq_modes, Symbol[s[3] for s in m.setcons])
            em === :dense || throw(
                ArgumentError(
                    "linear_solver = :direct requires all constraints to be " *
                    "dense; found mode :$(em). Use linear_solver = :krylov or :auto.",
                ),
            )
        end
    end
    t0 = time()

    # Initial point
    x = _init_x!(copy(x0), m, opts)
    for (k, (c, set, em)) in enumerate(m.setcons)
        barrier_strictly_feasible(c, set, x) || throw(
            ArgumentError(
                "initial point is not strictly feasible for set constraint $k " *
                "($(typeof(set))); IPMAlg requires a strictly feasible start for PSD constraints",
            ),
        )
    end
    # feasibility stage for infeasible starts (inequalities/bounds/equalities)
    if !_strictly_feasible_zone(m, x)
        opts.verbose && println("feasibility stage: restoring feasibility")
        _feasibility_stage!(m, x, opts)
    end
    st = IPMState(m, x, opts.mu_init)
    xt = similar(x)

    for iter in 1:opts.max_iter
        # inner Newton: solve the barrier problem at st.mu
        kkt_inf = Inf
        primal_inf = Inf
        status = :fail
        for _ in 1:opts.inner_max_iter
            status, kkt_inf, primal_inf = _newton_barrier_step!(m, st, opts, mode, xt)
            status === :ok || break
            kkt_inf <= opts.theta_mu * st.mu && break # barrier problem solved
        end
        status === :ok || return _final_result(m, st, :diverged, iter)
        # practical convergence when the inner Newton stalls
        if primal_inf <= opts.tol && kkt_inf <= max(100 * opts.tol, 1e-4) && st.mu <= 1e-9
            return _final_result(m, st, :converged, iter)
        end
        ostatus, _, _ = _outer_update!(m, st, opts, iter, t0)
        ostatus === :continue || return _final_result(m, st, ostatus, iter)
    end
    return _final_result(m, st, :max_iter, opts.max_iter)
end

function _final_result(m::IPMModel, st::IPMState, status::Symbol, iterations::Int)
    fmin = m.obj === nothing ? 0.0 : Float64(m.obj(st.x))
    return IPMResult(copy(st.x), fmin, status, m.counter[], iterations)
end

# Step bound from the variable bounds (fraction-to-the-boundary). The bounds
# are skipped when excluded from the barrier selection.
function _bound_step(x::AbstractVector, dx::AbstractVector, m::IPMModel, opts::IPMOptions)
    αmax = 1.0
    if opts.barrier.box
        @inbounds for i in 1:m.n
            if dx[i] < -1e-16 && isfinite(m.xl[i])
                αmax = min(αmax, (m.xl[i] - x[i]) / dx[i])
            elseif dx[i] > 1e-16 && isfinite(m.xu[i])
                αmax = min(αmax, (m.xu[i] - x[i]) / dx[i])
            end
        end
    end
    α = opts.tau * αmax
    return α < 0.0 ? 0.0 : min(α, 1.0)
end

# Fill the KKT right-hand side (shared by the direct and Krylov paths).
function fill_rhs!(st::IPMState, m::IPMModel, g::AbstractVector)
    n, mE = st.n, st.mE
    r = view(st.rhs, 1:n)
    copyto!(r, g)
    _jteq_λ!(r, m, st.x, st.lamE)
    r .*= -1.0
    st.rhs[(n + 1):(n + mE)] .= .-st.cE
    return nothing
end
