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
        m.fixed_bounds[i] && continue # pinned by an equality row, not a barrier
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

# Equality-Jacobian-times-direction J_E dx (for the directional merit
# derivative; the JuMP-converted models read it from the evaluator).
function _jeq_prod(m::IPMModel, st, dx::AbstractVector, mode::Symbol)
    isempty(st.cE) && return zeros(0)
    Jdx = zeros(st.mE)
    if m.jump !== nothing
        mul!(Jdx, _jump_eq_jacobian(m, st.x), dx)
        return Jdx
    end
    offset = 1
    for (k, c) in enumerate(m.nleqs)
        d = getdim(c)
        Jdx[offset:(offset + d - 1)] .= jacprod(c, st.x, dx)
        offset += d
    end
    if m.linAeq !== nothing
        mul!(Jdx, m.linAeq, dx)
    end
    return Jdx
end

# Minimum slack of the barriered inequalities and bounds (for the
# central-path safeguard of the pure-dual bypass: a full step must not
# crush a barriered slack against the boundary, where the barrier terms
# μ/s and the Hessian terms μ/s² explode).
function _min_barrier_slack(m::IPMModel, x::AbstractVector, opts::IPMOptions)
    mins = Inf
    for (k, c) in enumerate(m.nlineqs)
        m.nlineq_barrier[k] || continue
        for v in _vec(c(x))
            mins = min(mins, -v)
        end
    end
    if m.linAineq !== nothing && any(m.lin_ineq_barrier)
        v = m.linAineq * x .- m.linbineq
        for i in findall(m.lin_ineq_barrier)
            mins = min(mins, -v[i])
        end
    end
    if opts.barrier.box
        @inbounds for i in 1:m.n
            m.fixed_bounds[i] && continue
            isfinite(m.xl[i]) && (mins = min(mins, x[i] - m.xl[i]))
            isfinite(m.xu[i]) && (mins = min(mins, m.xu[i] - x[i]))
        end
    end
    return mins
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
# Constraint-violation measure for the filter: the 1-norm of the equality
# residuals plus the positive-part sum of the inequalities plus the set
# violations.
function _filter_theta(m::IPMModel, x::AbstractVector)
    cE = zeros(mE(m))
    cI = zeros(mI(m))
    eval_nonlinear_cons!(cE, cI, m, x)
    eval_linear_cons!(cE, cI, m, x)
    return sum(abs, cE) + sum(max.(cI, 0.0)) + _set_violation(m, x)
end

# Wächter–Biegler filter acceptance: a trial pair (θ, f) is acceptable if
# it improves the violation or the objective by the margin γ against every
# filter entry.
function _filter_acceptable(
    F::Vector{Tuple{Float64,Float64}}, θ::Float64, f::Float64, γ::Float64
)
    @inbounds for (θj, fj) in F
        (θ <= (1 - γ) * θj || f <= fj - γ * θj) || return false
    end
    return true
end

# Filter backtracking on the Newton direction: return the accepted step
# length (without moving the state); f-type steps (Armijo on the exact
# penalty) leave the filter unchanged, θ-type steps augment it. On total
# rejection, run the restoration phase (minimize the violation measure),
# move the state to the restored point and return -1.0 as the sentinel for
# "restoration accepted, no Newton step"; return 0.0 on failure.
function _filter_backtrack!(m::IPMModel, st::IPMState, opts::IPMOptions, xt, dx)
    γ = opts.filter_gamma
    α = _bound_step(st.x, dx, m, opts)
    while α > 1e-14
        xt .= st.x .+ α .* dx
        φt = barrier_objective(m, xt, st.mu, opts)
        if isfinite(φt)
            θt = _filter_theta(m, xt)
            if _filter_acceptable(st.filter, θt, φt, γ)
                # f-type if the exact-penalty Armijo also holds; otherwise a
                # θ-type step that augments the filter with the margin pair
                ν = max(1.0, maximum(abs, st.lamE; init=0.0) + 10.0)
                φ0 = penalty_merit(m, st, st.x, opts; ν=ν)
                dφ = dot(st.gf, dx) - ν * sum(abs, st.cE)
                if !(φt + ν * θt <= φ0 + 1e-4 * α * dφ)
                    push!(st.filter, ((1 - γ) * θt, φt - γ * θt))
                end
                return α
            end
        end
        α *= 0.5
    end
    # restoration: minimize the violation measure from the current point
    for _ in 1:opts.feasibility_max_iter
        g = zeros(m.n)
        if m.jump !== nothing
            _jump_violation_gradient!(g, m, st.x)
        else
            _, gr = value_gradient(z -> _filter_theta(m, z), st.x)
            gr === nothing && return 0.0
            copyto!(g, gr)
        end
        solved = false
        δ = 1e-10
        if m.jump !== nothing && m.n <= 512
            Hd = Matrix{Float64}(undef, m.n, m.n)
            ei = zeros(m.n)
            for j in 1:m.n
                ei[j] = 1.0
                _jump_violation_hessprod!(view(Hd, :, j), m, st.x, ei)
                ei[j] = 0.0
            end
            for _ in 1:7
                okd = try
                    F = cholesky!(Symmetric(Hd) + δ * I)
                    ldiv!(xt, F, .-g)
                    true
                catch
                    false
                end
                if okd
                    solved = true
                    break
                end
                δ *= 100.0
            end
        end
        if !solved
            op = ViolationHessianOp(
                m.jump === nothing ? (z -> _filter_theta(m, z)) : m, st.x, δ, m.n
            )
            sol, stats = Krylov.minres(
                op, .-g; rtol=1e-8, atol=1e-12, itmax=max(4 * m.n, 200), verbose=0
            )
            if stats.solved
                copyto!(xt, sol)
                solved = true
            end
        end
        solved || return 0.0
        θ0 = _filter_theta(m, st.x)
        dθ = dot(g, xt)
        αr = 1.0
        acceptedr = false
        while αr > 1e-14
            xr = st.x .+ αr .* xt
            if _filter_theta(m, xr) <= θ0 + 1e-4 * αr * dθ
                st.x .= xr
                acceptedr = true
                break
            end
            αr *= 0.5
        end
        acceptedr || return 0.0
        θr = _filter_theta(m, st.x)
        φr = barrier_objective(m, st.x, st.mu, opts)
        if isfinite(φr) && _filter_acceptable(st.filter, θr, φr, γ)
            push!(st.filter, ((1 - γ) * θr, φr - γ * θr))
            return -1.0
        end
    end
    return 0.0
end

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
    # the Krylov solve has no inertia control: a converged MINRES solution
    # of a wrong-inertia regularized system is not a merit-descent
    # direction — detect it and re-solve with the dense factorization
    if mode !== :direct
        νchk = max(1.0, maximum(abs, st.lamE; init=0.0) + maximum(abs, dlam) + 10.0)
        if dot(st.gf, dx) - νchk * sum(abs, st.cE) >= 0.0
            ok = solve_kkt_dense!(st, m, opts)
            ok || return (:fail, dual_inf, primal_inf)
            dx = _dx(st)
            dlam = _dlam(st)
        end
    end

    # --- backtracking line search on the exact-penalty merit ---
    α = _bound_step(st.x, dx, m, opts)
    # the exact-penalty weight must dominate the *next* multipliers λ + dλ,
    # not just the current ones (otherwise the Newton direction need not be
    # a merit-descent direction)
    ν = max(1.0, maximum(abs, st.lamE; init=0.0) + maximum(abs, dlam) + 10.0)
    φ0 = penalty_merit(m, st, st.x, opts; ν=ν)
    # Directional derivative of the exact-penalty merit as the theoretical
    # upper bound g'd - ν‖cE‖₁: the linearized ‖cE + JEdx‖ variant
    # overestimates the penalty change for nonlinear rows and rejects
    # legitimate Newton corrections. The kink at cE = 0 is handled by the
    # pure-dual bypass below instead.
    JEdx = _jeq_prod(m, st, dx, mode)
    dφ = dot(st.gf, dx) - ν * sum(abs, st.cE)
    accepted = false
    # A (near) pure-dual step, or a step that preserves the equalities to
    # first order (J_E dx ≈ 0): the exact-penalty merit is flat along dx and
    # its |cE| kink makes the Armijo test reject any step that nudges cE off
    # zero by more than a rounding unit (stalling the equality-multiplier
    # update). Accept such steps directly when the merit stays finite and
    # essentially non-increasing — this still rejects steps that would cross
    # a constraint boundary (the strict barrier makes the merit infinite).
    # (the JEdx branch only applies with equality constraints present: an
    # empty equality block would make maximum(abs, []) = 0 and accept every
    # full step through the bypass)
    if maximum(abs, dx) <= 1e-8 * (1 + maximum(abs, st.x)) ||
        (!isempty(st.cE) && maximum(abs, JEdx) <= 1e-8 * (1 + maximum(abs, st.cE)))
        xt .= st.x .+ α .* dx
        φ = penalty_merit(m, st, xt, opts; ν=ν)
        if isfinite(φ) && φ <= φ0 + 1e-3 * max(abs(φ0), 1.0)
            accepted = true
        end
    end
    if !accepted && opts.line_search === :filter
        αf = _filter_backtrack!(m, st, opts, xt, dx)
        if αf == -1.0
            accepted = true
            α = 0.0
        elseif αf > 1e-14
            accepted = true
            α = αf
        end
    end
    while !accepted && α > 1e-14
        xt .= st.x .+ α .* dx
        φ = penalty_merit(m, st, xt, opts; ν=ν)
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
    # μ update (quality-function rule) with a stagnation escape: when the
    # barrier KKT error is not improving and the iterate is (nearly)
    # feasible, force the linear μ decrease instead of waiting — the
    # endgame plateaus otherwise pin μ and stall the multiplier updates.
    E_mu = max(dual_inf, primal_inf)
    stagnant = st.last_E <= 1.1 * E_mu
    st.last_E = E_mu
    if E_mu <= opts.theta_mu * st.mu ||
        E_mu <= 1e-3 ||
        primal_inf <= 100 * opts.tol ||
        (stagnant && primal_inf <= 1e-3)
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
            primal_inf <= opts.tol && return _final_result(m, st, :converged, iter, opts)
        end
        status, _, _ = _outer_update!(m, st, opts, iter, t0)
        status === :continue || return _final_result(m, st, status, iter, opts)
    end
    return _final_result(m, st, :max_iter, opts.max_iter, opts)
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
        m.fixed_bounds[i] && continue
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
        m.fixed_bounds[i] && continue # pinned by the equality rows above
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
        # Dense fast path for small problems: assemble the violation Hessian
        # through the (evaluator-based) Hessian products once and solve the
        # regularized system directly — MINRES needs hundreds of operator
        # applications per Newton step (each an evaluator Hessian product),
        # which dominates the stage cost on small models.
        if m.jump !== nothing && m.n <= 512
            Hd = Matrix{Float64}(undef, m.n, m.n)
            ei = zeros(m.n)
            for j in 1:m.n
                ei[j] = 1.0
                _jump_violation_hessprod!(view(Hd, :, j), m, x, ei)
                ei[j] = 0.0
            end
            for retry in 1:7
                okd = try
                    F = cholesky!(Symmetric(Hd) + δ * I)
                    ldiv!(dx, F, .-g)
                    true
                catch
                    false
                end
                if okd
                    solved = true
                    break
                end
                δ *= 100.0
            end
        end
        if !solved
            # JuMP-converted models keep the evaluator-based product in every
            # retry — AD cannot trace through the MOI evaluator, so the retry
            # loop must not fall back to the closure operator for them.
            for _ in 1:7
                op = if m.jump !== nothing
                    ViolationHessianOp(m, x, δ, m.n)
                else
                    ViolationHessianOp(f!, x, δ, m.n)
                end
                sol, stats = Krylov.minres(
                    op, .-g; rtol=1e-8, atol=1e-12, itmax=max(4 * m.n, 200), verbose=0
                )
                if stats.solved
                    copyto!(dx, sol)
                    solved = true
                    break
                end
                δ *= 100.0
            end
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
        # all margins (1e-2) met: the JuMP-converted models use the dense
        # stage solve and stop at 1e-6 (deep enough for the 1e-6 strict
        # zone, and the empirically stable setting for their barrier
        # trajectories); the generic AD-traced stage keeps the tighter
        # 1e-12 polish of the historical behavior
        f!(x) <= (m.jump !== nothing ? 1e-6 : 1e-12) && return nothing
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
    if opts.line_search === :filter
        # the filter ceiling is set by the initial violation: the first step
        # may not increase θ beyond ~θ(x0), so the barrier path cannot run
        # into gross infeasibility on the first Newton corrections
        θ0 = _filter_theta(m, x)
        st.filter[1] = (max(θ0 * (1 + 100 * opts.filter_gamma) + 1e-8, 1e-8), -Inf)
    end
    xt = similar(x)
    nstall = 0
    fhist = fill(Inf, 5)

    for iter in 1:opts.max_iter
        # inner Newton: solve the barrier problem at st.mu
        kkt_inf = Inf
        primal_inf = Inf
        status = :fail
        stagnant = 0
        prev_f = st.f
        inner_stalled = false
        for _ in 1:opts.inner_max_iter
            status, kkt_inf, primal_inf = _newton_barrier_step!(m, st, opts, mode, xt)
            status === :ok || break
            kkt_inf <= opts.theta_mu * st.mu && break # barrier problem solved
            # stall detection on the objective progress: near the boundary
            # the fraction-to-boundary can cap every step without improving
            # the KKT error, but as long as the objective is still moving
            # the crawl is making progress — break only when the objective
            # has genuinely frozen, and let the outer μ update (and the
            # practical-convergence check) act instead of crawling
            stagnant = abs(st.f - prev_f) <= 1e-10 * max(abs(st.f), 1.0) ? stagnant + 1 : 0
            prev_f = st.f
            if stagnant >= 5
                inner_stalled = true
                break
            end
        end
        status === :ok || return _final_result(m, st, :diverged, iter, opts)
        # practical convergence when the inner Newton stalls
        if primal_inf <= opts.tol && kkt_inf <= max(100 * opts.tol, 1e-4) && st.mu <= 1e-9
            return _final_result(m, st, :converged, iter, opts)
        end
        ostatus, dual_inf, primal_inf = _outer_update!(m, st, opts, iter, t0)
        ostatus === :continue || return _final_result(m, st, ostatus, iter, opts)
        # practical convergence once the barrier has essentially vanished:
        # when the inner Newton has genuinely stalled (frozen objective) at
        # a feasible, stationarity-close point, the primal barrier cannot
        # always polish the residual below the absolute threshold near
        # active constraints (the fraction-to-the-boundary caps the Newton
        # steps) — accept the point instead of crawling. The stall gate
        # keeps this from firing early on well-behaved problems.
        if inner_stalled && st.mu <= 1e-5
            scale = max(1.0, isempty(st.gf) ? 0.0 : maximum(abs, st.gf))
            if primal_inf <= opts.tol && dual_inf <= max(opts.tol * scale, 1e-4)
                return _final_result(m, st, :converged, iter, opts)
            end
        end
        # global stall: at the terminal barrier, the inner Newton stalls AND
        # the objective has stopped improving over the last few outer
        # iterations (a slowly-crawling state with a still-decreasing f is
        # NOT stalled — the μ → 0 relaxation keeps moving it) — then stop
        # and let _final_result apply the practical-convergence upgrade
        if inner_stalled
            nstall += 1
        else
            nstall = 0
        end
        fhist[iter % 5 + 1] = st.f
        fwindow = abs(st.f - fhist[(iter + 1) % 5 + 1]) <= 1e-8 * max(abs(st.f), 1.0)
        if st.mu <= opts.mu_min && nstall >= 3 && fwindow
            return _final_result(m, st, :max_iter, iter, opts)
        end
    end
    return _final_result(m, st, :max_iter, opts.max_iter, opts)
end

# Objective gradient only (no barrier or multiplier terms).
function _objective_gradient(m::IPMModel, x::AbstractVector, g::AbstractVector)
    if m.jump !== nothing
        gobj = similar(g)
        NonconvexCore.jump_objective_gradient!(m.jump, gobj, x)
        copyto!(g, gobj)
    else
        _, go = value_gradient(_aug_obj_closure(m), x)
        go === nothing ? fill!(g, 0.0) : copyto!(g, go)
    end
    return g
end

# Full constraint Jacobian (all equality and inequality rows), dense.
function _full_jacobian(m::IPMModel, x::AbstractVector)
    if m.jump !== nothing
        return Matrix(NonconvexCore.jump_jacobian(m.jump, x))
    end
    rows = Matrix{Float64}[]
    for c in m.nleqs
        _, J = value_jacobian(c, x)
        push!(rows, Matrix(J))
    end
    for c in m.nlineqs
        _, J = value_jacobian(c, x)
        push!(rows, Matrix(J))
    end
    m.linAeq !== nothing && push!(rows, Matrix(m.linAeq))
    m.linAineq !== nothing && push!(rows, Matrix(m.linAineq))
    return reduce(vcat, rows)
end

# Projected stationarity residual: min_λ ‖∇f + P'λ‖∞ with P = [J; E_a],
# the constraint Jacobian plus the near-active bound rows, subject to
# λ ≥ 0 on the inequality and bound rows (the equality multipliers are
# free). The nonnegative-least-squares residual is the correct testable
# quantity: an unconstrained projection is vacuous when rank(J) = n (every
# objective gradient is then representable, signs aside), and the sign
# constraints are exactly what distinguishes a KKT point from an arbitrary
# feasible point.
function _projected_dual(m::IPMModel, x::AbstractVector, r::AbstractVector)
    J = _full_jacobian(m, x)
    rows = Matrix{Float64}[]
    push!(rows, J)
    bndrows = Int[]
    for i in 1:m.n
        m.fixed_bounds[i] && continue
        if (isfinite(m.xl[i]) && x[i] - m.xl[i] <= 1e-8) ||
            (isfinite(m.xu[i]) && m.xu[i] - x[i] <= 1e-8)
            row = zeros(1, m.n)
            row[1, i] = 1.0
            push!(rows, row)
            push!(bndrows, i)
        end
    end
    P = reduce(vcat, rows)
    nineq = mI(m)
    meq = size(J, 1) - nineq
    nb = length(bndrows)
    # lower bounds: ineq ≥ 0, eq free, near-active bounds ≥ 0
    lb = vcat(fill(0.0, nineq), fill(-Inf, meq), fill(0.0, nb))
    G = P * P' + 1e-10 * I
    b = P * r
    λ = G \ b
    active = λ .< lb
    for _ in 1:10
        any(active) || break
        free = .!active
        Gf = G[free, free]
        λf = Gf \ (b[free] - G[free, active] * lb[active])
        λ[free] .= λf
        λ[active] .= lb[active]
        activem = λ .< lb .- 1e-12
        activem == active && break
        active = activem
    end
    rp = r - P' * λ
    return maximum(abs, rp)
end

function _final_result(
    m::IPMModel, st::IPMState, status::Symbol, iterations::Int, opts::IPMOptions
)
    # Practical-convergence upgrade: the endgame can stall the μ schedule or
    # the line search at a point that is already primal-feasible and
    # stationarity-close; report it as converged instead of a failure.
    if status in (:diverged, :max_iter)
        primal_inf = max(
            isempty(st.cE) ? 0.0 : maximum(abs, st.cE),
            isempty(st.cI) ? 0.0 : maximum(v -> max(v, 0.0), st.cI),
            _set_violation(m, st.x),
        )
        rstat = copy(st.gf)
        _jteq_λ!(rstat, m, st.x, st.lamE)
        dual_inf = isempty(rstat) ? 0.0 : maximum(abs, rstat)
        # the projected test uses the true objective gradient: the barrier
        # terms in gf are implied multipliers that the NNLS must not
        # double-count
        gf_true = zeros(m.n)
        _objective_gradient(m, st.x, gf_true)
        dual_proj = _projected_dual(m, st.x, gf_true)
        if primal_inf <= 1000 * opts.tol &&
            min(dual_inf, dual_proj) <= max(100 * opts.tol, 1e-4)
            status = :converged
        end
    end
    fmin = m.obj === nothing ? 0.0 : Float64(m.obj(st.x))
    return IPMResult(copy(st.x), fmin, status, m.counter[], iterations)
end

# Step bound from the variable bounds (fraction-to-the-boundary). The bounds
# are skipped when excluded from the barrier selection.
function _bound_step(x::AbstractVector, dx::AbstractVector, m::IPMModel, opts::IPMOptions)
    αmax = 1.0
    if opts.barrier.box
        @inbounds for i in 1:m.n
            m.fixed_bounds[i] && continue # pinned by an equality row
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
