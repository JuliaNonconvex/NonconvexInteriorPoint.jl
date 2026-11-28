# Slacked (Ipopt-style) mode: inequality constraints are rewritten with
# explicit slack variables, `c_I(x) + s = 0`, `s ≥ 0`, with the slack barrier
# `-μ Σ ln sᵢ`. Which inequalities are slacked is selected per constraint by
# [`BarrierSelection`](@ref): the remaining inequalities stay in the
# (possibly typed) barrier together with the bounds and the set (PSD)
# constraints. Only the slacked inequality Jacobians enter the KKT system
# (they are part of the extended equality set).
#
# The KKT system is the slack-eliminated Ipopt form in (dx, dλ):
#
#   [ H + Σx + δw I     J'                ] [dx ]   [ rhs_x ]
#   [ J               -δc I - diag(s²/μ) ] [dλ ] = [ rhs_λ ]
#
# with J = [J_E; J_I(slacked)], diag(s²/μ) on the slacked rows only,
#
#   rhs_x = -(g + J'λ),   rhs_λ = [ -cE ; -cI + s²λI/μ - 2s ],
#
# and the slack step recovered as `ds = -(s²/μ) dλI - (s²λI/μ - s)`.

mutable struct SlackedState
    n::Int
    mE::Int
    mI::Int
    x::Vector{Float64}
    s::Vector{Float64}
    lamE::Vector{Float64}
    lamI::Vector{Float64}
    gf::Vector{Float64} # barrier gradient in x (bounds + sets + barriered ineqs)
    cE::Vector{Float64}
    cI::Vector{Float64} # slacked ineq rows only
    J::Matrix{Float64} # (mE + mI) × n
    f::Float64
    mu::Float64
    last_E::Float64 # previous barrier KKT error (stagnation detection)
    K::Matrix{Float64}
    rhs::Vector{Float64}
    sol::Vector{Float64}
    vals::Vector{Float64}
    V::Matrix{Float64}
end

function SlackedState(m::IPMModel, x::Vector{Float64}, s::Vector{Float64}, mu::Float64)
    n = m.n
    mE = NonconvexInteriorPoint.mE(m)
    mI = length(s)
    N = n + mE + mI
    st = SlackedState(
        n,
        mE,
        mI,
        copy(x),
        copy(s),
        zeros(mE),
        zeros(mI),
        zeros(n),
        zeros(mE),
        zeros(mI),
        zeros(mE + mI, n),
        0.0,
        mu,
        Inf,
        zeros(N, N),
        zeros(N),
        zeros(N),
        zeros(N),
        zeros(N, N),
    )
    return st
end

# Per-inequality masks: true = slacked (excluded from the barrier).
_nl_slacked(m::IPMModel) = .!m.nlineq_barrier
_lin_slacked(m::IPMModel) = .!m.lin_ineq_barrier
function n_slacked_rows(m::IPMModel)
    return sum(getdim(c) for (c, b) in zip(m.nlineqs, m.nlineq_barrier) if !b; init=0) +
           count(!, m.lin_ineq_barrier)
end

# ---------------------------------------------------------------------------
# Barrier and Lagrangian evaluations (barriered inequalities included)
# ---------------------------------------------------------------------------

function slacked_barrier_value(
    m::IPMModel, x::AbstractVector, s::AbstractVector, mu::Float64, opts::IPMOptions
)
    f = _aug_objective(m, x)
    isfinite(f) || return Inf
    any(<=(0.0), s) && return Inf
    sel = opts.barrier
    φ = f - mu * (isempty(s) ? 0.0 : sum(log, s))
    ε = 1e-6
    # barriered inequality values
    ineq_vals = if m.jump !== nothing
        jm = m.jump
        g = zeros(jm.nineq + jm.neq)
        NonconvexCore.jump_constraint_values!(jm, g, x)
        rowsbar = ineq_barrier_rows(m)
        g[findall(rowsbar)]
    else
        vcat(
            [_vec(c(x)) for (k, c) in enumerate(m.nlineqs) if m.nlineq_barrier[k]]...,
            if (m.linAineq === nothing || !any(m.lin_ineq_barrier))
                Float64[]
            else
                (v=m.linAineq * x .- m.linbineq; v[m.lin_ineq_barrier])
            end,
        )
    end
    bounds_feasible = true
    feasible = all(v -> v <= -ε, ineq_vals)
    if sel.box
        @inbounds for i in 1:m.n
            m.fixed_bounds[i] && continue
            isfinite(m.xl[i]) &&
                (x[i] - m.xl[i] <= ε && (bounds_feasible=false; feasible=false))
            isfinite(m.xu[i]) &&
                (m.xu[i] - x[i] <= ε && (bounds_feasible=false; feasible=false))
        end
    end
    if m.strict[] || feasible
        for v in ineq_vals
            v < 0 || return Inf
            φ += _scalar_barrier(-v, mu, sel.ineq_type)
        end
        if sel.box
            @inbounds for i in 1:m.n
                m.fixed_bounds[i] && continue
                if isfinite(m.xl[i])
                    d = x[i] - m.xl[i]
                    d <= 0 && return Inf
                    φ += _scalar_barrier(d, mu, sel.box_type)
                end
                if isfinite(m.xu[i])
                    d = m.xu[i] - x[i]
                    d <= 0 && return Inf
                    φ += _scalar_barrier(d, mu, sel.box_type)
                end
            end
        end
    else
        for v in ineq_vals
            φ += mu * _log_barrier_ext(-v, ε)
        end
        if sel.box
            @inbounds for i in 1:m.n
                m.fixed_bounds[i] && continue
                isfinite(m.xl[i]) && (φ += mu * _log_barrier_ext(x[i] - m.xl[i], ε))
                isfinite(m.xu[i]) && (φ += mu * _log_barrier_ext(m.xu[i] - x[i], ε))
            end
        end
    end
    for (k, (c, set, mode)) in enumerate(m.setcons)
        bv = sd_barrier_value_c(
            c, set, x, Val(_eff_mode(mode, mu, opts)), sel.sd_type, opts
        )
        isfinite(bv) || return Inf
        φ += mu * bv
    end
    return φ
end

_slacked_barrier_closure(m, s, mu, opts) = x -> slacked_barrier_value(m, x, s, mu, opts)

# Barrier gradient in x (adjoint-based: no inequality Jacobians, only the
# objective pullback, the bound diagonals, the barriered-inequality adjoints
# and the set-barrier adjoints).
function slacked_barrier_gradient!(
    g::AbstractVector,
    m::IPMModel,
    x::AbstractVector,
    s::AbstractVector,
    mu::Float64,
    opts::IPMOptions,
)
    if m.jump !== nothing
        return _jump_barrier_gradient!(g, m, x, mu, opts)
    end
    _, gobj = value_gradient(_aug_obj_closure(m), x)
    gobj === nothing ? fill!(g, 0.0) : copyto!(g, gobj)
    sel = opts.barrier
    ε = 1e-6
    ineq_vals = vcat(
        [_vec(c(x)) for (k, c) in enumerate(m.nlineqs) if m.nlineq_barrier[k]]...,
        if (m.linAineq === nothing || !any(m.lin_ineq_barrier))
            Float64[]
        else
            (v=m.linAineq * x .- m.linbineq; v[m.lin_ineq_barrier])
        end,
    )
    feasible = all(v -> v <= -ε, ineq_vals)
    if sel.box
        @inbounds for i in 1:m.n
            isfinite(m.xl[i]) && (x[i] - m.xl[i] <= ε && (feasible = false))
            isfinite(m.xu[i]) && (m.xu[i] - x[i] <= ε && (feasible = false))
        end
    end
    if !(m.strict[] || feasible)
        _, gext = value_gradient(_slacked_barrier_closure(m, s, mu, opts), x)
        gext === nothing ? fill!(g, 0.0) : copyto!(g, gext)
        return g
    end
    for (k, c) in enumerate(m.nlineqs)
        m.nlineq_barrier[k] || continue
        v = _vec(c(x))
        w = -_scalar_barrier_deriv.(-v, mu, sel.ineq_type)
        jtprod!(g, c, x, w)
    end
    if m.linAineq !== nothing && any(.!m.lin_ineq_barrier)
        v = m.linAineq * x .- m.linbineq
        mask = m.lin_ineq_barrier
        w = -_scalar_barrier_deriv.(-v[mask], mu, sel.ineq_type)
        mul!(g, m.linAineq[mask, :]', w, 1.0, 1.0)
    end
    if sel.box
        @inbounds for i in 1:m.n
            if isfinite(m.xl[i])
                g[i] += _scalar_barrier_deriv(x[i] - m.xl[i], mu, sel.box_type)
            end
            if isfinite(m.xu[i])
                g[i] -= _scalar_barrier_deriv(m.xu[i] - x[i], mu, sel.box_type)
            end
        end
    end
    for (k, (c, set, mode)) in enumerate(m.setcons)
        effm = _eff_mode(mode, mu, opts)
        grad = similar(g)
        sd_barrier_grad_x!(grad, c, set, x, Val(effm), sel.sd_type, opts)
        g .+= mu .* grad
    end
    return g
end

# Lagrangian closure over the equality constraints and the slacked
# inequalities (their multipliers come from the extended equality set).
function _slacked_lagrangian(m::IPMModel, lamE::AbstractVector, lamI::AbstractVector)
    return function (x)
        L = _aug_objective(m, x)
        offsetE = 1
        for (k, c) in enumerate(m.nleqs)
            d = getdim(c)
            L += dot(view(lamE, offsetE:(offsetE + d - 1)), _vec(c(x)))
            offsetE += d
        end
        offsetI = 1
        for (k, c) in enumerate(m.nlineqs)
            m.nlineq_barrier[k] && continue
            d = getdim(c)
            L += dot(view(lamI, offsetI:(offsetI + d - 1)), _vec(c(x)))
            offsetI += d
        end
        return L
    end
end

# Evaluate the equality constraints (all rows) and the slacked inequality
# rows into preallocated vectors.
function eval_slacked_cons!(
    cE::AbstractVector, cI::AbstractVector, m::IPMModel, x::AbstractVector
)
    if m.jump !== nothing
        # the evaluator computes every constraint value in one pass
        jm = m.jump
        g = zeros(jm.nineq + jm.neq)
        NonconvexCore.jump_constraint_values!(jm, g, x)
        copyto!(view(cE, 1:jm.neq), view(g, (jm.nineq + 1):(jm.nineq + jm.neq)))
        rowsbar = ineq_barrier_rows(m)
        rows = [i for i in 1:jm.nineq if !rowsbar[i]]
        copyto!(view(cI, 1:length(rows)), view(g, rows))
        offset = jm.neq + 1
        if m.linAeq !== nothing
            r = offset:(offset + size(m.linAeq, 1) - 1)
            mul!(view(cE, r), m.linAeq, x)
            view(cE, r) .-= m.linbeq
        end
        offset = length(rows) + 1
        if m.linAineq !== nothing && any(.!m.lin_ineq_barrier)
            mask = m.lin_ineq_barrier
            v = m.linAineq * x .- m.linbineq
            r = offset:(offset + count(!, mask) - 1)
            copyto!(view(cI, r), v[.!mask])
        end
        return nothing
    end
    # eq rows (all)
    offset = 1
    for c in m.nleqs
        v = _vec(c(x))
        d = length(v)
        copyto!(view(cE, offset:(offset + d - 1)), v)
        offset += d
    end
    if m.linAeq !== nothing
        r = (offset):(offset + size(m.linAeq, 1) - 1)
        mul!(view(cE, r), m.linAeq, x)
        view(cE, r) .-= m.linbeq
        offset += size(m.linAeq, 1)
    end
    # slacked ineq rows
    offset = 1
    for (k, c) in enumerate(m.nlineqs)
        m.nlineq_barrier[k] && continue
        v = _vec(c(x))
        d = length(v)
        copyto!(view(cI, offset:(offset + d - 1)), v)
        offset += d
    end
    if m.linAineq !== nothing && any(.!m.lin_ineq_barrier)
        mask = m.lin_ineq_barrier
        v = m.linAineq * x .- m.linbineq
        r = offset:(offset + count(!, mask) - 1)
        copyto!(view(cI, r), v[.!mask])
        offset += count(!, mask)
    end
    return nothing
end

# Assemble the dense Jacobian of the extended equality set:
# rows [nl eq; lin eq; slacked nl ineq; slacked lin ineq].
function assemble_slacked_jacobian!(st::SlackedState, m::IPMModel)
    if m.jump !== nothing
        # the evaluator's Jacobian in one pass, in the dense layout
        # [equality rows; linear-eq rows; slacked ineq rows; linear ineq rows]
        # that matches st.cE/st.cI and the λE/λI block split
        jm = m.jump
        J = Matrix(NonconvexCore.jump_jacobian(jm, st.x))
        row = 1
        neq = jm.neq
        copyto!(
            view(st.J, row:(row + neq - 1), :), view(J, (jm.nineq + 1):(jm.nineq + neq), :)
        )
        row += neq
        if m.linAeq !== nothing
            copyto!(view(st.J, row:(row + size(m.linAeq, 1) - 1), :), Matrix(m.linAeq))
            row += size(m.linAeq, 1)
        end
        rowsbar = ineq_barrier_rows(m)
        rows = [i for i in 1:jm.nineq if !rowsbar[i]]
        for i in rows
            copyto!(view(st.J, row, :), view(J, i, :))
            row += 1
        end
        if m.linAineq !== nothing && any(.!m.lin_ineq_barrier)
            mask = m.lin_ineq_barrier
            As = m.linAineq[.!mask, :]
            copyto!(view(st.J, row:(row + size(As, 1) - 1), :), Matrix(As))
            row += size(As, 1)
        end
        return st.J
    end
    row = 1
    for (k, c) in enumerate(m.nleqs)
        d = getdim(c)
        _, J = value_jacobian(c, st.x)
        copyto!(view(st.J, row:(row + d - 1), :), J)
        row += d
    end
    if m.linAeq !== nothing
        copyto!(view(st.J, row:(row + size(m.linAeq, 1) - 1), :), m.linAeq)
        row += size(m.linAeq, 1)
    end
    for (k, c) in enumerate(m.nlineqs)
        m.nlineq_barrier[k] && continue
        d = getdim(c)
        _, J = value_jacobian(c, st.x)
        copyto!(view(st.J, row:(row + d - 1), :), J)
        row += d
    end
    if m.linAineq !== nothing && any(.!m.lin_ineq_barrier)
        mask = m.lin_ineq_barrier
        As = m.linAineq[.!mask, :]
        copyto!(view(st.J, row:(row + size(As, 1) - 1), :), As)
        row += size(As, 1)
    end
    return st.J
end

# Bound-barrier Hessian diagonal Σx (type-aware).
function _slacked_sigma_x(st::SlackedState, m::IPMModel, opts::IPMOptions)
    σ = similar(st.x)
    t = opts.barrier.box_type
    @inbounds for i in 1:st.n
        σ[i] =
            (
                if opts.barrier.box && isfinite(m.xl[i])
                    _scalar_barrier_second(x0=st.x[i], d=m.xl[i], mu=st.mu, type=t)
                else
                    0.0
                end
            ) + (
                if opts.barrier.box && isfinite(m.xu[i])
                    _scalar_barrier_second(x0=st.x[i], d=m.xu[i], mu=st.mu, type=t)
                else
                    0.0
                end
            )
    end
    return σ
end

# Second derivative of the bound barrier φ(x - d) w.r.t. x: φ''(x - d).
function _scalar_barrier_second(; x0::Real, d::Real, mu::Real, type::Symbol)
    dist = type === :log ? abs(x0 - d) : abs(x0 - d)
    t = type
    return if t === :log
        mu / dist^2
    elseif t === :inv
        2 * mu / dist^3
    else
        3 * mu / dist^4
    end
end

# Assemble the dense reduced KKT matrix and right-hand side.
function assemble_slacked_kkt!(
    st::SlackedState, m::IPMModel, opts::IPMOptions, δw::Float64, δc::Float64
)
    n, mE, mI = st.n, st.mE, st.mI
    N = n + mE + mI
    fill!(st.K, 0.0)
    # (1,1): Lagrangian Hessian (slacked constraints) + set barrier Hessians +
    # bound barrier diagonal + δw I
    H = if m.jump !== nothing
        _jump_slacked_lagrangian_hessian!(zeros(n, n), m, st.x, st.lamE, st.lamI)
    else
        try
            value_hessian(_slacked_lagrangian(m, st.lamE, st.lamI), st.x)
        catch
            zeros(n, n)
        end
    end
    for (k, (c, set, mode)) in enumerate(m.setcons)
        Hh = similar(H)
        sd_barrier_hess_x!(
            Hh, c, set, st.x, Val(_eff_mode(mode, st.mu, opts)), opts.barrier.sd_type, opts
        )
        H .+= st.mu .* Hh
    end
    σx = _slacked_sigma_x(st, m, opts)
    K11 = view(st.K, 1:n, 1:n)
    copyto!(K11, H)
    @inbounds for i in 1:n
        K11[i, i] += σx[i] + δw
    end
    # (1,2)/(2,1) and (2,2)
    copyto!(view(st.K, 1:n, (n + 1):N), st.J')
    copyto!(view(st.K, (n + 1):N, 1:n), st.J)
    K22 = view(st.K, (n + 1):N, (n + 1):N)
    for i in 1:(mE + mI)
        K22[i, i] = -δc
    end
    @inbounds for i in 1:mI
        K22[mE + i, mE + i] -= st.s[i]^2 / st.mu
    end
    # right-hand side
    gx = st.gf .+ view(st.J, :, :)' * [st.lamE; st.lamI]
    st.rhs[1:n] .= .-gx
    st.rhs[(n + 1):(n + mE)] .= .-st.cE
    @inbounds for i in 1:mI
        st.rhs[n + mE + i] = -st.cI[i] + (st.s[i]^2 * st.lamI[i] / st.mu - 2 * st.s[i])
    end
    return nothing
end

function solve_slacked_kkt_dense!(
    st::SlackedState, m::IPMModel, opts::IPMOptions; δc0::Float64=1e-8
)
    n = st.n
    δc = δc0
    δw = 1e-10
    for trial in 1:14
        assemble_slacked_kkt!(st, m, opts, δw, δc)
        _scaled_eigen_kkt_solve!(st, n) && return true
        δc *= 10.0
        δw = max(δw * 100.0, 1e-4)
    end
    return false
end

# Slack step recovered from the eliminated slack equation.
# The directional change of the slack-consistency residual c_I + s along
# (dx, ds): J_I dx + ds for the slacked inequality rows.
function _slacked_cons_change(
    m::IPMModel, st::SlackedState, dx::AbstractVector, ds::AbstractVector
)
    if m.jump !== nothing
        Jsl = _jump_slacked_jacobian(m, st.x)
        return Jsl[1:st.mI, :] * dx .+ ds
    end
    Jdx = zeros(st.mI)
    offset = 1
    for (k, c) in enumerate(m.nlineqs)
        m.nlineq_barrier[k] && continue
        d = getdim(c)
        Jdx[offset:(offset + d - 1)] .= jacprod(c, st.x, dx)
        offset += d
    end
    if m.linAineq !== nothing && any(.!m.lin_ineq_barrier)
        mask = m.lin_ineq_barrier
        As = m.linAineq[.!mask, :]
        mul!(Jdx, As, dx)
    end
    return Jdx .+ ds
end

function _slacked_ds(st::SlackedState)
    ds = similar(st.s)
    @inbounds for i in 1:st.mI
        ds[i] =
            -(st.s[i]^2 / st.mu) * st.sol[st.n + st.mE + i] -
            (st.s[i]^2 * st.lamI[i] / st.mu - st.s[i])
    end
    return ds
end

# Exact-penalty merit for the line search.
function slacked_merit(
    m::IPMModel,
    st::SlackedState,
    x::AbstractVector,
    s::AbstractVector,
    opts::IPMOptions;
    ν=nothing,
)
    φ = slacked_barrier_value(m, x, s, st.mu, opts)
    isfinite(φ) || return Inf
    cE = zeros(st.mE)
    cI = zeros(st.mI)
    eval_slacked_cons!(cE, cI, m, x)
    ν = if ν === nothing
        max(1.0, maximum(abs, st.lamE; init=0.0), maximum(abs, st.lamI; init=0.0) + 10.0)
    else
        ν
    end
    return φ + ν * (sum(abs, cE) + sum(abs, cI .+ s))
end

# Step bound from the variable bounds and the slacks.
function _slacked_bound_step(
    x::AbstractVector,
    s::AbstractVector,
    dx::AbstractVector,
    ds::AbstractVector,
    m::IPMModel,
    opts::IPMOptions,
    mu::Float64,
)
    αmax = 1.0
    @inbounds for i in 1:m.n
        m.fixed_bounds[i] && continue
        if dx[i] < -1e-16 && isfinite(m.xl[i])
            αmax = min(αmax, (m.xl[i] - x[i]) / dx[i])
        elseif dx[i] > 1e-16 && isfinite(m.xu[i])
            αmax = min(αmax, (m.xu[i] - x[i]) / dx[i])
        end
    end
    @inbounds for i in eachindex(s)
        ds[i] < -1e-16 && (αmax = min(αmax, -s[i] / ds[i]))
    end
    α = opts.tau * αmax
    return α < 0.0 ? 0.0 : min(α, 1.0)
end

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

function _ipm_solve_slacked!(m::IPMModel, opts::IPMOptions, x0::Vector{Float64})
    n = m.n
    mE = NonconvexInteriorPoint.mE(m)
    mI = n_slacked_rows(m)
    mode = _solver_mode(m, opts)
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
    # feasibility stage for infeasible starts (the slacks absorb the
    # inequality violations by construction, but the bounds and equalities
    # still benefit from a pre-barrier feasibility push)
    if !_strictly_feasible_zone(m, x)
        opts.verbose && println("feasibility stage: restoring feasibility")
        _feasibility_stage!(m, x, opts)
    end
    cE0 = zeros(mE)
    cI0 = zeros(mI)
    eval_slacked_cons!(cE0, cI0, m, x)
    s = [max(-cI0[i] + 1.0, 1.0) for i in 1:mI]
    mu = opts.mu_init
    st = SlackedState(m, x, s, mu)
    @inbounds for i in 1:mI
        st.lamI[i] = mu / s[i]
    end
    xt = similar(x)
    stt = similar(s)

    for iter in 1:opts.max_iter
        # --- evaluate ---
        st.f = _aug_objective(m, st.x)
        m.counter[] += 1 + length(m.relaxed_eqs) + length(m.relaxed_ineqs)
        slacked_barrier_gradient!(st.gf, m, st.x, st.s, st.mu, opts)
        eval_slacked_cons!(st.cE, st.cI, m, st.x)
        mode === :direct && assemble_slacked_jacobian!(st, m)

        # --- convergence ---
        # true primal feasibility uses all ineq rows (barriered + slacked)
        cI_all = zeros(NonconvexInteriorPoint.mI(m))
        cE_all = zeros(mE)
        eval_nonlinear_cons!(cE_all, cI_all, m, st.x)
        eval_linear_cons!(cE_all, cI_all, m, st.x)
        primal_inf = max(
            isempty(cE_all) ? 0.0 : maximum(abs, cE_all),
            isempty(cI_all) ? 0.0 : maximum(v -> max(v, 0.0), cI_all),
            _set_violation(m, st.x),
        )
        rstat = copy(st.gf)
        _jteq_λ!(rstat, m, st.x, st.lamE)
        if m.jump !== nothing
            Jsl = _jump_slacked_jacobian(m, st.x)
            mul!(rstat, Jsl', [st.lamI; st.lamE], 1.0, 1.0)
        else
            for (k, c) in enumerate(m.nlineqs)
                m.nlineq_barrier[k] && continue
                d = getdim(c)
                jtprod!(
                    rstat,
                    c,
                    st.x,
                    view(st.lamI, _lamI_offset(m, k):(_lamI_offset(m, k) + d - 1)),
                )
            end
            if m.linAineq !== nothing && any(.!m.lin_ineq_barrier)
                mask = m.lin_ineq_barrier
                n_nl_sl = sum(
                    getdim(c) for (c, b) in zip(m.nlineqs, m.nlineq_barrier) if !b; init=0
                )
                As = m.linAineq[.!mask, :]
                mul!(
                    rstat,
                    As',
                    view(st.lamI, (n_nl_sl + 1):(n_nl_sl + size(As, 1))),
                    1.0,
                    1.0,
                )
            end
        end
        dual_inf = isempty(rstat) ? 0.0 : maximum(abs, rstat)
        s_dual_inf = isempty(st.lamI) ? 0.0 : maximum(abs, st.lamI .- st.mu ./ st.s)
        scale = max(1.0, isempty(st.gf) ? 0.0 : maximum(abs, st.gf))
        if st.mu <= 1e-9 && primal_inf <= opts.tol && dual_inf <= max(100 * opts.tol, 1e-4)
            return _slacked_final_result(m, st, :converged, iter, opts)
        end
        if st.mu <= opts.mu_min && primal_inf <= opts.tol && dual_inf <= opts.tol * scale
            return _slacked_final_result(m, st, :converged, iter, opts)
        end
        # practical convergence once the barrier has essentially vanished
        if st.mu <= 1e-5 &&
            primal_inf <= opts.tol &&
            dual_inf <= max(opts.tol * scale, 1e-4)
            return _slacked_final_result(m, st, :converged, iter, opts)
        end

        # --- KKT solve ---
        assemble_slacked_kkt!(st, m, opts, 0.0, 1e-8)
        ok = mode === :direct ? solve_slacked_kkt_dense!(st, m, opts) : true
        if mode === :krylov
            op = SlackedKKTOp(m, st, opts, 1e-10, 1e-8)
            prec = if n + mE + mI <= 1500
                H = if m.jump !== nothing
                    _jump_slacked_lagrangian_hessian!(zeros(n, n), m, st.x, st.lamE, st.lamI)
                else
                    try
                        value_hessian(_slacked_lagrangian(m, st.lamE, st.lamI), st.x)
                    catch
                        zeros(n, n)
                    end
                end
                σx = _slacked_sigma_x(st, m, opts)
                @inbounds for i in 1:n
                    H[i, i] += σx[i] + 1e-10
                end
                bp = KKTBlockPrecond(H, view(op.precond, (n + 1):(n + mE + mI)), n)
                bp === nothing ? KKTDiagPrecond(op.precond) : bp
            else
                KKTDiagPrecond(op.precond)
            end
            sol, stats = Krylov.minres(
                op,
                st.rhs;
                M=prec,
                ldiv=true,
                rtol=opts.krylov_tol,
                atol=opts.krylov_tol,
                itmax=opts.krylov_maxiter,
                verbose=0,
            )
            ok = stats.solved
            ok && (st.sol .= sol)
        end
        ok || (ok = solve_slacked_kkt_dense!(st, m, opts))
        ok || return _slacked_final_result(m, st, :diverged, iter, opts)
        dx = view(st.sol, 1:n)
        dlam = view(st.sol, (n + 1):(n + mE + mI))
        dlamI = view(dlam, (mE + 1):(mE + mI))
        ds = _slacked_ds(st)
        # the direction must be a descent for the merit; if the Krylov solve
        # produced a poor direction, fall back to the dense factorization
        # the exact-penalty weight must dominate the *next* multipliers
        # λ + dλ, not just the current ones — otherwise the Newton direction
        # is not a merit-descent direction (observed: dφ > 0, α → 1e-6 crawl)
        ν = max(
            1.0,
            maximum(abs, st.lamE; init=0.0) + maximum(abs, view(dlam, 1:mE)),
            maximum(abs, st.lamI; init=0.0) + maximum(abs, dlamI) + 10.0,
        )
        # directional derivative of the slacked exact-penalty merit, as the
        # theoretical upper bound g'd - ν‖c‖₁: the linearized ‖c + Jd‖
        # variant overestimates the penalty change for nonlinear rows (the
        # quadratic curvature term dominates for large steps) and rejects
        # legitimate Newton corrections
        dφ =
            dot(st.gf, dx) - st.mu * (isempty(st.s) ? 0.0 : sum(ds ./ st.s)) -
            ν * (sum(abs, st.cE) + sum(abs, st.cI .+ st.s))
        if dφ >= 0.0 && mode === :krylov
            ok = solve_slacked_kkt_dense!(st, m, opts)
            ok || return _slacked_final_result(m, st, :diverged, iter, opts)
            dx = view(st.sol, 1:n)
            dlam = view(st.sol, (n + 1):(n + mE + mI))
            dlamI = view(dlam, (mE + 1):(mE + mI))
            ds = _slacked_ds(st)
            dφ =
                dot(st.gf, dx) - st.mu * (isempty(st.s) ? 0.0 : sum(ds ./ st.s)) -
                ν * (sum(abs, st.cE) + sum(abs, st.cI .+ st.s))
        end

        # --- backtracking on the exact-penalty merit ---
        α = _slacked_bound_step(st.x, st.s, dx, ds, m, opts, st.mu)
        φ0 = slacked_merit(m, st, st.x, st.s, opts; ν=ν)
        accepted = false
        while α > 1e-14
            xt .= st.x .+ α .* dx
            stt .= st.s .+ α .* ds
            φ = slacked_merit(m, st, xt, stt, opts; ν=ν)
            if isfinite(φ) && φ <= φ0 + 1e-4 * α * dφ
                accepted = true
                break
            end
            α *= 0.5
        end
        if !accepted
            if primal_inf <= 100 * opts.tol && dual_inf <= max(100 * opts.tol, 1e-4)
                return _slacked_final_result(m, st, :converged, iter, opts)
            end
            return _slacked_final_result(m, st, :diverged, iter, opts)
        end
        st.x .= st.x .+ α .* dx
        st.s .= st.s .+ α .* ds
        st.lamE .+= α .* view(dlam, 1:mE)
        st.lamI .+= α .* dlamI
        st.lamI .= max.(st.lamI, 0.0)
        # μ-aware slack floor: the fixed s_min clamp freezes the slacks far
        # above the central path once μ falls below it (s* ≈ μ/λ*), pinning
        # the implied multipliers λI = μ/s near zero and turning the Σ block
        # μ/s² into a damping wall for the multiplier updates.
        st.s .= max.(st.s, min(opts.s_min, 1e-3 * st.mu))

        # --- μ update (quality-function rule) with a stagnation escape ---
        E_mu = max(dual_inf, s_dual_inf, primal_inf)
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
            return _slacked_final_result(m, st, :max_iter, iter, opts)
        end
    end
    return _slacked_final_result(m, st, :max_iter, opts.max_iter, opts)
end

function _slacked_final_result(
    m::IPMModel, st::SlackedState, status::Symbol, iterations::Int, opts::IPMOptions
)
    if status in (:diverged, :max_iter)
        cI_all = zeros(NonconvexInteriorPoint.mI(m))
        cE_all = zeros(st.mE)
        eval_nonlinear_cons!(cE_all, cI_all, m, st.x)
        eval_linear_cons!(cE_all, cI_all, m, st.x)
        primal_inf = max(
            isempty(cE_all) ? 0.0 : maximum(abs, cE_all),
            isempty(cI_all) ? 0.0 : maximum(v -> max(v, 0.0), cI_all),
            _set_violation(m, st.x),
        )
        rstat = copy(st.gf)
        _jteq_λ!(rstat, m, st.x, st.lamE)
        if m.jump !== nothing
            Jsl = _jump_slacked_jacobian(m, st.x)
            mul!(rstat, Jsl', [st.lamI; st.lamE], 1.0, 1.0)
        else
            for (k, c) in enumerate(m.nlineqs)
                m.nlineq_barrier[k] && continue
                d = getdim(c)
                jtprod!(
                    rstat,
                    c,
                    st.x,
                    view(st.lamI, _lamI_offset(m, k):(_lamI_offset(m, k) + d - 1)),
                )
            end
        end
        dual_inf = isempty(rstat) ? 0.0 : maximum(abs, rstat)
        # the projected test uses the true objective gradient: the explicit
        # multipliers in rstat and the barrier terms must not be
        # double-counted by the NNLS
        gf_true = zeros(m.n)
        _objective_gradient(m, st.x, gf_true)
        dual_proj = _projected_dual(m, st.x, gf_true)
        if primal_inf <= max(1000 * opts.tol, 1e-4) &&
            min(dual_inf, dual_proj) <= max(100 * opts.tol, 1e-4)
            status = :converged
        end
    end
    fmin = m.obj === nothing ? 0.0 : Float64(m.obj(st.x))
    return IPMResult(copy(st.x), fmin, status, m.counter[], iterations)
end

# Offset of the k-th slacked nonlinear inequality constraint within the λI
# vector.
function _lamI_offset(m::IPMModel, k::Int)
    offset = 1
    for j in 1:(k - 1)
        m.nlineq_barrier[j] && continue
        offset += getdim(m.nlineqs[j])
    end
    return offset
end

# ---------------------------------------------------------------------------
# Krylov operator for the slacked mode
# ---------------------------------------------------------------------------

mutable struct SlackedKKTOp
    m::IPMModel
    st::SlackedState
    opts::IPMOptions
    δw::Float64
    δc::Float64
    σx::Vector{Float64}
    precond::Vector{Float64}
end

function SlackedKKTOp(
    m::IPMModel, st::SlackedState, opts::IPMOptions, δw::Float64, δc::Float64
)
    σx = _slacked_sigma_x(st, m, opts)
    n, mE, mI = st.n, st.mE, st.mI
    d = fill(max(δc, 1.0), n + mE + mI)
    @inbounds for i in 1:n
        d[i] = max(abs(σx[i]) + δw, 1.0)
    end
    @inbounds for i in 1:mI
        d[n + mE + i] += st.s[i]^2 / st.mu
    end
    return SlackedKKTOp(m, st, opts, δw, δc, σx, d)
end

Base.size(op::SlackedKKTOp) = (op.st.n + op.st.mE + op.st.mI, op.st.n + op.st.mE + op.st.mI)
Base.size(op::SlackedKKTOp, d::Int) = size(op)[d]
Base.eltype(::SlackedKKTOp) = Float64

function LinearAlgebra.mul!(y, op::SlackedKKTOp, u)
    m = op.m
    st = op.st
    n, mE, mI = st.n, st.mE, st.mI
    dx = view(u, 1:n)
    dλ = view(u, (n + 1):(n + mE + mI))
    yx = view(y, 1:n)
    yλ = view(y, (n + 1):(n + mE + mI))
    # (1,1)
    if m.jump !== nothing
        _jump_slacked_lagrangian_hessprod!(yx, m, st.x, dx, st.lamE, st.lamI)
    else
        hvp!(yx, _slacked_lagrangian(m, st.lamE, st.lamI), st.x, dx)
    end
    for (k, (c, set, mode)) in enumerate(m.setcons)
        effm = _eff_mode(mode, st.mu, op.opts)
        bar = similar(yx)
        sd_barrier_hess_x_prod!(
            bar, c, set, st.x, dx, Val(effm), op.opts.barrier.sd_type, op.opts
        )
        yx .+= st.mu .* bar
    end
    @inbounds for i in 1:n
        yx[i] += (op.σx[i] + op.δw) * dx[i]
    end
    # (1,2): J' dλ (eq rows + slacked ineq rows)
    offset = 1
    if m.jump !== nothing
        Jsl = _jump_slacked_jacobian(m, st.x)
        mul!(yx, Jsl', dλ, 1.0, 1.0)
    else
        for (k, c) in enumerate(m.nleqs)
            d = getdim(c)
            jtprod!(yx, c, st.x, view(dλ, offset:(offset + d - 1)))
            offset += d
        end
        if m.linAeq !== nothing
            mul!(yx, m.linAeq', view(dλ, offset:(offset + size(m.linAeq, 1) - 1)), 1.0, 1.0)
            offset += size(m.linAeq, 1)
        end
        for (k, c) in enumerate(m.nlineqs)
            m.nlineq_barrier[k] && continue
            d = getdim(c)
            jtprod!(yx, c, st.x, view(dλ, offset:(offset + d - 1)))
            offset += d
        end
    end
    if m.linAineq !== nothing && any(.!m.lin_ineq_barrier)
        mask = m.lin_ineq_barrier
        As = m.linAineq[.!mask, :]
        mul!(yx, As', view(dλ, offset:(offset + size(As, 1) - 1)), 1.0, 1.0)
        offset += size(As, 1)
    end
    # (2,1): J dx
    offset = 1
    if m.jump !== nothing
        Jsl = _jump_slacked_jacobian(m, st.x)
        mul!(yλ, Jsl, dx)
    else
        for (k, c) in enumerate(m.nleqs)
            d = getdim(c)
            v = jacprod(c, st.x, dx)
            yλ[offset:(offset + d - 1)] .= v
            offset += d
        end
        if m.linAeq !== nothing
            mul!(view(yλ, offset:(offset + size(m.linAeq, 1) - 1)), m.linAeq, dx)
            offset += size(m.linAeq, 1)
        end
        for (k, c) in enumerate(m.nlineqs)
            m.nlineq_barrier[k] && continue
            d = getdim(c)
            v = jacprod(c, st.x, dx)
            yλ[offset:(offset + d - 1)] .= v
            offset += d
        end
    end
    if m.linAineq !== nothing && any(.!m.lin_ineq_barrier)
        mask = m.lin_ineq_barrier
        As = m.linAineq[.!mask, :]
        mul!(view(yλ, offset:(offset + size(As, 1) - 1)), As, dx)
        offset += size(As, 1)
    end
    # (2,2)
    @inbounds for i in 1:(mE + mI)
        yλ[i] -= op.δc * dλ[i]
    end
    @inbounds for i in 1:mI
        yλ[mE + i] -= (st.s[i]^2 / st.mu) * dλ[mE + i]
    end
    return y
end
