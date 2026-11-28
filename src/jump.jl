# JuMP-converted models: first- and second-order derivatives read from the
# shared MathOptInterface evaluator (the evaluator mutates internal buffers
# and cannot be traced by AD backends, so every derivative of a
# JuMP-converted model goes through these paths). The evaluator provides
# σ·∇²f + Σ μᵢ·∇²cᵢ for the whole model in one sparse pass; the rank-one
# ∇cᵢ∇cᵢ' terms of the barrier Hessian are added from its Jacobian.

# Second derivative of the scalar barriers (d²φ/dt² of `_scalar_barrier`).
function _scalar_barrier_second_deriv(t::Real, mu::Real, type::Symbol)
    if type === :log
        return mu / t^2
    elseif type === :inv
        return 2mu / t^3
    elseif type === :quad
        return 3mu / t^4
    end
    return throw(ArgumentError("unknown scalar barrier type :$(type)"))
end

# Evaluator weights for the barrier Hessian of a JuMP-converted model:
# `mu_w` are the ∇²cᵢ coefficients (−φ'(sᵢ)), `r1_w` the ∇cᵢ∇cᵢ' coefficients
# (φ''(sᵢ)), with the constraint values written into `gvals`. Equality rows
# (whose Hessians do not enter the primal barrier) and non-barriered
# inequality rows get zero weights.
function _jump_barrier_weights!(
    mu_w::AbstractVector,
    r1_w::AbstractVector,
    gvals::AbstractVector,
    m::IPMModel,
    x::AbstractVector,
    mu::Float64,
    opts::IPMOptions,
)
    jm = m.jump
    sel = opts.barrier
    rowsbar = ineq_barrier_rows(m)
    NonconvexCore.jump_constraint_values!(jm, gvals, x)
    ε = 1e-6
    feasible = all(i -> !(rowsbar[i] && gvals[i] > -ε), 1:jm.nineq)
    strict = m.strict[] || feasible
    fill!(mu_w, 0.0)
    fill!(r1_w, 0.0)
    for i in 1:jm.nineq
        rowsbar[i] || continue
        s = -gvals[i]
        if strict || s > ε
            mu_w[i] = -_scalar_barrier_deriv(s, mu, sel.ineq_type)
            r1_w[i] = _scalar_barrier_second_deriv(s, mu, sel.ineq_type)
        else
            # violated or near-boundary row: the merit keeps the C¹ linear
            # extension (bounded), but the Newton system carries the C²
            # quadratic curvature μ/ε² so the feasibility-restoring
            # direction stays well-conditioned
            mu_w[i] = mu / ε
            r1_w[i] = mu / ε^2
        end
    end
    return mu_w, r1_w
end

# Bound-barrier curvature for the JuMP Hessians: the strict second
# derivative, or zero on the linear extension (d ≤ ε in the extended
# regime).
function _jump_bound_curvature(
    m::IPMModel, x::AbstractVector, i::Int, mu::Float64, opts::IPMOptions, strict::Bool
)
    sel = opts.barrier
    cur = 0.0
    m.fixed_bounds[i] && return cur # pinned by an equality row, no curvature
    if isfinite(m.xl[i])
        d = x[i] - m.xl[i]
        cur += if strict || d > 1e-6
            _scalar_barrier_second_deriv(d, mu, sel.box_type)
        else
            mu / 1e-12
        end
    end
    if isfinite(m.xu[i])
        d = m.xu[i] - x[i]
        cur += if strict || d > 1e-6
            _scalar_barrier_second_deriv(d, mu, sel.box_type)
        else
            mu / 1e-12
        end
    end
    return cur
end

# Gradient of the barrier objective for a JuMP-converted model: the signed
# objective gradient from the evaluator, the barriered-inequality terms
# J'·w (w = −φ'(s) in the feasible regime, μ/ε in the extended regime) and
# the bound barriers. Mirrors the generic `barrier_gradient!` branches.
function _jump_barrier_gradient!(
    g::AbstractVector, m::IPMModel, x::AbstractVector, mu::Float64, opts::IPMOptions
)
    jm = m.jump
    sel = opts.barrier
    nineq, neq = jm.nineq, jm.neq
    nji = m.n_jump_ineq
    if m.jump_obj
        gobj = similar(g)
        NonconvexCore.jump_objective_gradient!(jm, gobj, x)
        copyto!(g, gobj)
        # the evaluator objective excludes the relaxed-constraint penalties
        for c in m.relaxed_eqs
            v = c(x)
            jtprod!(g, c, x, m.rho .* v)
        end
        for c in m.relaxed_ineqs
            v = c(x)
            jtprod!(g, c, x, m.rho .* max.(v, 0.0))
        end
    else
        _, go = value_gradient(_aug_obj_closure(m), x)
        go === nothing ? fill!(g, 0.0) : copyto!(g, go)
    end
    gvals = zeros(nineq + neq)
    NonconvexCore.jump_constraint_values!(jm, gvals, x)
    rowsbar = ineq_barrier_rows(m)
    # native barriered inequality values
    native_vals = vcat(
        [
            _vec(c(x)) for (k, c) in enumerate(m.nlineqs) if
            (nji > 0 ? k > 1 : true) && m.nlineq_barrier[k]
        ]...,
        if (m.linAineq === nothing || !any(m.lin_ineq_barrier))
            Float64[]
        else
            (v=m.linAineq * x .- m.linbineq; v[m.lin_ineq_barrier])
        end,
    )
    ε = 1e-6
    feasible = all(i -> !(rowsbar[i] && gvals[i] > -ε), 1:nineq)
    feasible &= all(v -> v <= -ε, native_vals)
    if sel.box
        @inbounds for i in 1:m.n
            m.fixed_bounds[i] && continue
            isfinite(m.xl[i]) && (x[i] - m.xl[i] <= ε && (feasible = false))
            isfinite(m.xu[i]) && (m.xu[i] - x[i] <= ε && (feasible = false))
        end
    end
    strict = m.strict[] || feasible
    w = zeros(nineq + neq)
    for i in 1:nineq
        rowsbar[i] || continue
        s = -gvals[i]
        w[i] = strict ? -_scalar_barrier_deriv(s, mu, sel.ineq_type) : mu / ε
    end
    J = NonconvexCore.jump_jacobian(jm, x)
    g .+= J' * w
    # native barriered rows: the adjoint (pullback) of each constraint
    for (k, c) in enumerate(m.nlineqs)
        (nji > 0 ? k > 1 : true) && m.nlineq_barrier[k] || continue
        v = _vec(c(x))
        wn = (
            if strict
                -_scalar_barrier_deriv.(-v, mu, sel.ineq_type)
            else
                fill(mu / ε, length(v))
            end
        )
        jtprod!(g, c, x, wn)
    end
    if m.linAineq !== nothing && any(m.lin_ineq_barrier)
        v = m.linAineq * x .- m.linbineq
        mask = m.lin_ineq_barrier
        wn = if strict
            -_scalar_barrier_deriv.(-v, mu, sel.ineq_type)
        else
            fill(mu / ε, length(v))
        end
        mul!(g, m.linAineq[mask, :]', wn[mask], 1.0, 1.0)
    end
    if sel.box
        @inbounds for i in 1:m.n
            m.fixed_bounds[i] && continue
            if isfinite(m.xl[i])
                d = x[i] - m.xl[i]
                g[i] += strict ? _scalar_barrier_deriv(d, mu, sel.box_type) : -mu / ε
            end
            if isfinite(m.xu[i])
                d = m.xu[i] - x[i]
                g[i] -= strict ? _scalar_barrier_deriv(d, mu, sel.box_type) : -mu / ε
            end
        end
    end
    return g
end

# The dense barrier Hessian ∇²φ_μ of a JuMP-converted model: the evaluator's
# Lagrangian Hessian with obj_factor = 1 and the barrier weights, plus the
# rank-one ∇cᵢ∇cᵢ' terms and the bound-barrier diagonals.
# Native part of the barrier for mixed models: the objective and relaxed
# penalties when they are not JuMP-backed, and the native barriered
# inequalities (no bounds — the bound curvature is added separately).
# AD-differentiable; `jump_feasible` carries the JuMP rows' feasibility so
# the strict/extension switch is shared.
function _native_barrier_part(
    m::IPMModel, x::AbstractVector, mu::Float64, opts::IPMOptions, jump_feasible::Bool
)
    sel = opts.barrier
    nji = m.n_jump_ineq
    φ = 0.0
    if m.jump_obj
        for c in m.relaxed_eqs
            v = c(x)
            φ += (m.rho / 2) * sum(abs2, v)
        end
        for c in m.relaxed_ineqs
            v = c(x)
            φ += (m.rho / 2) * sum(abs2, max.(v, 0.0))
        end
    else
        f = _aug_objective(m, x)
        isfinite(f) || return Inf
        φ = f
    end
    native_vals = vcat(
        [
            _vec(c(x)) for (k, c) in enumerate(m.nlineqs) if
            (nji > 0 ? k > 1 : true) && m.nlineq_barrier[k]
        ]...,
        if (m.linAineq === nothing || !any(m.lin_ineq_barrier))
            Float64[]
        else
            (v=m.linAineq * x .- m.linbineq; v[m.lin_ineq_barrier])
        end,
    )
    ε = 1e-6
    strict = m.strict[] || (jump_feasible && all(v -> v <= -ε, native_vals))
    if strict
        for v in native_vals
            v < 0 || return Inf
            φ += _scalar_barrier(-v, mu, sel.ineq_type)
        end
    else
        for v in native_vals
            φ += mu * _log_barrier_ext(-v, ε)
        end
    end
    return φ
end

# Whether the native part carries anything (objective/relaxed when not
# JuMP-backed, or native barriered inequalities).
function _has_native_barrier_part(m::IPMModel)
    m.jump === nothing && return true
    m.jump_obj || return true
    !isempty(m.relaxed_eqs) && return true
    !isempty(m.relaxed_ineqs) && return true
    nji = m.n_jump_ineq
    for (k, c) in enumerate(m.nlineqs)
        (nji > 0 ? k > 1 : true) && m.nlineq_barrier[k] && return true
    end
    m.linAineq !== nothing && any(m.lin_ineq_barrier) && return true
    return false
end

# The native part's contribution to the dense barrier Hessian.
function _native_barrier_hess!(H::AbstractMatrix, m, x, mu, opts, jump_feasible::Bool)
    _has_native_barrier_part(m) || return H
    ncl = z -> _native_barrier_part(m, z, mu, opts, jump_feasible)
    Hn = try
        value_hessian(ncl, x)
    catch
        return H
    end
    H .+= Hn
    return H
end

function _jump_barrier_hessian!(
    H::AbstractMatrix, m::IPMModel, x::AbstractVector, mu::Float64, opts::IPMOptions
)
    jm = m.jump
    sel = opts.barrier
    n = m.n
    nineq, neq = jm.nineq, jm.neq
    mu_w = zeros(nineq + neq)
    r1_w = zeros(nineq + neq)
    gvals = zeros(nineq + neq)
    _jump_barrier_weights!(mu_w, r1_w, gvals, m, x, mu, opts)
    hvals = zeros(length(jm.hess_rows))
    NonconvexCore.jump_hessian_lagrangian!(jm, hvals, x, m.jump_obj ? 1.0 : 0.0, mu_w)
    fill!(H, 0.0)
    @inbounds for k in eachindex(jm.hess_rows)
        i, j = jm.hess_rows[k], jm.hess_cols[k]
        H[i, j] += hvals[k]
        i == j || (H[j, i] += hvals[k])
    end
    J = Matrix(NonconvexCore.jump_jacobian(jm, x))
    @inbounds for i in 1:nineq
        r1_w[i] == 0.0 && continue
        for a in 1:n, b in 1:n
            H[a, b] += r1_w[i] * J[i, a] * J[i, b]
        end
    end
    # native part (objective/relaxed when not JuMP-backed + native barriers)
    jump_feasible = all(i -> !(ineq_barrier_rows(m)[i] && gvals[i] > -1e-6), 1:nineq)
    _native_barrier_hess!(H, m, x, mu, opts, jump_feasible)
    if sel.box
        ε = 1e-6
        strict_b =
            m.strict[] || all(
                i ->
                    m.fixed_bounds[i] || (
                        (!isfinite(m.xl[i]) || x[i] - m.xl[i] > ε) &&
                        (!isfinite(m.xu[i]) || m.xu[i] - x[i] > ε)
                    ),
                1:n,
            )
        @inbounds for i in 1:n
            H[i, i] += _jump_bound_curvature(m, x, i, mu, opts, strict_b)
        end
    end
    return H
end

# Hessian-vector product ∇²φ_μ(x)·dx for a JuMP-converted model (the
# matrix-free KKT operator): the evaluator product plus the rank-one and
# bound-diagonal terms.
function _jump_barrier_hessprod!(
    y::AbstractVector,
    m::IPMModel,
    x::AbstractVector,
    dx::AbstractVector,
    mu::Float64,
    opts::IPMOptions,
)
    jm = m.jump
    sel = opts.barrier
    n = m.n
    nineq, neq = jm.nineq, jm.neq
    mu_w = zeros(nineq + neq)
    r1_w = zeros(nineq + neq)
    gvals = zeros(nineq + neq)
    _jump_barrier_weights!(mu_w, r1_w, gvals, m, x, mu, opts)
    hvals = zeros(length(jm.hess_rows))
    NonconvexCore.jump_hessian_lagrangian!(jm, hvals, x, m.jump_obj ? 1.0 : 0.0, mu_w)
    fill!(y, 0.0)
    @inbounds for k in eachindex(jm.hess_rows)
        i, j = jm.hess_rows[k], jm.hess_cols[k]
        y[i] += hvals[k] * dx[j]
        i == j || (y[j] += hvals[k] * dx[i])
    end
    J = Matrix(NonconvexCore.jump_jacobian(jm, x))
    cdx = J * dx
    @inbounds for i in 1:nineq
        r1_w[i] == 0.0 && continue
        for a in 1:n
            y[a] += r1_w[i] * J[i, a] * cdx[i]
        end
    end
    # native part (objective/relaxed when not JuMP-backed + native barriers)
    if _has_native_barrier_part(m)
        jump_feasible = all(i -> !(ineq_barrier_rows(m)[i] && gvals[i] > -1e-6), 1:nineq)
        ncl = z -> _native_barrier_part(m, z, mu, opts, jump_feasible)
        try
            Hn = value_hessian(ncl, x)
            mul!(y, Hn, dx, 1.0, 1.0)
        catch
        end
    end
    if sel.box
        ε = 1e-6
        strict_b =
            m.strict[] || all(
                i ->
                    m.fixed_bounds[i] || (
                        (!isfinite(m.xl[i]) || x[i] - m.xl[i] > ε) &&
                        (!isfinite(m.xu[i]) || m.xu[i] - x[i] > ε)
                    ),
                1:n,
            )
        @inbounds for i in 1:n
            y[i] += _jump_bound_curvature(m, x, i, mu, opts, strict_b) * dx[i]
        end
    end
    return y
end

# The slacked-Lagrangian weights of a JuMP-converted model: the slacked
# inequality multipliers λI on the slacked rows (zero on the barriered rows)
# and the equality multipliers λE.
function _jump_slacked_weights(m::IPMModel, lamE::AbstractVector, lamI::AbstractVector)
    jm = m.jump
    nji = m.n_jump_ineq
    w = zeros(jm.nineq + jm.neq)
    rowsbar = ineq_barrier_rows(m)
    # the slacked jump rows carry the first λI entries (the row order puts
    # the jump block first)
    k = 1
    for i in 1:jm.nineq
        rowsbar[i] && continue
        w[i] = lamI[k]
        k += 1
    end
    for i in 1:jm.neq
        w[jm.nineq + i] = lamE[i]
    end
    return w
end

# Dense Hessian of the slacked Lagrangian for a JuMP-converted model.
function _jump_slacked_lagrangian_hessian!(
    H::AbstractMatrix,
    m::IPMModel,
    x::AbstractVector,
    lamE::AbstractVector,
    lamI::AbstractVector,
)
    jm = m.jump
    hvals = zeros(length(jm.hess_rows))
    NonconvexCore.jump_hessian_lagrangian!(
        jm, hvals, x, m.jump_obj ? 1.0 : 0.0, _jump_slacked_weights(m, lamE, lamI)
    )
    fill!(H, 0.0)
    @inbounds for k in eachindex(jm.hess_rows)
        i, j = jm.hess_rows[k], jm.hess_cols[k]
        H[i, j] += hvals[k]
        i == j || (H[j, i] += hvals[k])
    end
    # the native part (objective/relaxed when not JuMP-backed, the native
    # equality and slacked-inequality multiplier terms) via AD
    if _has_native_slacked_lagrangian(m)
        ncl = z -> _native_slacked_lagrangian_value(m, z, lamE, lamI)
        try
            H .+= value_hessian(ncl, x)
        catch
        end
    end
    return H
end

# Hessian-vector product of the slacked Lagrangian for a JuMP-converted
# model.
function _jump_slacked_lagrangian_hessprod!(
    y::AbstractVector,
    m::IPMModel,
    x::AbstractVector,
    dx::AbstractVector,
    lamE::AbstractVector,
    lamI::AbstractVector,
)
    jm = m.jump
    hvals = zeros(length(jm.hess_rows))
    NonconvexCore.jump_hessian_lagrangian!(
        jm, hvals, x, 1.0, _jump_slacked_weights(m, lamE, lamI)
    )
    fill!(y, 0.0)
    @inbounds for k in eachindex(jm.hess_rows)
        i, j = jm.hess_rows[k], jm.hess_cols[k]
        y[i] += hvals[k] * dx[j]
        i == j || (y[j] += hvals[k] * dx[i])
    end
    return y
end

# The slacked KKT's constraint Jacobian for a JuMP-converted model: the
# slacked inequality rows followed by the equality rows.
function _jump_slacked_jacobian(m::IPMModel, x::AbstractVector)
    jm = m.jump
    nji, nje = m.n_jump_ineq, m.n_jump_eq
    parts = Matrix{Float64}[]
    if nji > 0
        J = Matrix(NonconvexCore.jump_jacobian(jm, x))
        rowsbar = ineq_barrier_rows(m)
        rows = [i for i in 1:jm.nineq if !rowsbar[i]]
        push!(parts, J[rows, :])
    end
    # native slacked inequality rows (masked)
    for (k, c) in enumerate(m.nlineqs)
        (nji > 0 ? k > 1 : true) && !m.nlineq_barrier[k] || continue
        _, J = value_jacobian(c, x)
        push!(parts, Matrix(J))
    end
    if m.linAineq !== nothing && any(.!m.lin_ineq_barrier)
        push!(parts, Matrix(m.linAineq[.!m.lin_ineq_barrier, :]))
    end
    # equality rows
    if nje > 0
        push!(
            parts,
            Matrix(
                NonconvexCore.jump_jacobian(jm, x)[(jm.nineq + 1):(jm.nineq + jm.neq), :]
            ),
        )
    end
    for c in m.nleqs[(nje > 0 ? 2 : 1):end]
        _, J = value_jacobian(c, x)
        push!(parts, Matrix(J))
    end
    if m.linAeq !== nothing # fixed-variable pins live in the linear block
        push!(parts, Matrix(m.linAeq))
    end
    isempty(parts) && return zeros(0, m.n)
    return reduce(vcat, parts)
end

# The equality rows of the shared evaluator's Jacobian (JuMP-converted
# models): for the KKT operators' J_E blocks.
function _jump_eq_jacobian(m::IPMModel, x::AbstractVector)
    jm = m.jump
    nje = m.n_jump_eq
    parts = Matrix{Float64}[]
    if nje > 0
        push!(
            parts,
            Matrix(
                NonconvexCore.jump_jacobian(jm, x)[(jm.nineq + 1):(jm.nineq + jm.neq), :]
            ),
        )
    end
    for c in m.nleqs[(nje > 0 ? 2 : 1):end]
        _, J = value_jacobian(c, x)
        push!(parts, Matrix(J))
    end
    if m.linAeq !== nothing # fixed-variable pins live in the linear block
        push!(parts, Matrix(m.linAeq))
    end
    isempty(parts) && return zeros(0, m.n)
    return reduce(vcat, parts)
end

# Equality-multiplier term g += J_E' λE for a JuMP-converted model (the
# equality rows of the shared evaluator's Jacobian).
function _jump_jteq_λ!(
    r::AbstractVector, m::IPMModel, x::AbstractVector, lamE::AbstractVector
)
    jm = m.jump
    isempty(lamE) && return r
    nje = m.n_jump_eq
    if nje > 0
        J = NonconvexCore.jump_jacobian(jm, x)
        Jeq = J[(jm.nineq + 1):end, :]
        r .+= Jeq' * view(lamE, 1:nje)
    end
    offset = nje + 1
    for c in m.nleqs[(nje > 0 ? 2 : 1):end]
        d = getdim(c)
        jtprod!(r, c, x, view(lamE, offset:(offset + d - 1)))
        offset += d
    end
    if m.linAeq !== nothing # fixed-variable pins live in the linear block
        r .+= m.linAeq' * view(lamE, offset:length(lamE))
    end
    return r
end

# Gradient of the feasibility-stage violation objective for a
# JuMP-converted model: J'·a with a = max(cᵢ + ε, 0) on the inequality rows
# and cᵢ on the equality rows, plus the bound-violation terms.
function _jump_violation_gradient!(g::AbstractVector, m::IPMModel, x::AbstractVector)
    jm = m.jump
    ε = 1e-2
    nineq, neq = jm.nineq, jm.neq
    gvals = zeros(nineq + neq)
    NonconvexCore.jump_constraint_values!(jm, gvals, x)
    a = zeros(nineq + neq)
    @inbounds for i in 1:nineq
        gvals[i] + ε > 0 && (a[i] = gvals[i] + ε)
    end
    @inbounds for i in (nineq + 1):(nineq + neq)
        a[i] = gvals[i]
    end
    J = NonconvexCore.jump_jacobian(jm, x)
    mul!(g, J', a)
    if m.linAeq !== nothing # fixed-variable pins
        mul!(g, m.linAeq', m.linAeq * x .- m.linbeq, 1.0, 1.0)
    end
    # native constraint rows (mixed models)
    nji, nje = m.n_jump_ineq, m.n_jump_eq
    for (k, c) in enumerate(m.nlineqs)
        (nji > 0 ? k > 1 : true) || continue
        v = _vec(c(x))
        w = max.(v .+ ε, 0.0)
        jtprod!(g, c, x, w)
    end
    for c in m.nleqs[(nje > 0 ? 2 : 1):end]
        jtprod!(g, c, x, _vec(c(x)))
    end
    @inbounds for i in 1:m.n
        m.fixed_bounds[i] && continue
        if isfinite(m.xl[i]) && x[i] < m.xl[i] + ε
            g[i] -= m.xl[i] + ε - x[i]
        end
        if isfinite(m.xu[i]) && x[i] > m.xu[i] - ε
            g[i] += x[i] - (m.xu[i] - ε)
        end
    end
    return g
end

# Dense Hessian of the feasibility-stage violation objective for a
# JuMP-converted model: one evaluator Lagrangian-Hessian pass (the weighted
# constraint Hessians) plus the rank-one terms of the active rows and the
# bound-violation diagonals. Assembling it column-by-column through the
# Hessian product would cost one evaluator Hessian per column.
function _jump_violation_hessian!(H::AbstractMatrix, m::IPMModel, x::AbstractVector)
    jm = m.jump
    ε = 1e-2
    n = m.n
    nineq, neq = jm.nineq, jm.neq
    gvals = zeros(nineq + neq)
    NonconvexCore.jump_constraint_values!(jm, gvals, x)
    mu_w = zeros(nineq + neq)
    active = zeros(Bool, nineq + neq)
    @inbounds for i in 1:nineq
        if gvals[i] + ε > 0
            mu_w[i] = gvals[i] + ε
            active[i] = true
        end
    end
    @inbounds for i in (nineq + 1):(nineq + neq)
        mu_w[i] = gvals[i]
        active[i] = true
    end
    hvals = zeros(length(jm.hess_rows))
    NonconvexCore.jump_hessian_lagrangian!(jm, hvals, x, 0.0, mu_w)
    fill!(H, 0.0)
    @inbounds for k in eachindex(jm.hess_rows)
        i, j = jm.hess_rows[k], jm.hess_cols[k]
        H[i, j] += hvals[k]
        i == j || (H[j, i] += hvals[k])
    end
    J = Matrix(NonconvexCore.jump_jacobian(jm, x))
    @inbounds for i in 1:(nineq + neq)
        active[i] || continue
        for a in 1:n, b in 1:n
            H[a, b] += J[i, a] * J[i, b]
        end
    end
    if m.linAeq !== nothing # fixed-variable pins
        A = Matrix(m.linAeq)
        mul!(H, A', A, 1.0, 1.0)
    end
    @inbounds for i in 1:n
        m.fixed_bounds[i] && continue
        isfinite(m.xl[i]) && x[i] < m.xl[i] + ε && (H[i, i] += 1.0)
        isfinite(m.xu[i]) && x[i] > m.xu[i] - ε && (H[i, i] += 1.0)
    end
    return H
end

# Hessian-vector product of the feasibility-stage violation objective for a
# JuMP-converted model: the evaluator's weighted Hessian (obj_factor = 0,
# μᵢ = the active violation weights) plus the rank-one terms of the active
# rows and the bound-violation diagonals.
function _jump_violation_hessprod!(
    y::AbstractVector, m::IPMModel, x::AbstractVector, dx::AbstractVector
)
    jm = m.jump
    ε = 1e-2
    n = m.n
    nineq, neq = jm.nineq, jm.neq
    gvals = zeros(nineq + neq)
    NonconvexCore.jump_constraint_values!(jm, gvals, x)
    mu_w = zeros(nineq + neq)
    active = zeros(Bool, nineq + neq)
    @inbounds for i in 1:nineq
        if gvals[i] + ε > 0
            mu_w[i] = gvals[i] + ε
            active[i] = true
        end
    end
    @inbounds for i in (nineq + 1):(nineq + neq)
        mu_w[i] = gvals[i]
        active[i] = true
    end
    hvals = zeros(length(jm.hess_rows))
    NonconvexCore.jump_hessian_lagrangian!(jm, hvals, x, 0.0, mu_w)
    fill!(y, 0.0)
    @inbounds for k in eachindex(jm.hess_rows)
        i, j = jm.hess_rows[k], jm.hess_cols[k]
        y[i] += hvals[k] * dx[j]
        i == j || (y[j] += hvals[k] * dx[i])
    end
    J = Matrix(NonconvexCore.jump_jacobian(jm, x))
    cdx = J * dx
    @inbounds for i in 1:(nineq + neq)
        active[i] || continue
        for a in 1:n
            y[a] += J[i, a] * cdx[i]
        end
    end
    if m.linAeq !== nothing # fixed-variable pins
        mul!(y, m.linAeq', m.linAeq * dx, 1.0, 1.0)
    end
    # native constraint rows (mixed models): AD through a closure covering
    # the native part of the violation objective
    nji, nje = m.n_jump_ineq, m.n_jump_eq
    if _has_native_violation_part(m)
        ncl = z -> _native_violation_part(m, z)
        try
            Hn = value_hessian(ncl, x)
            mul!(y, Hn, dx, 1.0, 1.0)
        catch
        end
    end
    @inbounds for i in 1:n
        m.fixed_bounds[i] && continue
        if isfinite(m.xl[i]) && x[i] < m.xl[i] + ε
            y[i] += dx[i]
        end
        if isfinite(m.xu[i]) && x[i] > m.xu[i] - ε
            y[i] += dx[i]
        end
    end
    return y
end

# Whether the native part of the violation objective carries anything.
function _has_native_violation_part(m::IPMModel)
    nji, nje = m.n_jump_ineq, m.n_jump_eq
    for c in m.nleqs[(nje > 0 ? 2 : 1):end]
        return true
    end
    for (k, c) in enumerate(m.nlineqs)
        (nji > 0 ? k > 1 : true) && return true
    end
    return false
end

# The native part of the feasibility-stage violation objective (mixed
# models): the native inequality margin terms and the native equality
# squares (no bounds — those are handled directly).
function _native_violation_part(m::IPMModel, x::AbstractVector)
    ε = 1e-2
    v = 0.0
    nji, nje = m.n_jump_ineq, m.n_jump_eq
    for (k, c) in enumerate(m.nlineqs)
        (nji > 0 ? k > 1 : true) || continue
        for ci in _vec(c(x))
            ci + ε > 0 && (v += (ci + ε)^2 / 2)
        end
    end
    for c in m.nleqs[(nje > 0 ? 2 : 1):end]
        v += sum(abs2, _vec(c(x))) / 2
    end
    return v
end
