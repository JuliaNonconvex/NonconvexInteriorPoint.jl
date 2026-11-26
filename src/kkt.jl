# KKT system for the primal barrier form.
#
# Inequalities (including box bounds) enter the barrier objective
#
#   φ_μ(x) = f(x) - μ Σ ln sᵢ - μ Σ ln(xᵢ - xlᵢ) - μ Σ ln(xuᵢ - xᵢ) + Σ φ_K(c_K(x)),
#
# with the slacks derived exactly, sᵢ = -cIᵢ(x) (so the duals are λIᵢ = μ/sᵢ
# and the iterates stay on the central path by construction). Only the
# equality constraints remain as hard constraints, giving the symmetric
# augmented system in (dx, dλE):
#
#   [ H_φ + δw I    J_E' ] [dx ]   [ -g ]
#   [ J_E         -δc I  ] [dλE] = [ -cE ]
#
# with g = ∇φ_μ, H_φ = ∇²φ_μ, and desired inertia (n, 0, 0).

# Solver state: primal iterate and evaluation buffers.
mutable struct IPMState
    n::Int
    mE::Int
    mI::Int
    x::Vector{Float64}
    lamE::Vector{Float64} # equality multipliers
    gf::Vector{Float64} # ∇φ_μ (barrier gradient)
    cE::Vector{Float64}
    cI::Vector{Float64}
    J::Matrix{Float64} # dense equality Jacobian (direct mode only)
    f::Float64
    barvals::Vector{Float64}
    setg::Vector{Vector{Float64}}
    mu::Float64
    # buffers
    K::Matrix{Float64}
    rhs::Vector{Float64}
    sol::Vector{Float64}
    vals::Vector{Float64}
    V::Matrix{Float64}
end

function IPMState(m::IPMModel, x::Vector{Float64}, mu::Float64)
    n = m.n
    mE = NonconvexInteriorPoint.mE(m)
    mI = NonconvexInteriorPoint.mI(m)
    N = n + mE
    st = IPMState(
        n,
        mE,
        mI,
        copy(x),
        zeros(mE),
        zeros(n),
        zeros(mE),
        zeros(mI),
        zeros(mE, n),
        0.0,
        fill(0.0, length(m.setcons)),
        [zeros(n) for _ in m.setcons],
        mu,
        zeros(N, N),
        zeros(N),
        zeros(N),
        zeros(N),
        zeros(N, N),
    )
    return st
end

# Slack values at x (exact): sᵢ = -cIᵢ(x), box distances. (Called outside AD.)
function _slacks(m::IPMModel, x::AbstractVector)
    cI = zeros(NonconvexInteriorPoint.mI(m))
    cE = zeros(NonconvexInteriorPoint.mE(m))
    eval_nonlinear_cons!(cE, cI, m, x)
    eval_linear_cons!(cE, cI, m, x)
    return cE, cI
end

# Smooth C¹ extension of the scalar log barrier -ln(t) beyond t ≤ ε
# (linear extension with slope -1/ε, so the curvature stays bounded for
# deeply infeasible starts — the extension acts as a strong feasibility
# penalty whose strength μ/ε shrinks with μ).
function _log_barrier_ext(t::Real, ε::Real)
    t > ε && return -log(t)
    return -log(ε) - (t - ε) / ε
end

# Scalar barrier functions on a positive slack t (with the μ scaling):
#   :log -> -μ ln t,  :inv -> μ/t,  :quad -> μ/(2t²)
function _scalar_barrier(t::Real, mu::Real, type::Symbol)
    if type === :log
        -mu * log(t)
    elseif type === :inv
        mu / t
    elseif type === :quad
        mu / (2 * t^2)
    else
        throw(ArgumentError("unknown scalar barrier type :$(type)"))
    end
end
# Their derivatives φ'(t) (the adjoint weight for the gradient is -φ'(t)).
function _scalar_barrier_deriv(t::Real, mu::Real, type::Symbol)
    if type === :log
        -mu / t
    elseif type === :inv
        -mu / t^2
    elseif type === :quad
        -mu / t^3
    else
        throw(ArgumentError("unknown scalar barrier type :$(type)"))
    end
end

# Barrier objective φ_μ(x). Non-mutating (called inside AD). On the feasible
# side the barrier is strict (returns Inf on violation, so the line search
# keeps the iterates feasible); on the infeasible side a smooth C¹ extension
# pulls the iterate back to feasibility. PSD constraints always require
# strict interiority. The [`BarrierSelection`](@ref) decides which
# constraints are barrierified and with which type.
"""
    barrier_objective(m::IPMModel, x, mu, opts::IPMOptions)

The barrier objective `φ_μ(x) = f(x) + μ·(scalar and set barrier terms)`
of [`IPMAlg`](@ref): inequalities and box bounds contribute `μ`-scaled
scalar barriers, set constraints (PSD cones) their `sd_type` barrier.
Returns `Inf` on a violated constraint when the iterate is in the feasible
regime (so line searches reject infeasible points); infeasible starts use a
smooth C¹ extension of the log barrier.
"""
function barrier_objective(m::IPMModel, x::AbstractVector, mu::Float64, opts::IPMOptions)
    φ = _scalar_barrier_objective(m, x, mu, opts)
    isfinite(φ) || return φ
    sel = opts.barrier
    for (k, (c, set, mode)) in enumerate(m.setcons)
        bv = sd_barrier_value_c(
            c, set, x, Val(_eff_mode(mode, mu, opts)), sel.sd_type, opts
        )
        isfinite(bv) || return Inf
        φ += mu * bv
    end
    return φ
end

# The scalar part of the barrier objective (objective, relaxed penalties,
# inequality and bound barriers — no set barriers). Non-mutating and
# AD-differentiable: used for the Hessian(-vector) products of the KKT
# operators, which add the set-barrier Hessian products through the
# analytic oracle hooks instead of differentiating the (possibly
# matrix-free or trace-estimated) set-barrier values.
function _scalar_barrier_objective(
    m::IPMModel, x::AbstractVector, mu::Float64, opts::IPMOptions
)
    f = _aug_objective(m, x)
    isfinite(f) || return Inf
    sel = opts.barrier
    φ = f
    ε = 1e-6
    # inequality values (masked: only the barriered ones); vcat is
    # differentiable (no mutation inside the AD closure)
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
    if m.strict[] || feasible
        # strict barriers
        for v in ineq_vals
            v < 0 || return Inf
            φ += _scalar_barrier(-v, mu, sel.ineq_type)
        end
        if sel.box
            @inbounds for i in 1:m.n
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
        # smooth extensions pull the iterate back to feasibility
        for v in ineq_vals
            φ += mu * _log_barrier_ext(-v, ε)
        end
        if sel.box
            @inbounds for i in 1:m.n
                if isfinite(m.xl[i])
                    φ += mu * _log_barrier_ext(x[i] - m.xl[i], ε)
                end
                if isfinite(m.xu[i])
                    φ += mu * _log_barrier_ext(m.xu[i] - x[i], ε)
                end
            end
        end
    end
    return φ
end

# Exact-penalty merit for the line search: the barrier objective plus an
# ℓ₁ penalty on the equality-constraint violation (so the Newton direction
# is a descent direction even when the objective is locally flat).
function penalty_merit(m::IPMModel, st::IPMState, x::AbstractVector, opts::IPMOptions)
    φ = barrier_objective(m, x, st.mu, opts)
    isfinite(φ) || return Inf
    cE = zeros(st.mE)
    cI = zeros(st.mI)
    eval_nonlinear_cons!(cE, cI, m, x)
    eval_linear_cons!(cE, cI, m, x)
    ν = max(1.0, maximum(abs, st.lamE; init=0.0) + 10.0)
    return φ + ν * sum(abs, cE)
end

function _penalty_closure(m::IPMModel, st::IPMState, opts::IPMOptions)
    return x -> penalty_merit(m, st, x, opts)
end

function _barrier_closure(m::IPMModel, mu::Float64, opts::IPMOptions)
    return x -> barrier_objective(m, x, mu, opts)
end

"""
    barrier_objective_ext(m, x, mu, opts)

Smoothly-extended variant of [`barrier_objective`](@ref) for custom inner
optimizers that cannot handle `Inf` values: feasible constraints get the
strict log barrier, near-boundary values the C¹ linear extension, and
violated values a hard quadratic penalty (so the inner solve cannot escape
to infeasibility). PSD violations are still rejected with `Inf`.
"""
function barrier_objective_ext(
    m::IPMModel, x::AbstractVector, mu::Float64, opts::IPMOptions
)
    f = _aug_objective(m, x)
    isfinite(f) || return Inf
    sel = opts.barrier
    φ = f
    ε = 1e-6
    PEN = 1e8
    for (k, c) in enumerate(m.nlineqs)
        m.nlineq_barrier[k] || continue
        for v in _vec(c(x))
            t = -v
            if t > ε
                φ -= mu * log(t)
            elseif t > 0
                φ += mu * (-log(ε) - (t - ε) / ε)
            else
                φ += PEN * t^2
            end
        end
    end
    if m.linAineq !== nothing && any(m.lin_ineq_barrier)
        for v in (m.linAineq * x .- m.linbineq)[m.lin_ineq_barrier]
            t = -v
            if t > ε
                φ -= mu * log(t)
            elseif t > 0
                φ += mu * (-log(ε) - (t - ε) / ε)
            else
                φ += PEN * t^2
            end
        end
    end
    if sel.box
        @inbounds for i in 1:m.n
            if isfinite(m.xl[i])
                t = x[i] - m.xl[i]
                if t > ε
                    φ -= mu * log(t)
                elseif t > 0
                    φ += mu * (-log(ε) - (t - ε) / ε)
                else
                    φ += PEN * t^2
                end
            end
            if isfinite(m.xu[i])
                t = m.xu[i] - x[i]
                if t > ε
                    φ -= mu * log(t)
                elseif t > 0
                    φ += mu * (-log(ε) - (t - ε) / ε)
                else
                    φ += PEN * t^2
                end
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

"""
    barrier_gradient!(g, m, x, mu, opts)

Gradient `∇φ_μ(x)` of the barrier objective, assembled from adjoints
(transposed-Jacobian-vector products): the vector-inequality barrier terms
contribute `J_c'(μ/s)`, which requires only the pullback/adjoint of each
constraint function — the Jacobian is never materialized. This keeps the
gradient path first-order-compatible (usable without any second-order
machinery) and label-aware (matrix-free-labeled constraints contribute
through their adjoint exactly as in the KKT operator).

At an infeasible point (before the strict-barrier mode is established) the
smooth C² extension has no closed-form adjoint assembly, so the scalar
closure is differentiated there instead (also reverse-mode).
"""
function barrier_gradient!(
    g::AbstractVector, m::IPMModel, x::AbstractVector, mu::Float64, opts::IPMOptions
)
    # objective part (pullback-based scalar differentiation)
    _, gobj = value_gradient(_aug_obj_closure(m), x)
    gobj === nothing ? fill!(g, 0.0) : copyto!(g, gobj)
    sel = opts.barrier

    # inequality values (masked: only the barriered ones)
    ineq_vals = vcat(
        [_vec(c(x)) for (k, c) in enumerate(m.nlineqs) if m.nlineq_barrier[k]]...,
        if (m.linAineq === nothing || !any(m.lin_ineq_barrier))
            Float64[]
        else
            (v=m.linAineq * x .- m.linbineq; v[m.lin_ineq_barrier])
        end,
    )
    ε = 1e-6
    feasible = all(v -> v <= -ε, ineq_vals)
    if sel.box
        @inbounds for i in 1:m.n
            isfinite(m.xl[i]) && (x[i] - m.xl[i] <= ε && (feasible = false))
            isfinite(m.xu[i]) && (m.xu[i] - x[i] <= ε && (feasible = false))
        end
    end
    if !(m.strict[] || feasible)
        # infeasible start: differentiate the extended barrier closure
        _, gext = value_gradient(x -> barrier_objective_ext(m, x, mu, opts), x)
        gext === nothing ? fill!(g, 0.0) : copyto!(g, gext)
        return g
    end

    # vector-inequality barrier terms: g += J_c'(w), w = -φ'(s), s = -c(x),
    # via the adjoint (pullback) of each constraint function (masked)
    for (k, c) in enumerate(m.nlineqs)
        m.nlineq_barrier[k] || continue
        v = _vec(c(x))
        w = -_scalar_barrier_deriv.(-v, mu, sel.ineq_type)
        jtprod!(g, c, x, w)
    end
    if m.linAineq !== nothing && any(m.lin_ineq_barrier)
        v = m.linAineq * x .- m.linbineq
        w = -_scalar_barrier_deriv.(-v, mu, sel.ineq_type)
        mask = m.lin_ineq_barrier
        mul!(g, m.linAineq[mask, :]', w[mask], 1.0, 1.0)
    end
    # bound barriers (diagonal, masked)
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
    # set barriers (mode-aware, type-aware, scaled by μ)
    for (k, (c, set, mode)) in enumerate(m.setcons)
        effm = _eff_mode(mode, mu, opts)
        grad = similar(g)
        sd_barrier_grad_x!(grad, c, set, x, Val(effm), sel.sd_type, opts)
        g .+= mu .* grad
    end
    return g
end

# Assemble the dense KKT matrix and right-hand side (primal barrier form).
function assemble_kkt_dense!(
    st::IPMState, m::IPMModel, opts::IPMOptions, δw::Float64, δc::Float64
)
    n, mE = st.n, st.mE
    N = n + mE
    fill!(st.K, 0.0)
    m.counter[] += 2 * (length(m.nleqs) + length(m.nlineqs) + length(m.setcons))
    H = try
        value_hessian(_barrier_closure(m, st.mu, opts), st.x)
    catch
        zeros(n, n)
    end
    K11 = view(st.K, 1:n, 1:n)
    copyto!(K11, H)
    @inbounds for i in 1:n
        K11[i, i] += δw
    end
    copyto!(view(st.K, 1:n, (n + 1):N), st.J')
    copyto!(view(st.K, (n + 1):N, 1:n), st.J)
    K22 = view(st.K, (n + 1):N, (n + 1):N)
    for i in 1:mE
        K22[i, i] = -δc
    end
    # stationarity residual: -(g + J_E' λE)
    r = view(st.rhs, 1:n)
    copyto!(r, st.gf)
    _jteq_λ!(r, m, st.x, st.lamE)
    r .*= -1.0
    st.rhs[(n + 1):N] .= .-st.cE
    return nothing
end

# Assemble the dense equality Jacobian (rows: nl eq; lin eq).
function assemble_jacobian!(st::IPMState, m::IPMModel)
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
    return st.J
end

# Solve the dense KKT system with the inertia-correction loop. Returns true
# on success (and fills st.sol with (dx, dλE)).
function solve_kkt_dense!(st::IPMState, m::IPMModel, opts::IPMOptions; δc0::Float64=1e-8)
    n, mE = st.n, st.mE
    δc = δc0
    δw = 0.0
    for trial in 1:12
        assemble_kkt_dense!(st, m, opts, δw, δc)
        E = eigen(Symmetric(st.K))
        copyto!(st.vals, E.values)
        npos = count(>(0.0), st.vals)
        nzero = count(v -> abs(v) < 1e-14, st.vals)
        if npos == n && nzero == 0
            copyto!(st.V, E.vectors)
            _eigen_solve!(st.sol, st.V, st.vals, st.rhs)
            return true
        end
        δc *= 10.0
    end
    δc = 1e-4
    δw = 1e-8
    for trial in 1:12
        assemble_kkt_dense!(st, m, opts, δw, δc)
        E = eigen(Symmetric(st.K))
        copyto!(st.vals, E.values)
        npos = count(>(0.0), st.vals)
        nzero = count(v -> abs(v) < 1e-14, st.vals)
        if npos == n && nzero == 0
            copyto!(st.V, E.vectors)
            _eigen_solve!(st.sol, st.V, st.vals, st.rhs)
            return true
        end
        δc *= 10.0
        δw = max(δw * 100.0, 1e-4)
    end
    return false
end

function _eigen_solve!(sol, V, vals, rhs)
    y = V' * rhs
    @inbounds for i in eachindex(y)
        y[i] = abs(vals[i]) < 1e-14 ? 0.0 : y[i] / vals[i]
    end
    mul!(sol, V, y)
    return sol
end

# ---------------------------------------------------------------------------
# Krylov (matrix-free) KKT operator
# ---------------------------------------------------------------------------

# Implicit KKT operator for preconditioned MINRES (primal barrier form).
mutable struct KKTLinearOp
    m::IPMModel
    st::IPMState
    opts::IPMOptions
    δw::Float64
    δc::Float64
    precond::Vector{Float64}
end

function KKTLinearOp(m::IPMModel, st::IPMState, opts::IPMOptions, δw::Float64, δc::Float64)
    n, mE = st.n, st.mE
    d = fill(max(δc, 1.0), n + mE)
    for i in 1:n
        d[i] = max(δw, 1.0)
    end
    return KKTLinearOp(m, st, opts, δw, δc, d)
end

Base.size(op::KKTLinearOp) = (op.st.n + op.st.mE, op.st.n + op.st.mE)
Base.size(op::KKTLinearOp, d::Int) = size(op)[d]
Base.eltype(::KKTLinearOp) = Float64

function LinearAlgebra.mul!(y, op::KKTLinearOp, u)
    m = op.m
    st = op.st
    n, mE = st.n, st.mE
    dx = view(u, 1:n)
    dλ = view(u, (n + 1):(n + mE))
    yx = view(y, 1:n)
    yλ = view(y, (n + 1):(n + mE))
    # (1,1): Hessian of the barrier objective times dx — the scalar part
    # (objective, relaxed penalties, inequality/bound barriers) through AD,
    # the set barriers through the analytic oracle Hessian products (their
    # matrix-free/trace-estimated values are not AD-differentiable)
    hvp!(yx, z -> _scalar_barrier_objective(m, z, st.mu, op.opts), st.x, dx)
    for (k, (c, set, mode)) in enumerate(m.setcons)
        effm = _eff_mode(mode, st.mu, op.opts)
        bar = similar(yx)
        sd_barrier_hess_x_prod!(
            bar, c, set, st.x, dx, Val(effm), op.opts.barrier.sd_type, op.opts
        )
        yx .+= st.mu .* bar
    end
    yx .+= op.δw .* dx
    # (1,2): J_E' dλ
    offset = 1
    for (k, c) in enumerate(m.nleqs)
        d = getdim(c)
        jtprod!(yx, c, st.x, view(dλ, offset:(offset + d - 1)))
        offset += d
    end
    if m.linAeq !== nothing
        mul!(yx, m.linAeq', view(dλ, offset:(offset + size(m.linAeq, 1) - 1)), 1.0, 1.0)
        offset += size(m.linAeq, 1)
    end
    # (2,1): J_E dx
    offset = 1
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
    # (2,2)
    @inbounds for i in 1:mE
        yλ[i] -= op.δc * dλ[i]
    end
    return y
end

# Diagonal preconditioner applied by division (MINRES SPD preconditioner).
struct KKTDiagPrecond
    d::Vector{Float64}
end
function LinearAlgebra.ldiv!(y::AbstractVector, P::KKTDiagPrecond, x::AbstractVector)
    @inbounds for i in eachindex(y, x, P.d)
        y[i] = x[i] / P.d[i]
    end
    return y
end
function LinearAlgebra.ldiv!(P::KKTDiagPrecond, x::AbstractVector)
    @inbounds for i in eachindex(x, P.d)
        x[i] /= P.d[i]
    end
    return x
end
Base.size(P::KKTDiagPrecond) = (length(P.d), length(P.d))
Base.size(P::KKTDiagPrecond, d::Int) = size(P)[d]

# Solve the KKT system with preconditioned MINRES, increasing the
# regularization when the solve stalls.
function solve_kkt_krylov!(st::IPMState, m::IPMModel, opts::IPMOptions)
    δc = 1e-8
    δw = 1e-10
    for trial in 1:6
        op = KKTLinearOp(m, st, opts, δw, δc)
        prec = KKTDiagPrecond(op.precond)
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
        if stats.solved
            st.sol .= sol
            return true
        end
        δc *= 100.0
        δw = max(δw, 1e-10 * δc)
    end
    return false
end

# Solution views: (dx, dλE) = (sol[1:n], sol[n+1:end]).
_dx(st::IPMState) = view(st.sol, 1:st.n)
_dlam(st::IPMState) = view(st.sol, (st.n + 1):(st.n + st.mE))
