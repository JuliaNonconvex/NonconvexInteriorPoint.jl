# Barrier-oracle protocol for set constraints.
#
# A set constraint `c(x) ∈ S` contributes a barrier term `φ_S(c(x))` to the
# barrier objective. The protocol is defined at the constraint level with an
# evaluation mode (`Val{:dense}`, `Val{:matrixfree}` or `Val{:trace}`) and the
# options (threaded through for the randomized oracles), so external packages
# (NonconvexSemidefinite.jl) can supply specialized semidefinite oracles
# through a package extension without overwriting core methods.
#
# In the current model conversion only `PSDCone` (and `Reals`) survive as
# barrier set constraints; order sets become slacked inequalities. The
# callable `c` may return the set value as a matrix or a vector (for
# `LinearFunction`-backed constraints, of length `dim^2`).

"""
    barrier_value_c(c, set, x, mode, opts)

Barrier value `φ_S(c(x))` of the set constraint. Returns `-Inf` for a
violated constraint (strict interiority is required). The only barrier set
constraints in the current model conversion are `PSDCone`s (handled below);
other sets are converted to classical constraints.
"""
function barrier_value_c(c, set, x, ::Val{mode}, opts) where {mode}
    return barrier_value(set, _set_value(c, set, x))
end

"""
    barrier_value(set, v)

Barrier value of the set at the (matrix) value `v`. Fallback: unsupported.
"""
function barrier_value(set::AbstractSet, v)
    return throw(
        ArgumentError(
            "no barrier oracle for $(typeof(set)); " *
            "PSDCone matrix-free/trace modes require NonconvexSemidefinite to be loaded",
        ),
    )
end
barrier_value(::Reals, v) = 0.0

# The set value as a matrix (PSD cones accept vector storage).
function _set_value(c, set::PSDCone, x)
    v = c(x)
    return v isa AbstractMatrix ? Matrix(v) : reshape(vec(v), set.dim, set.dim)
end
function _set_value(c::LinearFunction, set::PSDCone, x)
    return reshape(vec(c(x)), set.dim, set.dim)
end
_set_value(c, set, x) = c(x)

"""
    barrier_grad_x!(g, c, set, x, mode, opts)

Gradient `∇_x φ_S(c(x))` of the barrier term.
"""
function barrier_grad_x!(g, c, set, x, ::Val{mode}, opts) where {mode}
    _, gr = value_gradient(z -> _finite_barrier(c, set, z), x)
    gr === nothing ? fill!(g, 0.0) : copyto!(g, gr)
    return g
end

"""
    barrier_hess_x!(H, c, set, x, mode, opts)

Hessian `∇²_xx φ_S(c(x))` of the barrier term, assembled into `H` (which is
added to the barrier-objective Hessian in the dense path).
"""
function barrier_hess_x!(H, c, set, x, ::Val{mode}, opts) where {mode}
    Hh = value_hessian(z -> _finite_barrier(c, set, z), x)
    copyto!(H, Hh)
    return H
end

"""
    barrier_hess_x_prod!(y, c, set, x, w, mode, opts)

Hessian-vector product `∇²_xx φ_S(c(x)) w`.
"""
function barrier_hess_x_prod!(y, c, set, x, w, ::Val{mode}, opts) where {mode}
    h = hvp(z -> _finite_barrier(c, set, z), x, w)
    copyto!(y, h)
    return y
end

# The dense barrier value, forced finite (so AD sees a smooth function).
_finite_barrier(c, set, x) = barrier_value(set, _set_value(c, set, x))

# ---------------------------------------------------------------------------
# PSDCone oracles: dispatch hooks with generic fallbacks, so the
# NonconvexSemidefinite extension can specialize per mode without
# overwriting core methods.
# ---------------------------------------------------------------------------

function barrier_value_c(c, set::PSDCone, x, ::Val{mode}, opts) where {mode}
    return _psd_barrier(c, set, x, Val(mode), opts)
end
function barrier_grad_x!(g, c, set::PSDCone, x, ::Val{mode}, opts) where {mode}
    return _psd_barrier_grad!(g, c, set, x, Val(mode), opts)
end
function barrier_hess_x!(H, c, set::PSDCone, x, ::Val{mode}, opts) where {mode}
    return _psd_barrier_hess!(H, c, set, x, Val(mode), opts)
end
function barrier_hess_x_prod!(y, c, set::PSDCone, x, w, ::Val{mode}, opts) where {mode}
    return _psd_barrier_hessprod!(y, c, set, x, w, Val(mode), opts)
end

# Generic fallbacks: dense via a Cholesky factorization; the matrix-free and
# trace modes require the NonconvexSemidefinite oracles.
function _psd_barrier(c, set::PSDCone, x, ::Val{mode}, opts) where {mode}
    mode === :dense || throw(
        ArgumentError(
            "PSD barrier mode :$(mode) requires NonconvexSemidefinite to be loaded"
        ),
    )
    X = Symmetric(_set_value(c, set, x))
    F = cholesky(X; check=false)
    return issuccess(F) ? -logdet(F) : -Inf
end
function _psd_barrier_grad!(g, c, set::PSDCone, x, ::Val{mode}, opts) where {mode}
    mode === :dense || throw(
        ArgumentError(
            "PSD barrier mode :$(mode) requires NonconvexSemidefinite to be loaded"
        ),
    )
    _, gr = value_gradient(z -> _psd_barrier(c, set, z, Val(:dense), opts), x)
    gr === nothing ? fill!(g, 0.0) : copyto!(g, gr)
    return g
end
function _psd_barrier_hess!(H, c, set::PSDCone, x, ::Val{mode}, opts) where {mode}
    mode === :dense || throw(
        ArgumentError(
            "PSD barrier mode :$(mode) requires NonconvexSemidefinite to be loaded"
        ),
    )
    Hh = value_hessian(z -> _psd_barrier(c, set, z, Val(:dense), opts), x)
    copyto!(H, Hh)
    return H
end
function _psd_barrier_hessprod!(y, c, set::PSDCone, x, w, ::Val{mode}, opts) where {mode}
    mode === :dense || throw(
        ArgumentError(
            "PSD barrier mode :$(mode) requires NonconvexSemidefinite to be loaded"
        ),
    )
    h = hvp(z -> _psd_barrier(c, set, z, Val(:dense), opts), x, w)
    copyto!(y, h)
    return y
end

# Strict interiority check for set constraints (used at initialization).
function barrier_strictly_feasible(c, set::PSDCone, x)
    X = Symmetric(_set_value(c, set, x))
    return all(λ -> λ > 0, eigvals(X))
end
barrier_strictly_feasible(c, set, x) = isfinite(barrier_value(set, _set_value(c, set, x)))

# ---------------------------------------------------------------------------
# Type-aware semidefinite barrier oracles (BarrierSelection's `sd_type`):
# dispatch through hooks so the NonconvexSemidefinite extension can
# specialize per type without overwriting core methods. The core supports
# only `:logdet` (through the mode-aware oracles above).
# ---------------------------------------------------------------------------

function sd_barrier_value_c(c, set, x, ::Val{mode}, type::Symbol, opts) where {mode}
    return _sd_barrier(c, set, x, Val(mode), type, opts)
end
function sd_barrier_grad_x!(g, c, set, x, ::Val{mode}, type::Symbol, opts) where {mode}
    return _sd_barrier_grad!(g, c, set, x, Val(mode), type, opts)
end
function sd_barrier_hess_x!(H, c, set, x, ::Val{mode}, type::Symbol, opts) where {mode}
    return _sd_barrier_hess!(H, c, set, x, Val(mode), type, opts)
end
function sd_barrier_hess_x_prod!(
    y, c, set, x, w, ::Val{mode}, type::Symbol, opts
) where {mode}
    return _sd_barrier_hessprod!(y, c, set, x, w, Val(mode), type, opts)
end

function _sd_barrier(c, set::PSDCone, x, ::Val{mode}, type::Symbol, opts) where {mode}
    type === :logdet || throw(
        ArgumentError(
            "sd barrier type :$(type) requires NonconvexSemidefinite to be loaded"
        ),
    )
    return _psd_barrier(c, set, x, Val(mode), opts)
end
function _sd_barrier_grad!(
    g, c, set::PSDCone, x, ::Val{mode}, type::Symbol, opts
) where {mode}
    type === :logdet || throw(
        ArgumentError(
            "sd barrier type :$(type) requires NonconvexSemidefinite to be loaded"
        ),
    )
    return _psd_barrier_grad!(g, c, set, x, Val(mode), opts)
end
function _sd_barrier_hess!(
    H, c, set::PSDCone, x, ::Val{mode}, type::Symbol, opts
) where {mode}
    type === :logdet || throw(
        ArgumentError(
            "sd barrier type :$(type) requires NonconvexSemidefinite to be loaded"
        ),
    )
    return _psd_barrier_hess!(H, c, set, x, Val(mode), opts)
end
function _sd_barrier_hessprod!(
    y, c, set::PSDCone, x, w, ::Val{mode}, type::Symbol, opts
) where {mode}
    type === :logdet || throw(
        ArgumentError(
            "sd barrier type :$(type) requires NonconvexSemidefinite to be loaded"
        ),
    )
    return _psd_barrier_hessprod!(y, c, set, x, w, Val(mode), opts)
end
