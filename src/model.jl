# Flattening of a NonconvexCore VecModel into the interior-point form.
#
# Conversion rules:
# - classical `EqConstraint`/`IneqConstraint` become equality/slacked-inequality
#   constraints (the wrappers already return the violation `f(x) - rhs`),
# - `FunctionInSet` constraints are normalized: order sets (`LessThan`,
#   `GreaterThan`, `Interval`, `Nonnegatives`, `Nonpositives`), `EqualTo` and
#   `Zeros` become classical equalities/inequalities, `Simplex` becomes a
#   nonnegativity inequality plus a sum-to-one equality, `SetProduct`s are
#   split into their components, `Reals` is dropped, and `PSDCone` (plus the
#   legacy `SDConstraint`) remains as a barrier set constraint,
# - constraints with a `LinearFunction` (or a constant Jacobian, when the
#   `linear_constraints` probe is enabled) are collected into native sparse
#   linear blocks with cached Jacobians,
# - constraints whose labels are selected in `relaxed_labels` are moved into a
#   quadratic penalty instead of being enforced.

const _SetCons = Tuple{Any,AbstractSet,Symbol}

"""
    IPMModel

The flattened interior-point model: objective, box bounds, nonlinear
equalities and (per-constraint barrier-masked) inequalities, native linear
equality/inequality blocks with cached Jacobians, and barrier set
constraints (PSD cones). Built from a NonconvexCore `VecModel` by
`build_ipm_model`; the masks `nlineq_barrier`/`lin_ineq_barrier` record
which inequalities enter the barrier (`true`) and which are slacked
(`false`).
"""
mutable struct IPMModel
    n::Int
    xl::Vector{Float64}
    xu::Vector{Float64}
    # variables with xl == xu (fixed): they carry no bound barrier — a
    # zero-width log barrier is infinite everywhere — and are instead
    # pinned by identity rows appended to the linear equality block
    fixed_bounds::BitVector
    x0::Vector{Float64}
    # objective (callable, or nothing for feasibility problems)
    obj::Any
    # nonlinear (non-relaxed) constraints in violation form
    nleqs::Vector{Any}
    nlineqs::Vector{Any}
    nleq_modes::Vector{Symbol}
    nlineq_modes::Vector{Symbol}
    # per-inequality barrier selection (true = in the barrier, false =
    # slacked): per nonlinear constraint and per linear row
    nlineq_barrier::Vector{Bool}
    lin_ineq_barrier::Vector{Bool}
    # native linear constraints (constant Jacobians)
    linAeq::Union{Nothing,SparseMatrixCSC{Float64,Int}}
    linbeq::Union{Nothing,Vector{Float64}}
    linAineq::Union{Nothing,SparseMatrixCSC{Float64,Int}}
    linbineq::Union{Nothing,Vector{Float64}}
    # barrier set constraints (PSDCone): (callable, set, mode)
    setcons::Vector{_SetCons}
    # relaxed constraints (quadratic penalty)
    relaxed_eqs::Vector{Any}
    relaxed_ineqs::Vector{Any}
    rho::Float64
    # strict-barrier mode: set once the iterate is feasible w.r.t. the
    # inequalities/bounds, so the line search can never accept infeasible
    # points again (the smooth extension is only for infeasible starts)
    strict::Base.RefValue{Bool}
    counter::Base.RefValue{Int}
    # The shared MathOptInterface evaluator of a JuMP-converted model
    # (nothing otherwise): values, Jacobians and Lagrangian Hessians are
    # read from it directly — the evaluator cannot be traced by AD.
    jump::Union{Nothing,NonconvexCore.JuMPEvaluator}
end

mE(m::IPMModel) = n_eq_nl(m) + n_eq_lin(m)
mI(m::IPMModel) = n_ineq_nl(m) + n_ineq_lin(m)

isfixed_bound(m::IPMModel, i::Integer) = m.fixed_bounds[i]

# Numbers of equality/inequality rows (constraint functions may be
# vector-valued).
n_eq_nl(m::IPMModel) = sum(c -> getdim(c), m.nleqs; init=0)
n_eq_lin(m::IPMModel) = m.linAeq === nothing ? 0 : size(m.linAeq, 1)
n_ineq_nl(m::IPMModel) = sum(c -> getdim(c), m.nlineqs; init=0)
n_ineq_lin(m::IPMModel) = m.linAineq === nothing ? 0 : size(m.linAineq, 1)

# Per-row barrier masks (true = in the barrier, false = slacked).
function ineq_barrier_rows(m::IPMModel)
    rows = Bool[]
    for (k, c) in enumerate(m.nlineqs)
        append!(rows, fill(m.nlineq_barrier[k], getdim(c)))
    end
    append!(rows, m.lin_ineq_barrier)
    return rows
end

# Any inequality excluded from the barrier (→ the slacked/mixed path).
function any_ineq_slacked(m::IPMModel)
    any(!, m.nlineq_barrier) && return true
    any(!, m.lin_ineq_barrier) && return true
    return false
end

# Constraint labels (NonconvexCore's flag mechanism).
_flags(c) = hasfield(typeof(c), :flags) ? c.flags : Set{Symbol}()

# Per-constraint evaluation mode selected through the labels.
function _mode(flags::Set{Symbol}, opts::IPMOptions)
    sel = opts.evaluation
    if sel.trace_all || !isempty(intersect(flags, sel.trace))
        return :trace
    elseif sel.matrixfree_all || !isempty(intersect(flags, sel.matrixfree))
        return :matrixfree
    end
    return :dense
end

# A scalar bound as a vector of the right length.
_vec_b(v::Real, dim::Int) = fill(Float64(v), dim)
_vec_b(v::AbstractVector, ::Int) = vec(Float64.(v))

# Extract (A, b) from a LinearFunction-backed constraint, if any.
function _linear_parts(f)
    f isa LinearFunction || return nothing
    b = f.b === nothing ? zeros(eltype(f.A), size(f.A, 1)) : f.b
    return f.A, b
end

# Convert a `FunctionInSet` whose function is a `LinearFunction` into native
# linear blocks. Returns `nothing` if the set cannot be converted this way
# (i.e. PSD blocks), or a tuple of `(:eq, A, rhs)` / `(:ineq, A, ub)` pieces.
function _convert_linear_set(c::FunctionInSet)
    f = getfunction(c)
    (A, b) = _linear_parts(f)
    A === nothing && return nothing
    set = c.set
    out = Tuple{Symbol,Any,Any}[]
    _linear_set_rows!(out, A, b, set)
    return out
end

function _linear_set_rows!(out, A, b, set::LessThan)
    return push!(out, (:ineq, A, _vec_b(set.upper, size(A, 1)) .- b))
end
function _linear_set_rows!(out, A, b, set::GreaterThan)
    return push!(out, (:ineq, -A, b .- _vec_b(set.lower, size(A, 1))))
end
function _linear_set_rows!(out, A, b, set::Interval)
    push!(out, (:ineq, A, _vec_b(set.upper, size(A, 1)) .- b))
    return push!(out, (:ineq, -A, b .- _vec_b(set.lower, size(A, 1))))
end
function _linear_set_rows!(out, A, b, set::EqualTo)
    return push!(out, (:eq, A, _vec_b(set.value, size(A, 1)) .- b))
end
function _linear_set_rows!(out, A, b, ::Zeros)
    return push!(out, (:eq, A, .-b))
end
function _linear_set_rows!(out, A, b, ::Nonnegatives)
    return push!(out, (:ineq, -A, b))
end
function _linear_set_rows!(out, A, b, ::Nonpositives)
    return push!(out, (:ineq, A, .-b))
end
function _linear_set_rows!(out, A, b, set::Simplex)
    push!(out, (:ineq, -A, b))
    onesrow = ones(1, size(A, 1))
    return push!(out, (:eq, onesrow * A, [1.0 - sum(b)]))
end
function _linear_set_rows!(out, A, b, set::SetProduct)
    comps = collect(components(set))
    # Split the rows of A per component (vector sets take `dim` rows, PSD
    # blocks take `dim^2` rows).
    offset = 1
    for s in comps
        d = _set_rows(s)
        rows = collect(offset:(offset + d - 1))
        Ablock = A[rows, :]
        bblock = b[rows]
        _linear_set_rows!(out, Ablock, bblock, s)
        offset += d
    end
end
_linear_set_rows!(out, A, b, ::Reals) = nothing
function _linear_set_rows!(out, A, b, set::PSDCone)
    # Handled as a barrier set constraint; nothing to do here.
    return nothing
end
function _linear_set_rows!(out, A, b, set)
    return throw(ArgumentError("set $set of $(typeof(set)) is not supported by IPMAlg"))
end

# Number of rows a component set occupies in a vector value.
_set_rows(s::PSDCone) = s.dim^2
_set_rows(s) = getdim(s)

# Generic (nonlinear) conversion of a FunctionInSet into classical
# constraints + barrier set constraints. Returns a vector of entries:
# `(:eq, callable)`, `(:ineq, callable)`, `(:set, callable, set)`.
function _convert_set(c::FunctionInSet)
    out = Any[]
    _convert_set_rec!(out, c, getfunction(c), c.set)
    return out
end

function _convert_set_rec!(out, c, f, set::LessThan)
    ub = _vec_b(set.upper, set.dim)
    return push!(out, (:ineq, FunctionWrapper(x -> f(x) .- ub, set.dim)))
end
function _convert_set_rec!(out, c, f, set::GreaterThan)
    lb = _vec_b(set.lower, set.dim)
    return push!(out, (:ineq, FunctionWrapper(x -> lb .- f(x), set.dim)))
end
function _convert_set_rec!(out, c, f, set::Interval)
    ub = _vec_b(set.upper, set.dim)
    lb = _vec_b(set.lower, set.dim)
    push!(out, (:ineq, FunctionWrapper(x -> f(x) .- ub, set.dim)))
    return push!(out, (:ineq, FunctionWrapper(x -> lb .- f(x), set.dim)))
end
function _convert_set_rec!(out, c, f, set::EqualTo)
    return push!(
        out, (:eq, FunctionWrapper(x -> f(x) .- _vec_b(set.value, set.dim), set.dim))
    )
end
function _convert_set_rec!(out, c, f, set::Zeros)
    return push!(out, (:eq, FunctionWrapper(f, set.dim)))
end
function _convert_set_rec!(out, c, f, set::Nonnegatives)
    return push!(out, (:ineq, FunctionWrapper(x -> -f(x), set.dim)))
end
function _convert_set_rec!(out, c, f, set::Nonpositives)
    return push!(out, (:ineq, FunctionWrapper(f, set.dim)))
end
function _convert_set_rec!(out, c, f, set::Simplex)
    push!(out, (:ineq, FunctionWrapper(x -> -f(x), set.dim)))
    return push!(out, (:eq, FunctionWrapper(x -> [sum(f(x)) - 1.0], 1)))
end
function _convert_set_rec!(out, c, f, ::Reals)
    return nothing
end
function _convert_set_rec!(out, c, f, set::PSDCone)
    return push!(out, (:set, f, set))
end
function _convert_set_rec!(out, c, f, set::SetProduct)
    comps = components(set)
    for (j, s) in enumerate(comps)
        fj = x -> _component_value(f(x), j)
        _convert_set_rec!(out, c, fj, s)
    end
end
# The j-th component of a product-set value (tuple or per-component vector).
_component_value(v::Tuple, j) = v[j]
_component_value(v::AbstractVector, j) = v[j]
function _convert_set_rec!(out, c, f, set)
    return throw(ArgumentError("set $set of $(typeof(set)) is not supported by IPMAlg"))
end

# Jacobian-constancy probe: detect (numerically) linear constraints so their
# Jacobians can be cached once. The perturbation is deterministic (no RNG:
# the user's rng in IPMOptions is reserved for the trace estimators, whose
# probe draws must not be disturbed by model construction).
function _probe_linear(c, x0)
    v0, J0 = value_jacobian(c, x0)
    Δ = 1e-3 .* sin.(1.0:length(x0))
    v1, J1 = value_jacobian(c, x0 .+ Δ)
    b = _vec(v0) .- J0 * x0
    if isapprox(J0, J1; rtol=1e-6, atol=1e-8) &&
        isapprox(_vec(v1), J0 * (x0 .+ Δ) .+ b; rtol=1e-6, atol=1e-8)
        return J0, b
    end
    return nothing
end

"""
    build_ipm_model(m::VecModel, opts::IPMOptions)

Convert the `VecModel` `m` into the interior-point [`IPMModel`](@ref) form.
"""
function build_ipm_model(m::AbstractModel, opts::IPMOptions)
    # A JuMP-converted model keeps its constraints in one shared MOI
    # evaluator: the linear-constancy probe must not split its rows (the
    # evaluator already handles affine rows in closed form).
    jump = NonconvexCore.jump_evaluator(m)
    n = length(getmin(m))
    xl = Float64.(getmin(m))
    xu = Float64.(getmax(m))
    # Unbounded variables get pseudo-bounds (Ipopt-style bound push): every
    # variable then carries bound-barrier curvature, so the barrier Hessian
    # has no structural null directions.
    x0v = Float64.(getinit(m))
    @inbounds for i in 1:n
        isfinite(xl[i]) || (xl[i] = -1e4 * (1 + abs(x0v[i])))
        isfinite(xu[i]) || (xu[i] = 1e4 * (1 + abs(x0v[i])))
    end
    # Fixed variables (xl == xu): no bound barrier is possible on a
    # zero-width interval, so they are pinned by linear equality rows
    # instead. The mask keeps every barrier/strict-feasibility/bound-step
    # site from treating them as box barriers.
    fixed_bounds = falses(n)
    @inbounds for i in 1:n
        if xl[i] == xu[i]
            fixed_bounds[i] = true
        end
    end
    x0 = Float64.(getinit(m))
    counter = Ref(0)

    # Objective (augmented later with relaxed-constraint penalties).
    obj = getobjective(m)
    objf = obj

    relaxed = _relaxed_set(opts)

    nleqs = Any[]
    nlineqs = Any[]
    nleq_modes = Symbol[]
    nlineq_modes = Symbol[]
    nlineq_barrier = Bool[]
    setcons = _SetCons[]
    relaxed_eqs = Any[]
    relaxed_ineqs = Any[]
    sel = opts.barrier

    # Linear blocks collected as (A, rhs) pairs, converted to sparse at the end.
    lin_eq_parts = Tuple{AbstractMatrix,Vector{Float64}}[]
    lin_ineq_parts = Tuple{AbstractMatrix,Vector{Float64}}[]
    lin_ineq_barrier_parts = Bool[]

    # --- classical equality constraints ---
    for c in getfunctions(geteqconstraints(m))
        cf = c
        flags = _flags(c)
        if !isempty(intersect(flags, relaxed))
            push!(relaxed_eqs, cf)
            continue
        end
        f = getfunction(c)
        if f isa LinearFunction
            A, b = _linear_parts(f)
            push!(lin_eq_parts, (A, _vec(c.rhs) .- b))
            continue
        end
        if opts.linear_constraints && jump === nothing
            probed = _probe_linear(c, x0)
            if probed !== nothing
                J0, b = probed
                push!(lin_eq_parts, (J0, _vec(c.rhs) .- b))
                continue
            end
        end
        push!(nleqs, cf)
        push!(nleq_modes, _mode(flags, opts))
    end

    # --- classical inequality constraints ---
    for c in getfunctions(getineqconstraints(m))
        cf = c
        flags = _flags(c)
        if !isempty(intersect(flags, relaxed))
            push!(relaxed_ineqs, cf)
            continue
        end
        in_bar = (!opts.slacks) && sel.ineq && isempty(intersect(flags, sel.labels))
        f = getfunction(c)
        if f isa LinearFunction
            A, b = _linear_parts(f)
            push!(lin_ineq_parts, (A, _vec(c.rhs) .- b))
            push!(lin_ineq_barrier_parts, in_bar)
            continue
        end
        if opts.linear_constraints && jump === nothing
            probed = _probe_linear(c, x0)
            if probed !== nothing
                J0, b = probed
                push!(lin_ineq_parts, (J0, _vec(c.rhs) .- b))
                push!(lin_ineq_barrier_parts, in_bar)
                continue
            end
        end
        push!(nlineqs, cf)
        push!(nlineq_modes, _mode(flags, opts))
        push!(nlineq_barrier, in_bar)
    end

    # --- legacy semidefinite constraints ---
    for c in getfunctions(getsdconstraints(m))
        cf = c
        d = getdim(c)
        flags = _flags(c)
        mode = _mode(flags, opts)
        (sel.sd && isempty(intersect(flags, sel.labels))) || continue # sd exclusion: not enforced
        push!(setcons, (cf, PSDCone(d), mode))
    end

    # --- set constraints ---
    for c in getfunctions(getsetconstraints(m))
        flags = _flags(c)
        mode = _mode(flags, opts)
        if !isempty(intersect(flags, relaxed))
            throw(
                ArgumentError(
                    "relaxing set constraints is not supported by IPMAlg " *
                    "(constraint flags: $(collect(flags)))",
                ),
            )
        end
        # sd/barrier exclusion: the constraint is not enforced
        if !(sel.sd) || !isempty(intersect(flags, sel.labels))
            continue
        end
        in_bar = (!opts.slacks) && sel.ineq && isempty(intersect(flags, sel.labels))
        if c isa FunctionInSet && getfunction(c) isa LinearFunction
            pieces = _convert_linear_set(c)
            if pieces !== nothing
                for (kind, A, rhs) in pieces
                    if kind === :eq
                        push!(lin_eq_parts, (A, Vector(rhs)))
                    else
                        push!(lin_ineq_parts, (A, Vector(rhs)))
                        push!(lin_ineq_barrier_parts, in_bar)
                    end
                end
                continue
            end
        end
        for entry in _convert_set(c)
            kind = entry[1]
            if kind === :eq
                push!(nleqs, entry[2])
                push!(nleq_modes, mode)
            elseif kind === :ineq
                push!(nlineqs, entry[2])
                push!(nlineq_modes, mode)
                push!(nlineq_barrier, in_bar)
            else # :set
                push!(setcons, (entry[2], entry[3], mode))
            end
        end
    end

    # --- assemble sparse linear blocks ---
    # Fixed-variable pins (identity rows) close the equality block: they
    # must come last so cE, the Jacobian rows and the multipliers all agree.
    if any(fixed_bounds)
        for i in findall(fixed_bounds)
            row = spzeros(1, n)
            row[1, i] = 1.0
            push!(lin_eq_parts, (row, [xl[i]]))
        end
    end
    linAeq =
        isempty(lin_eq_parts) ? nothing : sparse(reduce(vcat, [p[1] for p in lin_eq_parts]))
    linbeq = isempty(lin_eq_parts) ? nothing : reduce(vcat, [p[2] for p in lin_eq_parts])
    linAineq = if isempty(lin_ineq_parts)
        nothing
    else
        sparse(reduce(vcat, [p[1] for p in lin_ineq_parts]))
    end
    linbineq =
        isempty(lin_ineq_parts) ? nothing : reduce(vcat, [p[2] for p in lin_ineq_parts])

    return IPMModel(
        n,
        xl,
        xu,
        fixed_bounds,
        x0,
        objf,
        nleqs,
        nlineqs,
        nleq_modes,
        nlineq_modes,
        nlineq_barrier,
        lin_ineq_barrier_parts,
        linAeq,
        linbeq,
        linAineq,
        linbineq,
        setcons,
        relaxed_eqs,
        relaxed_ineqs,
        Float64(opts.relaxed_penalty_rho),
        Ref(false),
        counter,
        jump,
    )
end

# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

# Augmented objective: user objective plus quadratic penalties for the
# relaxed constraints. NOTE: called inside AD (gradient/Hessian closures), so
# it must not mutate state. The user objective may be domain-limited (e.g.
# -log det on PSD matrices): a violation yields Inf so the line search
# rejects the trial point instead of throwing.
function _aug_objective(m::IPMModel, x::AbstractVector)
    f = try
        m.obj === nothing ? 0.0 : m.obj(x)::Real
    catch e
        (e isa DomainError || e isa PosDefException || e isa ArgumentError) &&
            return Inf
        rethrow()
    end
    isfinite(f) || return Inf
    for c in m.relaxed_eqs
        v = c(x)
        f += (m.rho / 2) * sum(abs2, v)
    end
    for c in m.relaxed_ineqs
        v = c(x)
        f += (m.rho / 2) * sum(abs2, max.(v, 0.0))
    end
    return f
end

_aug_obj_closure(m::IPMModel) = x -> _aug_objective(m, x)

# Lagrangian closure over the *nonlinear* enforced constraints (linear
# constraints have zero Hessians and are excluded).
function _lagrangian_closure(m::IPMModel, lamE::AbstractVector, lamI::AbstractVector)
    # λ layout: [λE_nl; λE_lin; λI_nl; λI_lin]
    return function (x)
        L = _aug_objective(m, x)
        for (k, c) in enumerate(m.nleqs)
            v = c(x)
            L += dot(lamE[k], v)
        end
        for (k, c) in enumerate(m.nlineqs)
            v = c(x)
            L += dot(lamI[k], v)
        end
        return L
    end
end

# A value as a vector (scalars become 1-element vectors).
_vec(v::Number) = [v]
_vec(v) = vec(v)

# Evaluate all enforced nonlinear constraints into preallocated vectors.
function eval_nonlinear_cons!(
    cE::AbstractVector, cI::AbstractVector, m::IPMModel, x::AbstractVector
)
    m.counter[] += length(m.nleqs) + length(m.nlineqs)
    offset = 1
    for c in m.nleqs
        v = _vec(c(x))
        d = length(v)
        copyto!(view(cE, offset:(offset + d - 1)), v)
        offset += d
    end
    offset = 1
    for c in m.nlineqs
        v = _vec(c(x))
        d = length(v)
        copyto!(view(cI, offset:(offset + d - 1)), v)
        offset += d
    end
    return nothing
end

# Values of the linear constraint blocks.
function eval_linear_cons!(cE, cI, m::IPMModel, x::AbstractVector)
    if m.linAeq !== nothing
        m.counter[] += 1
        r = (n_eq_nl(m) + 1):(n_eq_nl(m) + n_eq_lin(m))
        mul!(view(cE, r), m.linAeq, x)
        view(cE, r) .-= m.linbeq
    end
    if m.linAineq !== nothing
        m.counter[] += 1
        r = (n_ineq_nl(m) + 1):(n_ineq_nl(m) + n_ineq_lin(m))
        mul!(view(cI, r), m.linAineq, x)
        view(cI, r) .-= m.linbineq
    end
    return nothing
end
