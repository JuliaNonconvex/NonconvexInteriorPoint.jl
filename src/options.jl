"""
    IPMEvaluationSelection

Selects the evaluation strategy of individual constraints through NonconvexCore's
label mechanism (`flags`): a constraint whose `flags` intersect `matrixfree` is
evaluated matrix-free (implicit products, iterative solves, no dense assembly),
and a constraint whose `flags` intersect `trace` is evaluated with randomized
trace estimation (implies matrix-free). The `_all` flags select every
constraint. An explicit label selection dominates the `_all` flags; `trace`
dominates `matrixfree`; unselected constraints use the dense strategy.
"""
struct IPMEvaluationSelection
    matrixfree::Set{Symbol}
    trace::Set{Symbol}
    matrixfree_all::Bool
    trace_all::Bool
end
function IPMEvaluationSelection(;
    matrixfree::Union{Symbol,AbstractVector{Symbol}}=Symbol[],
    trace::Union{Symbol,AbstractVector{Symbol}}=Symbol[],
    matrixfree_all::Bool=false,
    trace_all::Bool=false,
)
    mf = matrixfree isa Symbol ? Set{Symbol}([matrixfree]) : Set{Symbol}(matrixfree)
    tr = trace isa Symbol ? Set{Symbol}([trace]) : Set{Symbol}(trace)
    return IPMEvaluationSelection(mf, tr, matrixfree_all, trace_all)
end

"""
    BarrierSelection

Selects which constraints enter the barrier of [`IPMAlg`](@ref) and with
which barrier type (mirroring the constraint-relaxation selections of the
ecosystem):

- `ineq = false`: inequality constraints are excluded from the barrier and
  enforced as hard slacked constraints (`c_I(x) + s = 0`) — the per-category
  counterpart of `slacks = true`;
- `box = false`: variable bounds are excluded from the barrier (and not
  enforced);
- `sd = false`: semidefinite/set constraints are excluded from the barrier
  (and not enforced — use with care);
- `labels`: constraints carrying any of these labels are excluded from the
  barrier (inequalities become slacked, sets are not enforced);
- `ineq_type`: `:log` (`-ln t`), `:inv` (`1/t`) or `:quad` (`1/(2t²)`)
  scalar barrier on the inequality slacks;
- `box_type`: the same for the bound distances;
- `sd_type`: `:logdet` (`-log det X`, the default), `:inverse` (`tr(X⁻¹)`)
  or `:inverse_square` (`tr(X⁻²)/2`) for the semidefinite barriers
  (implemented by NonconvexSemidefinite for the dense oracle; the
  matrix-free and trace oracles support `:logdet`).
"""
struct BarrierSelection
    ineq::Bool
    box::Bool
    sd::Bool
    labels::Set{Symbol}
    ineq_type::Symbol
    box_type::Symbol
    sd_type::Symbol
end
function BarrierSelection(;
    ineq::Bool=true,
    box::Bool=true,
    sd::Bool=true,
    labels::Union{Symbol,AbstractVector{Symbol}}=Symbol[],
    ineq_type::Symbol=:log,
    box_type::Symbol=:log,
    sd_type::Symbol=:logdet,
)
    ineq_type in (:log, :inv, :quad) ||
        throw(ArgumentError("ineq_type must be :log, :inv or :quad"))
    box_type in (:log, :inv, :quad) ||
        throw(ArgumentError("box_type must be :log, :inv or :quad"))
    sd_type in (:logdet, :inverse, :inverse_square) ||
        throw(ArgumentError("sd_type must be :logdet, :inverse or :inverse_square"))
    lab = labels isa Symbol ? Set{Symbol}([labels]) : Set{Symbol}(labels)
    return BarrierSelection(ineq, box, sd, lab, ineq_type, box_type, sd_type)
end
"""
    IPMOptions

Options for [`IPMAlg`](@ref), constructed with keyword arguments
(`IPMOptions(; tol = 1e-8, mu_init = 0.1, ...)`).

## Keywords

- `tol::Real = 1e-8`: primal and (scaled) dual feasibility tolerance.
- `mu_init::Real = 0.1`: initial barrier parameter.
- `mu_min::Real = 1e-11`: terminal barrier parameter.
- `mu_linear_decrease_factor::Real = 0.2`: linear `μ` decrease factor.
- `mu_superlinear_decrease_power::Real = 1.5`: superlinear `μ` decrease power.
- `theta_mu::Real = 1.5`: switch to the superlinear `μ` update when the
  barrier KKT error drops below `theta_mu * μ`.
- `max_iter::Int = 3000`: maximum number of iterations.
- `linear_solver::Symbol = :auto`: `:auto` (Krylov if any matrix-free or
  trace-estimated constraint is present, dense otherwise), `:direct`
  (assembled symmetric KKT, requires no matrix-free/trace constraints) or
  `:krylov` (preconditioned MINRES on the augmented KKT).
- `slacks::Bool = false`: when `true`, inequality constraints are rewritten
  with explicit slack variables (`c_I(x) + s = 0`, `s ≥ 0`, slack barrier
  `-μ Σ ln sᵢ`), matching the Ipopt-style slacked formulation; when `false`
  (the default), inequalities and PSD constraints enter the log barrier
  directly, their gradients are assembled from adjoints (no Jacobians are
  materialized), and only the equality constraints contribute Jacobians to
  the KKT system.
- `barrier::BarrierSelection = BarrierSelection()`: which constraints enter
  the barrier and with which barrier type (see [`BarrierSelection`](@ref)).
- `inner::Union{Nothing,NonconvexCore.AbstractOptimizer} = nothing`: an
  optional inner optimizer used to solve each barrier subproblem (any
  NonconvexCore optimizer); `nothing` uses the built-in Newton solver.
- `inner_options = nothing`: options for the inner optimizer.
- `inner_max_iter::Int = 50`: maximum Newton iterations per barrier
  subproblem for the built-in inner solver.
- `evaluation::IPMEvaluationSelection = IPMEvaluationSelection()`: per-label
  evaluation strategy selection.
- `krylov_tol::Real = 1e-8`: relative tolerance of the Krylov KKT solve.
- `krylov_maxiter::Int = 200`: maximum Krylov iterations per KKT solve.
- `trace_nprobes::Int = 10`: number of Hutchinson/SLQ probes.
- `trace_lanczos_deg::Int = 30`: Lanczos degree in stochastic Lanczos
  quadrature.
- `trace_method::Symbol = :slq`: log-determinant estimation method for
  `:trace`-labeled PSD constraints — one of `:slq` (stochastic Lanczos
  quadrature), `:chebyshev`, `:hutchpp`, `:rational`, `:taylor`,
  `:integration`, `:lowrank` (see `NonconvexSemidefinite.logdet_estimate`).
- `trace_cg_tol::Real = 1e-6`: conjugate-gradient tolerance for `X⁻¹v` solves.
- `trace_final_exact::Bool = false`: switch trace-estimated constraints to
  the exact oracle for the final barrier iterations.
- `linear_constraints::Bool = true`: detect and handle linear constraints
  natively with cached constant Jacobians (structural `LinearFunction`
  detection plus a Jacobian-constancy probe).
- `bound_relax_factor::Real = 1e-8`: relative relaxation of the variable
  bounds at initialization.
- `relaxed_labels = nothing`: `Symbol`, vector of `Symbol`s, or `nothing`;
  constraints carrying any of these labels are moved from hard constraints
  into a quadratic penalty (relaxation).
- `relaxed_penalty_rho::Real = 1e4`: penalty weight for relaxed constraints.
- `tau::Real = 0.995`: fraction-to-the-boundary parameter.
- `s_min::Real = 1e-8`: minimum slack/bound distance kept by
  fraction-to-the-boundary.
- `max_time::Real = Inf`: time limit in seconds.
- `feasibility_max_iter::Int = 200`: maximum regularized-Newton iterations
  of the pre-barrier feasibility stage, which runs when the initial point
  violates inequalities, bounds or equalities (set to `0` to disable it —
  the smoothly extended barrier then restores feasibility on its own).
- `verbose::Bool = false`: print iteration information.
- `rng::Random.AbstractRNG = Random.default_rng()`: random number generator
  for the trace estimators.
"""
struct IPMOptions
    tol::Float64
    mu_init::Float64
    mu_min::Float64
    mu_linear_decrease_factor::Float64
    mu_superlinear_decrease_power::Float64
    theta_mu::Float64
    max_iter::Int
    linear_solver::Symbol
    slacks::Bool
    barrier::BarrierSelection
    inner::Union{Nothing,AbstractOptimizer}
    inner_options::Any
    inner_max_iter::Int
    evaluation::IPMEvaluationSelection
    krylov_tol::Float64
    krylov_maxiter::Int
    trace_nprobes::Int
    trace_lanczos_deg::Int
    trace_method::Symbol
    trace_cg_tol::Float64
    trace_final_exact::Bool
    linear_constraints::Bool
    bound_relax_factor::Float64
    relaxed_labels::Union{Nothing,Symbol,AbstractVector{Symbol}}
    relaxed_penalty_rho::Float64
    tau::Float64
    s_min::Float64
    max_time::Float64
    feasibility_max_iter::Int
    verbose::Bool
    rng::Random.AbstractRNG
end
function IPMOptions(;
    tol::Real=1e-8,
    mu_init::Real=0.1,
    mu_min::Real=1e-11,
    mu_linear_decrease_factor::Real=0.2,
    mu_superlinear_decrease_power::Real=1.5,
    theta_mu::Real=1.5,
    max_iter::Integer=3000,
    linear_solver::Symbol=:auto,
    slacks::Bool=false,
    barrier::BarrierSelection=BarrierSelection(),
    inner::Union{Nothing,AbstractOptimizer}=nothing,
    inner_options=nothing,
    inner_max_iter::Integer=50,
    evaluation::IPMEvaluationSelection=IPMEvaluationSelection(),
    krylov_tol::Real=1e-8,
    krylov_maxiter::Integer=200,
    trace_nprobes::Integer=10,
    trace_lanczos_deg::Integer=30,
    trace_method::Symbol=:slq,
    trace_cg_tol::Real=1e-6,
    trace_final_exact::Bool=false,
    linear_constraints::Bool=true,
    bound_relax_factor::Real=1e-8,
    relaxed_labels=nothing,
    relaxed_penalty_rho::Real=1e4,
    tau::Real=0.995,
    s_min::Real=1e-8,
    max_time::Real=Inf,
    feasibility_max_iter::Integer=200,
    verbose::Bool=false,
    rng::Random.AbstractRNG=Random.default_rng(),
)
    linear_solver in (:auto, :direct, :krylov) ||
        throw(ArgumentError("linear_solver must be :auto, :direct or :krylov"))
    max_iter > 0 || throw(ArgumentError("max_iter must be positive"))
    return IPMOptions(
        Float64(tol),
        Float64(mu_init),
        Float64(mu_min),
        Float64(mu_linear_decrease_factor),
        Float64(mu_superlinear_decrease_power),
        Float64(theta_mu),
        Int(max_iter),
        linear_solver,
        slacks,
        barrier,
        inner,
        inner_options,
        Int(inner_max_iter),
        evaluation,
        Float64(krylov_tol),
        Int(krylov_maxiter),
        Int(trace_nprobes),
        Int(trace_lanczos_deg),
        trace_method,
        Float64(trace_cg_tol),
        trace_final_exact,
        linear_constraints,
        Float64(bound_relax_factor),
        relaxed_labels,
        Float64(relaxed_penalty_rho),
        Float64(tau),
        Float64(s_min),
        Float64(max_time),
        Int(feasibility_max_iter),
        verbose,
        rng,
    )
end

# Labels of constraints that are relaxed (moved to a penalty) rather than
# enforced.
function _relaxed_set(opts::IPMOptions)
    l = opts.relaxed_labels
    l === nothing && return Set{Symbol}()
    l isa Symbol && return Set{Symbol}([l])
    return Set{Symbol}(l)
end
