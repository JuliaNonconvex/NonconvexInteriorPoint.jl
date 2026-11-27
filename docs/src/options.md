# Options

`IPMAlg` is configured with [`IPMOptions`](@ref), passed to `optimize` with
the `options` keyword:

```julia
r = optimize(m, IPMAlg(), x0; options = IPMOptions(tol = 1e-10, verbose = true))
```

## Convergence and barrier schedule

- `tol = 1e-8`: primal and (scaled) dual feasibility tolerance.
- `mu_init = 0.1`: initial barrier parameter `μ`.
- `mu_min = 1e-11`: terminal barrier parameter.
- `mu_linear_decrease_factor = 0.2`: linear `μ` decrease factor.
- `mu_superlinear_decrease_power = 1.5`: superlinear `μ` decrease power.
- `theta_mu = 1.5`: the superlinear `μ` update kicks in when the barrier KKT
  error drops below `theta_mu * μ`.
- `max_iter = 3000`: maximum number of outer (barrier) iterations.
- `max_time = Inf`: time limit in seconds.
- `feasibility_max_iter = 200`: maximum regularized-Newton iterations of
  the pre-barrier feasibility stage, run when the initial point violates
  inequalities, bounds or equalities (`0` disables it — the smoothly
  extended barrier then restores feasibility on its own).

A solution is accepted as converged when `μ` is at (or below) `mu_min` and
the primal feasibility and the scaled stationarity residual of the barrier
problem are below `tol` (see [Theory](theory.md) for the practical
convergence criterion).

## Barrier selection

- `barrier::BarrierSelection = BarrierSelection()`: which constraints enter
  the barrier and with which barrier type, see
  [Barrier selection](#barrier-selection-1).
- `slacks = false`: when `true`, inequalities are rewritten with explicit
  slack variables (`c_I(x) + s = 0`, `s ≥ 0`, slack barrier `-μ Σ ln sᵢ`),
  matching the Ipopt-style slacked formulation. When `false` (the default),
  inequalities and PSD constraints enter the barrier directly with gradients
  assembled from adjoints only (no Jacobians are materialized for them), and
  only the equality constraints contribute Jacobians to the KKT system.

## Inner optimizer

- `inner = nothing`: an optional inner optimizer used to solve each barrier
  subproblem (any `NonconvexCore` optimizer, e.g. `IpoptAlg()`); `nothing`
  uses the built-in Newton solver.
- `inner_options = nothing`: options for the inner optimizer.
- `inner_max_iter = 50`: maximum Newton iterations per barrier subproblem
  for the built-in inner solver.

The outer loop drives `μ → 0` and delegates each barrier problem to the
inner optimizer:

```julia
using NonconvexIpopt
r = optimize(
    m, IPMAlg(), x0;
    options = IPMOptions(
        inner = IpoptAlg(),
        inner_options = IpoptOptions(print_level = 0),
    ),
)
```

## Linear algebra

- `linear_solver = :auto`: `:auto` (Krylov if any matrix-free or
  trace-estimated constraint is present, dense otherwise), `:direct`
  (assembled symmetric KKT, requires no matrix-free/trace constraints), or
  `:krylov` (preconditioned MINRES on the augmented KKT).
- `krylov_tol = 1e-8`: relative tolerance of the Krylov KKT solve.
- `krylov_maxiter = 200`: maximum Krylov iterations per KKT solve.

## Constraint handling

- `linear_constraints = true`: detect and handle linear constraints natively
  with cached constant Jacobians (structural `LinearFunction` detection plus
  a Jacobian-constancy probe).
- `bound_relax_factor = 1e-8`: relative relaxation of the variable bounds at
  initialization.
- `relaxed_labels = nothing`: `Symbol`, vector of `Symbol`s, or `nothing`;
  constraints carrying any of these labels are moved from hard constraints
  into a quadratic penalty (relaxation).
- `relaxed_penalty_rho = 1e4`: penalty weight for relaxed constraints.
- `evaluation::IPMEvaluationSelection = IPMEvaluationSelection()`: per-label
  evaluation strategy selection, see
  [Matrix-free and trace-estimated evaluation](matrixfree.md).

## Line search

- `tau = 0.995`: fraction-to-the-boundary parameter.
- `s_min = 1e-8`: minimum slack/bound distance kept by
  fraction-to-the-boundary (slacked formulation).

## Misc

- `verbose = false`: print iteration information.
- `rng = Random.default_rng()`: random number generator for the trace
  estimators.

## Evaluation selection

[`IPMEvaluationSelection`](@ref) selects the evaluation strategy of
individual constraints through NonconvexCore's label mechanism (`flags`):

- `matrixfree::Union{Symbol,Vector{Symbol}} = Symbol[]`: constraints whose
  `flags` intersect this set are evaluated matrix-free (implicit products,
  iterative solves, no dense assembly);
- `trace::Union{Symbol,Vector{Symbol}} = Symbol[]`: constraints whose
  `flags` intersect this set are evaluated with randomized trace estimation
  (implies matrix-free);
- `matrixfree_all::Bool = false`, `trace_all::Bool = false`: select every
  constraint.

An explicit label selection dominates the `_all` flags; `trace` dominates
`matrixfree`; unselected constraints use the dense strategy.

```julia
add_set_constraint!(m, x -> X(x), PSDCone(n); flags = Set([:big_sparse]))
options = IPMOptions(evaluation = IPMEvaluationSelection(trace = [:big_sparse]))
```

## Barrier selection

[`BarrierSelection`](@ref) selects which constraints enter the barrier of
`IPMAlg` and with which barrier type (mirroring the constraint-relaxation
selections of the ecosystem):

- `ineq::Bool = true`: `false` excludes inequality constraints from the
  barrier and enforces them as hard slacked constraints (`c_I(x) + s = 0`)
  — the per-category counterpart of `slacks = true`;
- `box::Bool = true`: `false` excludes variable bounds from the barrier
  (and does not enforce them);
- `sd::Bool = true`: `false` excludes semidefinite/set constraints from the
  barrier (and does not enforce them — use with care);
- `labels::Union{Symbol,Vector{Symbol}} = Symbol[]`: constraints carrying
  any of these labels are excluded from the barrier (inequalities become
  slacked, sets are not enforced);
- `ineq_type::Symbol = :log`: `:log` (`-ln t`), `:inv` (`1/t`) or `:quad`
  (`1/(2t²)`) scalar barrier on the inequality slacks;
- `box_type::Symbol = :log`: the same for the bound distances;
- `sd_type::Symbol = :logdet`: `:logdet` (`-log det X`, the default),
  `:inverse` (`tr(X⁻¹)`) or `:inverse_square` (`tr(X⁻²)/2`) for the
  semidefinite barriers (implemented by NonconvexSemidefinite for the dense
  oracle; the matrix-free and trace oracles support `:logdet`).

```julia
# Keep the box bounds out of the barrier; use the inverse semidefinite
# barrier and exclude :hard-labeled constraints from the barrier.
options = IPMOptions(
    barrier = BarrierSelection(
        box = false, sd_type = :inverse, labels = [:hard],
    ),
)
```
