# NonconvexInteriorPoint.jl — Design

A native interior-point solver for general (nonconvex) nonlinear programs in the
JuliaNonconvex ecosystem, built on NonconvexCore.jl. This document records the
algorithmic and architectural decisions and their final implementation status.
It incorporates the literature survey from
[issue #40](https://github.com/JuliaNonconvex/NonconvexSemidefinite.jl/issues/40)
and a review of the state of the art in the surrounding ecosystem.

---

## 1. Goals

1. **State-of-the-art barrier IPM** for smooth nonconvex NLPs, following the
   Ipopt → MadNLP → Uno lineage (regularized KKT systems, inertia control,
   adaptive barrier updates).
2. **Native linear constraints**: first-class, constant-Jacobian handling
   (Ipopt's `jac_c_constant` pattern).
3. **Set constraints** from NonconvexCore (including `PSDCone`) via a
   barrier-oracle protocol.
4. **Semidefinite code stays in NonconvexSemidefinite.jl**, consumed here through
   a package extension (weakdep), as agreed in issue #40.
5. **Matrix-free options, selectable per constraint via labels**: each
   constraint's evaluation strategy (dense / matrix-free / trace-estimated) is
   chosen through NonconvexCore's `flags` label mechanism, so a single model
   can mix strategies. The KKT solve falls back to a preconditioned Krylov
   solver whenever any matrix-free constraint is present. Randomized **trace
   estimation** is available as an approximate mode for labeled constraints.
6. **Relaxation**: bound relaxation for infeasible starts and flag-based
   constraint relaxation (AugLagLab `RelaxConstraintSelection` style).
7. Full test coverage, docs, and CI identical to the rest of the ecosystem.

Explicit non-goal: mixed-integer support (declared via `solver_capabilities`).

## 2. Algorithm (as implemented)

A **primal barrier** interior-point method. The state is `x` only: inequalities,
box bounds and set constraints enter a `μ`-scaled barrier objective, their
slacks are derived exactly (`s = -c_I(x)`), and only the equality constraints
contribute Jacobians and multipliers to the KKT system. Barrier gradients are
assembled from adjoints (pullbacks) — Jacobians of inequality/set constraints
are never materialized.

### 2.1 Reformulation

```
min  f(x)
s.t. c_E(x) = 0,
     c_I(x) ≤ 0     (violation form; NonconvexCore's f(x) ≤ 0 convention),
     x_l ≤ x ≤ x_u,
     g_j(x) ∈ K_j   (set constraints, see §4)
```

The barrier subproblem at `μ > 0`:

```
min  φ_μ(x) = f(x) - μ Σ ln(-c_{I,i}(x)) - μ Σ ln(x_j - x_l,j) - μ Σ ln(x_u,j - x_j)
              + μ Σ (set barrier terms)
s.t. c_E(x) = 0
```

- Infeasible starts use a smooth **C¹ extension** of the log barrier (bounded
  curvature, acting as a feasibility penalty of strength `μ/ε`); the iterate is
  switched to the strict barrier once feasible.
- `slacks = true` (or `BarrierSelection` exclusions, §6) switches to the
  Ipopt-style slacked formulation `c_I(x) + s = 0, s ≥ 0` with state `(x, s)`
  and the slack barrier `-μ Σ ln sᵢ`.
- The PSD barrier requires a strictly feasible (positive definite) start; the
  solver throws an informative error otherwise.

### 2.2 KKT system

```
[ H + δw I   J_Eᵀ ] [d_x]   [-g - J_Eᵀ λ_E]
[ J_E        -δc I] [d_λ] = [-c_E         ]
```

with `H = ∇²φ_μ`. The regularization `(δw, δc)` is grown (inertia correction)
until the symmetric matrix has `n` positive and `m_E` negative eigenvalues.
The system is solved either as an assembled dense matrix (eigen-based solve) or
matrix-free with **preconditioned MINRES** (Krylov.jl) whenever any
matrix-free/trace-estimated constraint is present (`linear_solver = :auto`).
`H` is applied through Hessian-vector products (DI second-order machinery);
with dense evaluation it is assembled by second-order AD of `φ_μ`.

### 2.3 Line search and globalization

Backtracking on the **exact-penalty merit**

```
P_ν(x) = φ_μ(x) + ν ||c_E(x)||_1,   ν = max(1, |λ_E| + 10),
```

with fraction-to-the-boundary step limits and the Armijo condition. A (near)
pure-dual Newton step — negligible `d_x`, e.g. when only the equality
multipliers need updating — bypasses the backtracking when the merit stays
finite and essentially non-increasing (the `ν|c_E|` kink would otherwise stall
the multiplier update at the rounding scale). This choice replaced an
experimental filter line search, which did not pay for itself on the
nonconvex test set.

### 2.4 Barrier update and termination

Quality-function rule (Ipopt): when the barrier KKT error
`E_μ = max(dual infeasibility, primal infeasibility)` satisfies `E_μ ≤ θ_μ·μ`
the decrease is superlinear (`μ ← μ^1.5`), otherwise linear (`μ ← 0.2 μ`),
with a linear escape below `E_μ ≤ 1e-3` or for near-feasible iterates.

Termination combines a strict criterion (`μ ≤ μ_min`, primal feasible, scaled
dual infeasibility within `tol`) with a **practical endgame criterion**: at
`μ ≤ 1e-9`, primal feasibility within `tol` and a stationarity residual within
`max(100·tol, 1e-4)` are accepted — the barrier surface is too flat in
floating point for further progress at that scale.

### 2.5 Inner optimizer

`IPMOptions(inner = alg)` delegates each barrier subproblem to an inner
NonconvexCore optimizer (e.g. Ipopt) while the outer loop drives `μ → 0`. The
subproblem uses an always-finitely-extended barrier objective (violated
constraints carry a hard quadratic penalty). The inner-optimizer path
terminates when the inner solver succeeds at `μ ≤ μ_min` with a
primal-feasible point — the subproblem solution is `O(μ_min)` from the
original optimum. The built-in Newton solver (`inner = nothing`) is the
default and is the better-tested path.

## 3. Native linear constraints

- Linear constraints are recognized structurally (`NonconvexCore.LinearFunction`,
  `add_linear_constraint!`) or by a Jacobian-constancy probe, and stored as
  `(A, rhs)` blocks with a cached sparse Jacobian — never re-differentiated.
- `linear_constraints = true` (default) enables this; set `false` to force the
  generic AD path (diagnostics).
- Per-row barrier masks record which linear inequality rows are barrierified
  and which are slacked (§6).

## 4. Set constraints

NonconvexCore's sets (`FunctionInSet`, `PSDCone`, order sets, `Simplex`,
`SetProduct`) are handled as follows:

- Order sets, `Zeros`, `Nonnegatives`/`Nonpositives` and `Simplex` are
  converted to classical (often linear) equality/inequality constraints at
  model build time and enter the scalar barrier.
- `PSDCone` stays a set constraint and receives a barrier oracle through the
  package extension (`ext/NonconvexInteriorPointSemidefiniteExt.jl`), with the
  actual semidefinite code living in **NonconvexSemidefinite.jl**
  (`PSDBarrierOracle`), per issue #40:
  - **dense**: `logdet`/`tr(X⁻¹)`/`tr(X⁻²)/2` via Cholesky, analytic gradients
    and Hessians (no second-order AD through the factorization);
  - **matrix-free**: value/gradient/Hessian products through iterative solves
    (`X⁻¹v` by CG) only;
  - **trace-estimated**: randomized log-determinant estimates (§5).
  The `sd_type` (`:logdet`/`:inverse`/`:inverse_square`) is selected through
  `BarrierSelection`; the non-log-determinant types require the dense mode.
- The barrier-oracle interface is the mode/type-dispatched hook family
  (`barrier_value_c`, `barrier_grad_x!`, `barrier_hess_x!`,
  `barrier_hess_x_prod!` and their `sd_*` type-aware counterparts) with
  generic dense fallbacks in core, so the extension specializes per mode
  without overwriting core methods.

This mirrors **Hypatia.jl's** cone-oracle decomposition (MIT), the closest
design precedent for barrier-based set handling in the Julia ecosystem.

## 5. Matrix-free and trace-estimated evaluation — per constraint, via labels

Evaluation strategy is selected **per constraint** through NonconvexCore's
label mechanism: every constraint carries `flags::Set{Symbol}`, and the solver
reads the same `flags` field that NonconvexCore's meta-algorithms select on.
The selection is declared in `IPMOptions`:

```julia
IPMEvaluationSelection(;
    matrixfree::Union{Symbol,AbstractVector{Symbol}} = Symbol[],  # labels evaluated matrix-free
    trace::Union{Symbol,AbstractVector{Symbol}} = Symbol[],       # labels evaluated with trace estimation
    matrixfree_all::Bool = false,
    trace_all::Bool = false,
)
```

Semantics:

- **`matrixfree` label**: the constraint's Jacobian/Hessian is never
  materialized — products are implicit AD jvps — and PSD barriers are
  evaluated through iterative solves. Deterministic, no randomness.
- **`trace` label**: the constraint's log-determinant barrier is evaluated with
  **randomized trace estimation** (implies matrix-free). On non-set constraints
  `:trace` reduces to the `matrixfree` behavior.
- `:trace` dominates `:matrixfree` on the same constraint; an explicit label
  selection dominates the `*_all` flags; unlabeled constraints use the default
  dense strategy.
- **KKT solve mode**: `linear_solver = :auto | :direct | :krylov`. `:auto`
  (default) upgrades to the Krylov path whenever any `matrixfree`/`trace`
  constraint is present; `:direct` throws in that case; `:krylov` forces
  Krylov regardless.

### 5.1 Dense path

Assembled symmetric KKT matrix, eigen-based solve with the inertia check of
§2.2.

### 5.2 Krylov path

Preconditioned MINRES (Krylov.jl) on the augmented KKT with `(δw, δc)` damping
(Tulip/RipQP regularization trick). Exact inertia checks are unavailable on
the Krylov path, so curvature control relies on the damping heuristic plus the
merit line search. Hessian-vector products via AD jvps; labeled constraints
contribute only implicit products.

### 5.3 Trace-estimated barrier mode (per labeled constraint)

The randomized log-determinant estimators live in
**NonconvexSemidefinite.jl** (`logdet_estimate`, `logdet_estimate_of`) and are
selected with `trace_method`:

| method | estimator | solves only |
|---|---|---|
| `:slq` | stochastic Lanczos quadrature (Gauss quadrature of `log`) | ✓ |
| `:chebyshev` | Chebyshev expansion of `log` (needs spectral bounds) | ✓ |
| `:hutchpp` | Hutch++-style: Nyström part + deflated-range SLQ | ✓ |
| `:rational` | rational approximation via Gauss-Laguerre (coarse on wide spectra) | ✓ (shifted CG) |
| `:taylor` | Taylor expansion of `log` (needs spectral bounds) | ✓ |
| `:integration` | `log x = ∫₀¹ (x-1)/(1+t(x-1)) dt` via Gauss-Legendre | ✓ (shifted CG) |
| `:lowrank` | Lanczos low-rank approximation of `A - I` (near-identity matrices) | ✓ |

No factorization is required anywhere. The solve-based estimators (`:slq`,
`:hutchpp`, `:rational`, `:integration`, `:lowrank`) carry ChainRules reverse
rules whose pullbacks reuse the already-computed linear solves — so first-order
solvers can differentiate through the estimates. The polynomial methods'
(`:chebyshev`, `:taylor`) reverse rules throw an informative `ArgumentError`
(not implemented yet).

Estimator noise is incompatible with tight KKT convergence, so `trace`-labeled
constraints are an *approximate* regime with optional exact-oracle polish
(`trace_final_exact = true` switches labeled constraints to the exact oracle
for the final barrier iterations). Documented as a research-grade mode.

## 6. Barrier selection and relaxation

`BarrierSelection` selects, per category and per label, which constraints enter
the barrier and with which barrier type (the relaxation-selection counterpart
of AugLagLab's `RelaxConstraintSelection`):

- `ineq = false` excludes inequalities from the barrier (they become slacked)
  — the per-category counterpart of `slacks = true`;
- `box = false` excludes variable bounds;
- `sd = false` excludes set constraints;
- `labels` excludes constraints carrying any of the given labels;
- `ineq_type`/`box_type` ∈ `(:log, :inv, :quad)` choose the scalar barrier
  (`-ln t`, `1/t`, `1/(2t²)`);
- `sd_type` ∈ `(:logdet, :inverse, :inverse_square)` chooses the PSD barrier.

Relaxation:

1. `bound_relax_factor` (Ipopt): relax bounds so `x0` is strictly interior.
2. `relaxed_labels`: constraints carrying these labels are moved to a
   quadratic penalty (`relaxed_penalty_rho`) instead of being enforced.

## 7. Relationship to NonconvexSemidefinite.jl and NonconvexAugLagLab.jl

| Algorithm (existing) | Verdict |
|---|---|
| `NonconvexSemidefinite.BarrierAlg` (outer barrier loop + generic sub-solver) | **Replaceable** by running `IPMAlg` directly on the model: a native barrier IPM with a PSD barrier oracle supersedes the outer-loop barrier with a generic NLP sub-solver. |
| `NonconvexSemidefinite.SDPQuadAlg` (Cholesky/Burer–Monteiro reparametrization) | Complementary — keep (reparametrization is orthogonal to the solver; can even use `IPMAlg` as its sub-solver). |
| `NonconvexSemidefinite.SDCuttingPlaneAlg` | Complementary — keep; its cuts are linear constraints our native linear path handles well (future synergy). |
| `NonconvexAugLagLab.AugLagAlg` on SDP problems | `IPMAlg` becomes the recommended alternative for mid-size SDP NLPs; AugLag remains the tool for very large problems (esp. with Anderson acceleration). The trace-estimated barrier mode is new capability neither has. |
| `NonconvexAugLagLab.QuasiNewtonAlg` | Out of scope. |

## 8. API sketch

```julia
using NonconvexInteriorPoint  # re-exports NonconvexCore

# Solver + options
IPMAlg                      # <: NonconvexCore.AbstractOptimizer
IPMOptions(; tol = 1e-8, mu_init = 0.1, mu_min = 1e-11,
    mu_linear_decrease_factor = 0.2, mu_superlinear_decrease_power = 1.5,
    theta_mu = 1.5, max_iter = 3000, linear_solver = :auto,
    slacks = false, barrier = BarrierSelection(), inner = nothing,
    inner_options = nothing, inner_max_iter = 50,
    evaluation = IPMEvaluationSelection(),
    krylov_tol = 1e-8, krylov_maxiter = 200,
    trace_nprobes = 10, trace_lanczos_deg = 30, trace_method = :slq,
    trace_cg_tol = 1e-6, trace_final_exact = false,
    linear_constraints = true, bound_relax_factor = 1e-8,
    relaxed_labels = nothing, relaxed_penalty_rho = 1e4,
    tau = 0.995, s_min = 1e-8, max_time = Inf,
    verbose = false, rng = Random.default_rng(), ...)

BarrierSelection(; ineq = true, box = true, sd = true, labels = Symbol[],
    ineq_type = :log, box_type = :log, sd_type = :logdet)

# Standard ecosystem entry points
Workspace(model::VecModel, ::IPMAlg, x0; options = IPMOptions(), kwargs...)
optimize!(ws::IPMWorkspace)::IPMResult          # minimizer, minimum, status, fcalls, ...
NonconvexCore.solver_capabilities(::IPMAlg)     # eq/ineq/sd/sets/box, continuous
```

Capabilities declared: linear/nonlinear eq, linear/nonlinear ineq,
`sets_eq/ineq/sd/simplex/product`, `variable_sets_*`, box, continuous,
`infinite_bounds`; integer/binary unsupported.

## 9. Package layout and ecosystem integration

Single-package layout mirroring NonconvexAugLagLab.jl:

```
src/NonconvexInteriorPoint.jl   # module, exports, includes
src/options.jl                  # IPMOptions, BarrierSelection, IPMEvaluationSelection
src/result.jl                   # IPMResult
src/ad.jl                       # hvp (DI second-order), jacprod, jtprod
src/model.jl                    # flattening into the IPMModel form (masks, linear blocks, set conversion)
src/barrier.jl                  # barrier oracle hook family (value/grad/Hessian, sd types)
src/trace_estimation.jl         # Hutchinson / Hutch++ trace estimators (legacy core helpers)
src/kkt.jl                      # barrier objective/gradient, merit, KKT assembly + eigen/MINRES solves
src/line_search.jl              # step bounds, mode helpers
src/slacked.jl                  # the slacked formulation path (slacks = true / masked)
src/workspace.jl                # IPMAlg, outer μ loops, built-in and inner-optimizer paths
ext/NonconvexInteriorPointSemidefiniteExt.jl  # weakdep NonconvexSemidefinite
test/                           # @testitem per source file + aqua/jet
docs/                           # DocumenterFragments fragment
.github/workflows/              # CI, Documentation, CompatHelper, TagBot,
                                # format-check, formatter, JET (copied from
                                # NonconvexAugLagLab, branch triggers = main)
```

- License: **MIT** (ecosystem standard). All design references used have
  permissive licenses: Ipopt EPL-2.0, MadNLP MIT, Uno MIT, Hypatia MIT,
  Clarabel Apache-2.0, COSMO Apache-2.0, Tulip MPL-2.0, RipQP MIT.
- Weakdeps: `NonconvexSemidefinite` (PSD barrier oracles). No GPL-licensed
  solver (ECOS, SDPA, SeDuMi) is touched.
- Default branch `main`; all workflow `branches:` filters use `main`.

## 10. Design influences (summary)

| Source | What we take |
|---|---|
| Ipopt | IC-style regularized KKT + adaptive μ; `bound_relax_factor`; linear-constraint caching; slacked formulation option |
| MadNLP | Julia implementation patterns, KKT regularization |
| Uno | globalization variants (studied; merit backtracking chosen), presets |
| Clarabel | scaling & iterative refinement discipline (study only; HSDE is convex-conic) |
| Hypatia | cone/barrier **oracle decomposition** (the template for §4) |
| Tulip / RipQP | regularized KKT + Krylov MINRES for the matrix-free path (§5) |
| FANOK / SLQ / Hutch++ | trace-estimated barrier mode (§5) |
| Kočvara–Stingl PBM | problem class (nonconvex NSDP, buckling); matrix-free AL precedent |

## 11. Milestones

1. **M1 — core IPM**: box + inequalities + equalities, primal barrier with
   exact-penalty merit backtracking. ✅
2. **M2 — adaptive μ (quality-function rule), practical convergence, result
   statuses**. ✅
3. **M3 — native linear constraints** with cached Jacobians. ✅
4. **M4 — set constraints** incl. `PSDCone` via the extension (dense,
   matrix-free, trace oracles; sd barrier types). ✅
5. **M5 — per-constraint matrix-free path** (`matrixfree` labels) + Krylov KKT
   solve. ✅
6. **M6 — trace estimation mode** (`trace` labels; SLQ/Chebyshev/Hutch++/
   rational/Taylor/integration/low-rank with rrules in NonconvexSemidefinite). ✅
7. **M7 — BarrierSelection, slacked mode, inner-optimizer hook, relaxation
   flags, docs, CI, Aqua/JET green.** ✅ (v0.1)

## 12. Risks & mitigations

- **Indefinite KKT in nonconvex problems** → `(δw, δc)` inertia correction.
- **Krylov on indefinite augmented systems** → MINRES with `(δw, δc)` damping;
  dense fallback when the Krylov solve fails.
- **Barrier endgame flatness** → practical convergence criterion (§2.4) plus
  the near-pure-dual step handling (§2.3).
- **Trace-estimator noise vs. Newton convergence** → opt-in per-constraint
  (`trace` label) approximate mode with exact-oracle polish; no KKT-convergence
  claims for labeled constraints in that mode.
- **AD overhead** → DI prepared caches, adjoint-based barrier gradients (no
  Jacobians for inequalities/sets), native linear constraints.
- **Ecosystem drift** → tests pin against NonconvexCore 2.x compat bounds;
  CompatHelper keeps them fresh.
