# NonconvexInteriorPoint.jl — Design

A native interior-point solver for general (nonconvex) nonlinear programs in the
JuliaNonconvex ecosystem, built on NonconvexCore.jl. This document records the
algorithmic and architectural decisions before implementation. It incorporates
the literature survey from
[issue #40](https://github.com/JuliaNonconvex/NonconvexSemidefinite.jl/issues/40)
and a review of the state of the art in the surrounding ecosystem.

---

## 1. Goals

1. **State-of-the-art barrier IPM** for smooth nonconvex NLPs, following the
   Ipopt → MadNLP → Uno lineage (filter line search, inertia control,
   restoration).
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
6. **Relaxation**: bound/constraint relaxation for infeasible starts, a
   restoration phase, and flag-based constraint relaxation (AugLagLab
   `RelaxConstraintSelection` style).
7. Full test coverage, docs, and CI identical to the rest of the ecosystem.

Explicit non-goal: mixed-integer support (declared via `solver_capabilities`).

## 2. Algorithm

Primal-dual interior point with a filter line search, as in
Wächter & Biegler (Math. Prog. 2006), MadNLP, and Uno.

### 2.1 Reformulation

```
min  f(x)
s.t. c_E(x) = 0,
     c_I(x) + s = 0,   s ≥ 0,
     x_l ≤ x ≤ x_u,
     g_j(x) ∈ K_j      (set constraints, see §4)
```

Inequalities are slacked (`s`), and the log barrier `−μ Σ ln sᵢ` is applied to
the slack bounds (Ipopt-style; box bounds on `x` use the same mechanism).
Set constraints receive a set-specific barrier (see §4).

### 2.2 KKT system (augmented form)

```
[ W + Σ + δw I   Aᵀ ] [Δx]   [rhs_x]
[ A             −δc I] [Δλ] = [rhs_c]
```

with `W` the Hessian of the Lagrangian (including barrier terms), `A` the
constraint Jacobian, `Σ` the barrier Hessian diagonal for slacks/bounds, and
`(δw, δc)` regularization from the inertia-correction loop. Inertia of the
LDLᵀ factorization must match `(n + m_neg_evals, m, 0)`; otherwise Wächter &
Biegler's IC procedure updates `(δw, δc)`. Fraction-to-the-boundary rules guard
primal and dual step sizes; slacks are kept strictly positive.

### 2.3 Line search and globalization

- **Filter** method (default), with a feasibility **restoration phase**;
- optional plain backtracking on the barrier merit function;
- optional second-order corrections (filter mode).

### 2.4 Barrier update

Adaptive `μ` update driven by the barrier "quality function" (Ipopt),
with monotone/superlinear fallback, `mu_init`, `mu_min`, and a KKT-based
termination criterion (`NonconvexCore.Tolerance` / `KKTCriteria`).

### 2.5 Infeasibility handling

Homogeneous-embedding-style *detection* is out of scope (that machinery is
convex-conic, à la Clarabel); instead: bound relaxation, restoration, and a
final status `:infeasible` when the restoration phase fails and the constraint
violation stagnates.

## 3. Native linear constraints

- Linear constraints are recognized structurally (`NonconvexCore.LinearFunction`,
  `add_linear_constraint!`) or by a Jacobian-constancy probe, and stored as
  `(A, l, u)` with a cached sparse Jacobian — never re-differentiated.
- `linear_constraints = true` (default) enables this; set `false` to force the
  generic AD path (diagnostics).
- The constant `A` is exploited in preconditioners (Schur complement w.r.t. the
  linear block).

## 4. Set constraints

NonconvexCore ≥ 2.3 already models sets (`FunctionInSet`, `VariableInSet`,
`AbstractSet`, `PSDCone`, order sets, `Simplex`, `SetProduct`). What is missing
is a barrier-oracle protocol. We add a small, NonconvexCore-independent
interface in this package:

```julia
barrier_value(set, x)::Real
barrier_grad!(g, set, x)            # g = ∇ₓ φ
barrier_hess!(H, set, x)            # H = ∇²ₓₓ φ (sparse-friendly)
```

- Order sets (`LessThan`, `GreaterThan`, `Interval`, `EqualTo`, `Zeros`),
  `Nonnegatives`/`Nonpositives`, and `Simplex` get closed-form barriers.
- `PSDCone` is implemented in **NonconvexSemidefinite.jl** (extended there,
  since semidefinite code must live in that package) with three evaluation
  variants — **dense** (`logdet` via Cholesky), **matrix-free** (iterative
  solves, deterministic), and **trace-estimated** (§5) — selected per
  constraint by its `flags` label at workspace build time. This package loads
  them through `ext/NonconvexInteriorPointSemidefiniteExt.jl`.
- Fallback for sets without a barrier oracle: smooth log-barrier of
  `set_violation` (component-wise), flagged in the docs as suboptimal.

This mirrors **Hypatia.jl's** cone-oracle decomposition (MIT), the closest
design precedent for barrier-based set handling in the Julia ecosystem.

## 5. Matrix-free and trace-estimated evaluation — per constraint, via labels

Evaluation strategy is selected **per constraint** through NonconvexCore's
label mechanism: every constraint carries `flags::Set{Symbol}`
(`add_ineq_constraint!(m, f; flags=[:large])`, `add_set_constraint!(m, f, set;
flags=...)`, `add_sd_constraint!(m, f, d; flags=...)`, `VariableInSet(..., flags
=...)`), and the solver reads the same `flags` field that NonconvexCore's
meta-algorithms (`RelaxConstraintSelection`, `BarrierSelection`) select on. The
selection is declared in `IPMOptions`:

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
  materialized — products are implicit AD `jvp`s — and set/PSD barriers are
  evaluated through iterative solves (`X⁻¹v` by preconditioned CG, `log det X`
  by Lanczos quadrature with exact quadrature). Deterministic, no randomness.
- **`trace` label**: the constraint is evaluated with **randomized trace
  estimation** (implies matrix-free): Hutchinson/Hutch++ for `tr(X⁻¹E)` and
  `tr(X⁻¹A X⁻¹B)` contractions, stochastic Lanczos quadrature for `log det X`.
  On non-set constraints (whose barrier acts on scalar slacks) `:trace` reduces
  to the `matrixfree` behavior — no stochastic part exists there.
- `:trace` dominates `:matrixfree` on the same constraint; an explicit label
  selection dominates the `*_all` flags; unlabeled constraints use the default
  dense strategy.
- **KKT solve mode**: `linear_solver = :auto | :direct | :krylov`. `:auto`
  (default) upgrades to the Krylov path whenever any `matrixfree`/`trace`
  constraint is present (a fully assembled KKT does not exist then); `:direct`
  throws in that case; `:krylov` forces Krylov regardless.

### 5.1 Dense path

`ldlt` (LinearAlgebra) on the assembled augmented system, for small/mid
problems with no labeled constraints.

### 5.2 Krylov path

Preconditioned MINRES (symmetric indefinite; GMRES/BiCGSTAB when an asymmetric
preconditioner is selected) with `(δw, δc)` damping preserving curvature — the
Tulip/RipQP regularization trick. Exact inertia checks are unavailable on the
Krylov path (no factorization), so curvature control relies on the damping
heuristic plus the filter. Hessian-vector products via AD `jvp`s, Jacobian
products likewise; labeled constraints contribute only implicit products,
unlabeled ones may be assembled and applied block-wise.
Preconditioners: diagonal, block-diagonal (box + slacks), L-BFGS of `W`,
Schur-based for the native linear block.

### 5.3 Trace-estimated barrier mode (per labeled constraint)

For large sparse PSD constraints (the nonconvex generalization of issue #40's
"stochastic PBM" gap):

- `log det X` by **stochastic Lanczos quadrature** (Ubaru–Chen–Saad);
- `tr(X⁻¹E)` gradient contractions by **Hutchinson** (Rademacher/Gaussian
  probes) with `X⁻¹v` via preconditioned CG — never forming `X⁻¹`;
- Hessian contractions `tr(X⁻¹A X⁻¹B)` likewise;
- variance reduction via Hutch++/Nyström (`:hutchpp`);
- knobs: `trace_nprobes`, `trace_cg_tol`.

Estimator noise is incompatible with tight KKT convergence, so `trace`-labeled
constraints are handled as an *approximate* regime: they converge to a
statistical tolerance, with optional exact-oracle polish (`trace_final_exact =
true` switches labeled constraints to the dense or matrix-free oracle for the
final barrier iterations). Explicitly documented as a research-grade mode.

## 6. Relaxation

1. `bound_relax_factor` (Ipopt): relax bounds/slacks so `x0` is strictly
   interior-feasible with respect to bounds and slacked inequalities.
2. **Restoration phase** for the filter.
3. `relaxed::RelaxConstraintSelection`-style flag selection: constraints with
   the `:relax` label (or an explicit selection) are moved to a
   log-barrier/quadratic penalty instead of hard constraints — reusing the
   label mechanism already in NonconvexCore (AugLagLab precedent).

## 7. Relationship to NonconvexSemidefinite.jl and NonconvexAugLagLab.jl

| Algorithm (existing) | Verdict |
|---|---|
| `NonconvexSemidefinite.BarrierAlg` (outer barrier loop + generic sub-solver) | **Replaceable** by running `IPMAlg` directly on the model: a native barrier IPM with a PSD barrier oracle supersedes the outer-loop barrier with a generic NLP sub-solver. Keep `BarrierAlg` as a compat shim (deprecated path) in a later release. |
| `NonconvexSemidefinite.SDPQuadAlg` (Cholesky/Burer–Monteiro reparametrization) | Complementary — keep (reparametrization is orthogonal to the solver; can even use `IPMAlg` as its sub-solver). |
| `NonconvexSemidefinite.SDCuttingPlaneAlg` | Complementary — keep; its cuts are linear constraints our native linear path handles well (future synergy). |
| `NonconvexAugLagLab.AugLagAlg` on SDP problems | `IPMAlg` becomes the recommended alternative for mid-size SDP NLPs; AugLag remains the tool for very large problems (esp. with Anderson acceleration). The trace-estimated barrier mode is new capability neither has. |
| `NonconvexAugLagLab.QuasiNewtonAlg` | Out of scope. |

This evaluation lives in the docs (`docs/src/replacements.md`), with benchmark
scripts (buckling-constrained truss, Gaussian MLE, nearest correlation matrix)
comparing `IPMAlg`, `BarrierAlg`, `AugLagAlg`, `IpoptAlg`.

## 8. API sketch

```julia
using NonconvexInteriorPoint  # re-exports NonconvexCore

# Solver + options
IPMAlg                      # <: NonconvexCore.AbstractOptimizer
IPMOptions(; tol = Tolerance(), mu_init = 0.1, mu_min = 1e-11,
    mu_linear_decrease_factor = 0.2, mu_superlinear_decrease_power = 1.5,
    max_iter = 3000, linear_solver = :auto,
    evaluation = IPMEvaluationSelection(),
    krylov_tol = 1e-8, trace_nprobes = 10, trace_final_exact = false,
    linear_constraints = true, bound_relax_factor = 1e-8,
    relaxed = RelaxConstraintSelection(), filter = true, soc = false,
    verbose = false, ...)

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
src/options.jl, result.jl, workspace.jl
src/model.jl                    # flattening into IPM form (slacks, linear blocks)
src/barrier.jl                  # barrier value/grad/Hessian assembly (incl. sets)
src/kkt.jl                      # augmented system assembly + inertia correction
src/linear_solvers.jl           # :direct (ldlt) / :krylov (Krylov.jl)
src/preconditioners.jl
src/filter.jl, line_search.jl, restoration.jl, mu_update.jl
src/sets.jl                     # barrier-oracle protocol + generic fallbacks
src/trace_estimation.jl         # Hutchinson / Hutch++ / SLQ
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
| Ipopt | filter + IC + restoration + adaptive μ; `bound_relax_factor`; linear-constraint caching |
| MadNLP | Julia implementation patterns, linear-solver abstraction, KKT scaling |
| Uno | globalization variants (filter), restoration, presets |
| Clarabel | scaling & iterative refinement discipline (study only; HSDE is convex-conic) |
| Hypatia | cone/barrier **oracle decomposition** (the template for §4) |
| Tulip / RipQP | regularized KKT + iterative refinement for the Krylov path (§5) |
| COSMO | chordal decomposition hook for sparse SDPs (via NonconvexSemidefinite) |
| FANOK / SLQ / Hutch++ | trace-estimated barrier mode (§5) |
| Kočvara–Stingl PBM | problem class (nonconvex NSDP, buckling); matrix-free AL precedent |

## 11. Milestones

1. **M1 — core IPM**: box + slacked inequalities + equalities, direct LDLᵀ,
   backtracking; passes vs Ipopt on CUTEst-style small problems.
2. **M2 — filter + restoration + SOC**, adaptive μ, result statuses.
3. **M3 — native linear constraints** with cached Jacobians.
4. **M4 — set constraints** incl. `PSDCone` via the extension (dense oracle).
5. **M5 — per-constraint matrix-free path** (`matrixfree` labels) + Krylov KKT
   solve + preconditioners.
6. **M6 — trace estimation mode** (`trace` labels: Hutchinson/Hutch++/SLQ) +
   benchmarks.
7. **M7 — relaxation flags, docs, CI, Aqua/JET green, benchmark vs Ipopt/
   MadNLP and the buckling NSDP suite; replacement evaluation for `BarrierAlg`.**

## 12. Risks & mitigations

- **Indefinite KKT in nonconvex problems** → inertia correction + filter/restoration.
- **Krylov on indefinite augmented systems** → MINRES with `(δw, δc)` damping;
  GMRES fallback; preconditioner library kept small initially.
- **Trace-estimator noise vs. Newton convergence** → opt-in per-constraint
  (`trace` label) approximate mode with exact-oracle polish; no KKT-convergence
  claims for labeled constraints in that mode.
- **AD overhead** → DifferentiationInterface prepared caches, sparse Jacobian
  detection (NonconvexCore `sparse_jacobian`), Lagrangian built once per
  iteration.
- **Ecosystem drift** → tests pin against NonconvexCore 2.x compat bounds;
  CompatHelper keeps them fresh.
