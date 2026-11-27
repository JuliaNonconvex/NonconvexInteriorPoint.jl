# Theory

`IPMAlg` is a primal barrier interior-point method in the lineage of
Ipopt, MadNLP and Uno, with two distinguishing twists: inequalities and
set constraints never require materialized Jacobians (their barrier
gradients are assembled from adjoints/pullbacks only), and semidefinite
barriers can be estimated stochastically. This page documents the
algorithmic theory; a worked application (buckling-constrained truss
design) is in [A buckling-constrained example](buckling.md).

## The primal barrier formulation

Consider

```math
minimize \; f(x) \quad s.t. \; c_E(x) = 0, \; c_I(x) ≤ 0, \; x_l ≤ x ≤ x_u,
```

with `c_I(x) ≤ 0` in violation form (NonconvexCore's inequality convention:
`add_ineq_constraint!(m, f)` enforces `f(x) ≤ 0`). The barrier subproblem at
barrier parameter `μ > 0` is

```math
minimize \; φ_μ(x) = f(x) - μ \sum_i \ln(-c_{I,i}(x))
                      - μ \sum_j \ln(x_j - x_{l,j})
                      - μ \sum_j \ln(x_{u,j} - x_j)
\quad s.t. \; c_E(x) = 0.
```

The state is `x` only — no slack variables: the slacks are
`s = -c_I(x)`, the barrier multipliers are `λ_I = μ/s`, and the iterates
follow the central path `x(μ) → x*` as `μ → 0`. Equality constraints are
enforced through the KKT system with multipliers `λ_E`; a violated start is
handled with a smooth C¹ extension of the log barrier (bounded curvature,
acting as a feasibility penalty of strength `μ/ε`).

### Scalar barrier types

`BarrierSelection` replaces `-ln t` on the inequality slacks and bound
distances by other classical barriers:

| `ineq_type`/`box_type` | barrier `φ(t)` | `φ'(t)` | `φ''(t)` |
|---|---|---|---|
| `:log` | `-ln t` | `-1/t` | `1/t²` |
| `:inv` | `1/t` | `-1/t²` | `2/t³` |
| `:quad` | `1/(2t²)` | `-1/t³` | `3/t⁴` |

All three are convex and unbounded at `t → 0⁺` (so the barrier minimizer
stays interior), but they have different curvature profiles: the
higher-order barriers concentrate curvature near the boundary, which can
help or hurt the endgame conditioning depending on the problem. The log
barrier is the only self-concordant one — hence the default.

### The slacked formulation

The `slacks = true` option (or `BarrierSelection` exclusions) switches to
the Ipopt-style slacked formulation: `c_I(x) + s = 0`, `s ≥ 0`, with state
`(x, s)`, slack barrier `-μ Σ ln sᵢ`, and the primal-dual KKT system

```math
\begin{bmatrix}
H + σ_x I & J_E^T & J_I^T \\
J_E & -δ_c I & 0 \\
J_I & 0 & -Σ I
\end{bmatrix}
\begin{bmatrix} d_x \\ d_{λ_E} \\ d_{λ_I} \end{bmatrix}
=
- \begin{bmatrix} g + J_E^T λ_E + J_I^T λ_I \\ c_E \\ c_I + s \end{bmatrix},
\qquad Σ = \mathrm{diag}(λ_{I,i}/s_i),
```

the Schur complement of the Ipopt/MadNLP style (the barriered inequalities
keep the slacks implicit in the unslacked form, which eliminates the
`Σ` block).

## Newton step

At each barrier iterate the regularized KKT system

```math
\begin{bmatrix} H + δ_w I & J_E^T \\ J_E & -δ_c I \end{bmatrix}
\begin{bmatrix} d_x \\ d_λ \end{bmatrix}
=
-\begin{bmatrix} g + J_E^T λ_E \\ c_E \end{bmatrix}
```

is solved, where `H = ∇²φ_μ` and `J_E = ∇c_E`. The regularization `(δ_w, δ_c)`
is grown until the inertia is correct (`n` positive and `m_E` negative
eigenvalues — inertia correction), following the Wächter–Biegler
regularization ladder. The system is solved either as an assembled dense
symmetric matrix (eigen-based) or matrix-free with preconditioned MINRES
when any matrix-free/trace-estimated constraint is present.

The Hessian application is split by design: the **scalar part** (objective,
relaxed penalties, inequality and bound barriers) is applied through
second-order AD (forward-over-reverse Hessian-vector products), while the
**set barriers** are applied through their analytic Hessian products — the
matrix-free and trace-estimated PSD values are not AD-differentiable (they
involve iterative solvers), and the analytic products are deterministic
functions of `x`, which the Krylov KKT solver requires of its operator.

## Line search

Steps are taken with backtracking on the exact-penalty merit

```math
P_ν(x) = φ_μ(x) + ν \|c_E(x)\|_1, \qquad ν = \max(1, |λ_E| + 10),
```

with fraction-to-the-boundary step limits and the Armijo condition. The
penalty parameter dominates the equality multipliers, so a step that
increases `φ_μ` can still be accepted if it reduces the equality violation.
A (near) pure-dual Newton step (negligible `d_x`, e.g. when only the
equality multipliers need updating) bypasses the backtracking when the
merit stays finite and essentially non-increasing — the `ν‖c_E‖₁` kink at
`c_E = 0` would otherwise stall the multiplier update at the rounding
scale.

## Barrier update and termination

`μ` is updated with the quality-function rule: when the barrier KKT error
`E_μ = max(dual infeasibility, primal infeasibility)` satisfies
`E_μ ≤ θ_μ · μ` the decrease is superlinear (`μ ← μ^1.5`), otherwise linear
(`μ ← 0.2 μ`) — with a linear escape below `E_μ ≤ 1e-3` or for
near-feasible iterates so the endgame always progresses.

Termination combines a strict criterion — `μ ≤ μ_min`, primal feasible, and
scaled dual infeasibility within `tol` — with a practical endgame criterion:
at `μ ≤ 1e-9`, primal feasibility within `tol` and a stationarity residual
within `max(100 tol, 1e-4)` are accepted, since the barrier surface is too
flat in floating point for further progress at that scale.

## Feasibility stage

When the initial point violates the inequalities, bounds or equalities,
`IPMAlg` first runs a **feasibility stage**: it minimizes the C¹ constraint
violation

```math
½ Σ max(c_{I,i}(x) + ε, 0)² + ½ ‖c_E(x)‖² + ½ Σ (bound violations)²
```

with a regularized Newton method (MINRES on `H + δI`, Armijo backtracking)
until every inequality and bound distance clears the margin `ε = 1e-2` —
enough that the first barrier iterations are not pinned to the boundary
(the barrier multiplier `μ/s` starts moderate, and a landing exactly on the
boundary degenerates the barrier Hessian). The stage budget is
`feasibility_max_iter` (default 200, `0` disables it); whatever it cannot
restore (e.g. a degenerate corner of a nonconvex region) is left to the C¹
extended barrier of the main loop. PSD constraints still require a strictly
feasible starting point.

## Semidefinite barriers

For `X(x) ⪰ 0` the barrier is `-log det X` (or `tr(X⁻¹)` /
`tr(X⁻²)/2` via `BarrierSelection(sd_type = ...)`). The gradient is
assembled from the adjoint (pullback) of the constraint map with the
matrix-space gradient

```math
-⟨X^{-1}, ∂X/∂x_i⟩, \qquad
-⟨X^{-2}, ∂X/∂x_i⟩, \qquad
-\tfrac{1}{2}⟨X^{-2}, ∂X/∂x_i⟩
```

for `:logdet`, `:inverse` and `:inverse_square` — the Jacobian of `X(x)` is
never materialized. The Hessian uses the analytic second-order forms

```math
∇²_x \mathrm{tr}(X^{-q})/q \cdot Y = \sum_{k=0}^{q} X^{-(k+1)} Y X^{-(q+1-k)},
\qquad q = 1, 2,
```

so no second-order AD through the factorization is needed (the dense value
path is dual-safe, allowing forward-over-reverse differentiation when the
oracles are composed inside first-order solvers).

## Matrix-free and trace-estimated log-determinant estimation

The `:trace` evaluation mode replaces `log det X` by a randomized estimate.
Every estimator operates through matrix-vector products and shifted linear
solves only — no factorization is required anywhere. The mathematical
identities behind the suite (in NonconvexSemidefinite.jl):

**Stochastic Lanczos quadrature (`:slq`)** — Golub–Meurant/Ubaru–Chen–Saad.
The `k`-step Lanczos tridiagonalization of `A` seeded at a probe `z` yields
Ritz pairs `(θᵢ, q̂ᵢ)` that define the `k`-point Gauss quadrature of the
spectral measure of `A` induced by `z`:

```math
z^T \log(A) z \approx \sum_{i=1}^k (z^T q̂_i)^2 \log θ_i,
```

and, since `E[z z^T] = I` for Gaussian (or normalized) probes,
`(n/nprobes) Σ z^T log(A) z` is an unbiased estimator of `tr log A = log det A`.
Exact when the Lanczos space spans the spectrum.

**Chebyshev (`:chebyshev`)** — expand `g(t) = log(α t + β)` on
`[lmin, lmax]` in Chebyshev polynomials (coefficients by Gauss–Chebyshev
quadrature), then estimate `tr T_j(Â)` of the shifted/scaled matrix by
Hutchinson with the three-term recurrence.

**Hutch++ (`:hutchpp`)** — split the determinant with a Nyström part and a
deflated tail:

```math
\log\det A = \log\det(Q^T A Q)
            + \log\det\bigl((I - QQ^T) A (I - QQ^T)\bigr),
```

where `Q` is a random sketch basis. The first term is computed on the small
matrix `Q'AQ` through its matrix-vector products (Lanczos for the value,
CG for the adjoint); the second by SLQ with probes restricted to the range
of `I - QQ'` (the null space would corrupt the Gauss quadrature).

**Rational (`:rational`)** — the partial-fraction identity
`log x = ∫₀^∞ (1/(1+t) - 1/(x+t)) dt`, discretized with Gauss–Laguerre
nodes: `log x ≈ Σ_k w_k e^{t_k} (1/(1+t_k) - 1/(x+t_k))`. The resolvent
traces `tr((A + t_k I)⁻¹)` are estimated by Hutchinson with shifted
conjugate gradients. Documented as coarse on wide spectra: the `e^{t_k}`
reweighting makes the integrand grow exponentially, so the quadrature bias
does not shrink much with the node count.

**Taylor (`:taylor`)** — `log A = log c + Σ_{j≥1} (-1)^{j+1} ((A - cI)/c)^j/j`
at the spectrum midpoint `c`; the moments `tr((A - cI)^j)` are accumulated
from the stored Krylov vectors.

**Trace integration (`:integration`)** — the integral identity
`log x = ∫₀¹ (x-1)/(1+t(x-1)) dt`, discretized with Gauss–Legendre; each
node needs `tr((tA + (1-t)I)⁻¹(A-I))` via Hutchinson and one shifted CG
solve.

**Low-rank (`:lowrank`)** — for near-identity matrices `A = I + B`:
`log det(I + B) = Σ log(1 + θᵢ)` from a Lanczos run on `B`.

The solve-based estimators (`:slq`, `:hutchpp`, `:rational`,
`:integration`, `:lowrank`) carry ChainRules reverse rules whose pullbacks
reuse the linear solves already computed for the value — never a
factorization — so first-order solvers can differentiate through the
estimates. The estimator noise is incompatible with tight KKT convergence,
so the trace mode is an approximate regime with optional exact-oracle
polish (`trace_final_exact`).

## Convergence

For convex problems the log-barrier central path inherits the standard
self-concordance analysis: each `μ`-reduction step (at a fixed factor)
requires a bounded number of Newton steps, giving polynomial complexity in
the problem dimension and the accuracy. For the nonconvex problems this
solver targets, no global convergence guarantee holds (nor can it — finding
stationary points of general nonconvex NLPs is intractable); the practical
behavior matches the Ipopt/MadNLP/Uno family: the merit line search and
inertia correction make the iteration well-defined whenever the iterates
stay bounded, and the practical endgame criterion accepts the flat-barrier
limit.

## Inner optimizer

When `IPMOptions(inner = alg)` is set, the outer loop drives `μ → 0` and
delegates each barrier subproblem to an inner `NonconvexCore` optimizer
(e.g. Ipopt), using an always-finitely-extended barrier objective (violated
constraints carry a hard quadratic penalty). The inner-optimizer path
terminates when the inner solver succeeds at `μ ≤ μ_min` with a
primal-feasible point — the subproblem solution is `O(μ_min)` from the
original optimum.
