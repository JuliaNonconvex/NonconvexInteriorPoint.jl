# Theory

`IPMAlg` is a primal barrier interior-point method in the lineage of
Ipopt, MadNLP and Uno, with two distinguishing twists: inequalities and
set constraints never require materialized Jacobians (their barrier
gradients are assembled from adjoints/pullbacks only), and semidefinite
barriers can be estimated stochastically.

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

The `slacks = true` option (or `BarrierSelection` exclusions) switches to
the Ipopt-style slacked formulation `c_I(x) + s = 0`, `s ≥ 0` with the
slack barrier `-μ Σ ln sᵢ` and state `(x, s)`.

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
eigenvalues — inertia correction). The system is solved either as an
assembled dense symmetric matrix (eigen-based) or matrix-free with
preconditioned MINRES when any matrix-free/trace-estimated constraint is
present. `H` is applied through Hessian-vector products; with dense
evaluation it is assembled by second-order AD of `φ_μ`.

## Feasibility stage

When the initial point violates the inequalities, bounds or equalities,
`IPMAlg` first runs a **feasibility stage**: it minimizes the C¹ constraint
violation

```math
½ Σ max(c_{I,i}(x) + ε, 0)² + ½ ‖c_E(x)‖² + ½ Σ (bound violations)²
```

with a regularized Newton method (MINRES on `H + δI`, Armijo backtracking)
until every inequality and bound distance clears the margin `ε = 1e-2` —
enough that the first barrier iterations are not pinned to the boundary.
The stage budget is `feasibility_max_iter` (default 200, `0` disables it);
whatever it cannot restore (e.g. a degenerate corner) is left to the C¹
extended barrier of the main loop, so infeasible starts always make
progress. PSD constraints still require a strictly feasible starting
point.

## Line search

Steps are taken with backtracking on the exact-penalty merit

```math
P_ν(x) = φ_μ(x) + ν \|c_E(x)\|_1, \qquad ν = \max(1, |λ_E| + 10),
```

with fraction-to-the-boundary step limits and the Armijo condition. A
(near) pure-dual Newton step (negligible `d_x`, e.g. when only the equality
multipliers need updating) bypasses the backtracking when the merit stays
finite and essentially non-increasing.

## Barrier update and termination

`μ` is updated with the quality-function rule: when the barrier KKT error
`E_μ = max(dual infeasibility, primal infeasibility)` satisfies
`E_μ ≤ θ_μ · μ` the decrease is superlinear (`μ ← μ^1.5`), otherwise linear
(`μ ← 0.2 μ`) — with a linear escape below `E_μ ≤ 1e-3` or near-feasible
iterates so the endgame always progresses.

Termination combines a strict criterion — `μ ≤ μ_min`, primal feasible, and
scaled dual infeasibility within `tol` — with a practical endgame criterion:
at `μ ≤ 1e-9`, primal feasibility within `tol` and stationarity residual
within `max(100 tol, 1e-4)` are accepted, since the barrier surface is too
flat in floating point for further progress at that scale.

## Semidefinite barriers

For `X(x) ⪰ 0` the barrier is `-log det X` (or `tr(X⁻¹)` /
`tr(X⁻²)/2` via `BarrierSelection(sd_type = ...)`). The gradient
`-⟨X⁻¹, ∂X/∂x_i⟩` is assembled from the adjoint (pullback) of the
constraint map — never from a Jacobian — and the Hessian products use the
analytic second-order forms. With the `:trace` evaluation mode the
log-determinant is replaced by a stochastic estimate (see
[Matrix-free and trace-estimated evaluation](matrixfree.md)), and the
estimates carry reverse rules requiring only linear solves, so first-order
solvers can differentiate through them.

## Inner optimizer

When `IPMOptions(inner = alg)` is set, the outer loop drives `μ → 0` and
delegates each barrier subproblem to an inner `NonconvexCore` optimizer
(e.g. Ipopt), using an always-finitely-extended barrier objective (violated
constraints carry a hard quadratic penalty). The inner-optimizer path
terminates when the inner solver succeeds at `μ ≤ μ_min` with a
primal-feasible point — the subproblem solution is `O(μ_min)` from the
original optimum.
