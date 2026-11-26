# NonconvexInteriorPoint.jl

Release notes for each version of the package.

## Unreleased

### Added

- Initial release of `NonconvexInteriorPoint.jl`: a native interior-point
  method (`IPMAlg`) for smooth, generally nonconvex nonlinear programs in the
  Nonconvex ecosystem.
  - Primal barrier IPM with box bounds, inequalities, equalities, and set
    constraints via a barrier-oracle protocol; the barrier gradients are
    assembled from adjoints only (no Jacobians for inequalities or sets).
  - Ipopt-style slacked formulation through `slacks = true`, and
    per-constraint barrier selection through `BarrierSelection` (labels,
    scalar barrier types `:log`/`:inv`/`:quad`, semidefinite barrier types
    `:logdet`/`:inverse`/`:inverse_square`).
  - Native linear constraints with cached constant Jacobians.
  - Regularized (inertia-corrected) KKT systems solved densely or
    matrix-free with Krylov methods.
  - Per-constraint evaluation strategies selected via NonconvexCore labels
    (`IPMEvaluationSelection`): dense, matrix-free, and trace-estimated.
  - Randomized log-determinant estimation (stochastic Lanczos quadrature,
    Chebyshev, Hutch++, rational, Taylor, trace integration, low-rank) with
    reverse rules requiring only linear solves, implemented in
    NonconvexSemidefinite.
  - An optional inner optimizer for the barrier subproblems.
  - Label-based constraint relaxation and bound relaxation.
  - Semidefinite barrier oracles through the optional NonconvexSemidefinite
    dependency.
