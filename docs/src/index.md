# NonconvexInteriorPoint

## Description

`NonconvexInteriorPoint.jl` provides [`IPMAlg`](@ref), a native primal
barrier interior-point method for smooth, generally nonconvex nonlinear
programs in the JuliaNonconvex ecosystem. It supports box bounds,
inequalities, equalities, native linear constraints with cached constant
Jacobians, and — through NonconvexCore's set constraints — generalized
inequalities including semidefinite cones, via a barrier-oracle protocol
(see [Set constraints](sets.md)). Inequalities and set constraints enter the
barrier with adjoint-only gradients (no Jacobians are materialized); only
the equality constraints contribute Jacobians to the KKT system. A
`slacks = true` option switches to the Ipopt-style slacked formulation.

A distinguishing feature is the per-constraint selection of evaluation
strategies through NonconvexCore's label (`flags`) mechanism: constraints
can be evaluated densely, matrix-free (implicit products, iterative solves),
or with randomized trace estimation (Hutchinson, Hutch++, stochastic Lanczos
quadrature, and more) for log-determinant barriers. This enables large
sparse SDP-constrained problems (e.g. buckling-constrained design) without
ever materializing dense matrices. See [Matrix-free and trace-estimated
evaluation](matrixfree.md).

The algorithm follows the Ipopt/MadNLP/Uno lineage: regularized
(inertia-corrected) KKT systems solved densely or matrix-free with Krylov
methods, backtracking on an exact-penalty merit, an adaptive barrier
update, per-constraint barrier selection with pluggable barrier types, and
an optional inner optimizer for the barrier subproblems. The design
document is
[DESIGN.md](https://github.com/JuliaNonconvex/NonconvexInteriorPoint.jl/blob/main/DESIGN.md).

See [Quick start](quickstart.md) for the first steps, [Options](options.md) for
the full option reference, [Set constraints](sets.md) for barrier oracles,
[Matrix-free and trace-estimated evaluation](matrixfree.md) for the
per-constraint evaluation modes, [Result](result.md) for the returned result
fields, [Theory](theory.md) for the algorithmic background and
[Docstrings](docstrings.md) for the API reference.

## Design influences

The solver is informed by permissively-licensed state of the art: Ipopt
(EPL-2.0), MadNLP (MIT), Uno (MIT), Hypatia (MIT), Clarabel (Apache-2.0),
COSMO (Apache-2.0), Tulip (MPL-2.0) and RipQP (MIT).
