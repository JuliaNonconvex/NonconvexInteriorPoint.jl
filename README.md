<p align="center">
  <img src="assets/logo.png" width="400" alt="NonconvexInteriorPoint.jl logo">
</p>

[![Documentation (stable)](https://img.shields.io/badge/docs-stable-blue.svg)](https://julianonconvex.github.io/NonconvexInteriorPoint.jl/stable)
[![Documentation (dev)](https://img.shields.io/badge/docs-dev-blue.svg)](https://julianonconvex.github.io/NonconvexInteriorPoint.jl/dev)
[![Build Status](https://github.com/JuliaNonconvex/NonconvexInteriorPoint.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaNonconvex/NonconvexInteriorPoint.jl/actions)
[![Coverage](https://codecov.io/gh/JuliaNonconvex/NonconvexInteriorPoint.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaNonconvex/NonconvexInteriorPoint.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-Blue-4473c9.svg)](https://github.com/invenia/BlueStyle)
[![JET](https://img.shields.io/badge/%F0%9F%9B%A9%EF%B8%8F_tested_with-JET.jl-233f9a)](https://github.com/aviatesk/JET.jl)

# NonconvexInteriorPoint.jl

A native interior-point method for smooth, generally nonconvex nonlinear programs with exotic constraint types.

`NonconvexInteriorPoint.jl` provides [`IPMAlg`](@ref), a native primal
barrier interior-point solver in the spirit of Ipopt, MadNLP and Uno:
regularized (inertia-corrected) KKT solves — dense or matrix-free with
Krylov methods —, backtracking on an exact-penalty merit, an adaptive
barrier update, native linear constraints, and — through NonconvexCore's
set constraints — generalized inequalities including semidefinite cones.
Inequalities and set constraints enter the barrier with adjoint-only
gradients; a `slacks = true` option switches to the Ipopt-style slacked
formulation; an optional inner optimizer can solve the barrier
subproblems; and the scalar/semidefinite barrier types are selectable
per constraint ([`BarrierSelection`]). Evaluation of each constraint can
be selected per constraint via labels: dense, matrix-free, or
trace-estimated (randomized Hutchinson, Hutch++, stochastic Lanczos
quadrature, rational, Taylor, and low-rank log-determinant estimators),
enabling large sparse SDP-constrained problems (buckling-type) without
materializing dense matrices. The design is documented in
[DESIGN.md](DESIGN.md).

`NonconvexInteriorPoint.jl` is part of the JuliaNonconvex ecosystem built on
NonconvexCore.jl under the Nonconvex.jl umbrella.

## Quick start

```julia
using NonconvexInteriorPoint

m = Model(x -> (x[1] - 1)^2 + (x[2] - 1)^2) # Rosenbrock-like objective
addvar!(m, [0.0, 0.0], [Inf, Inf])
add_ineq_constraint!(m, x -> x[1]^2 + x[2]^2 - 2) # circle inside radius sqrt(2)

r = optimize(m, IPMAlg(), [0.5, 0.5])
@show r.minimizer r.minimum r.status
```

## Documentation

- [**STABLE**](https://julianonconvex.github.io/NonconvexInteriorPoint.jl/stable) — most recently tagged version documentation.
- [**DEV**](https://julianonconvex.github.io/NonconvexInteriorPoint.jl/dev) — in-development version documentation.

## Installation

```julia
import Pkg
Pkg.add("NonconvexInteriorPoint")
```
