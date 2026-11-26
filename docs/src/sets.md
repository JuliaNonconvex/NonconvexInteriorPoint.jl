# Set constraints

`IPMAlg` supports the generalized inequality / set constraints of
NonconvexCore through a barrier-oracle protocol. Sets are declared with
`add_set_constraint!`:

```julia
m = Model(x -> sum(x .^ 2))
addvar!(m, [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])
add_set_constraint!(m, x -> x, Simplex(3))
r = optimize(m, IPMAlg(), [0.5, 0.3, 0.2])
```

## Supported sets

Order sets (`GreaterThan`, `LessThan`, `EqualTo`, `Interval`), `Zeros`,
`Nonnegatives`/`Nonpositives` and `Simplex` are converted to classical
(possibly linear, see below) equality/inequality constraints at model build
time and enter the barrier as scalars:

| Set | Conversion |
|:----|:-----------|
| `EqualTo(v)` | equality `f(x) - v = 0` |
| `Zeros` | equality `f(x) = 0` |
| `GreaterThan(l)` | inequality `f(x) - l ≥ 0` |
| `LessThan(u)` | inequality `f(x) - u ≤ 0` |
| `Interval(l, u)` | both inequalities |
| `Nonnegatives` | inequality `f(x) ≥ 0` |
| `Nonpositives` | inequality `f(x) ≤ 0` |
| `Simplex(d)` | `f(x) ≥ 0` and `sum(f(x)) = 1` |
| `SetProduct` | the conversions of its components |
| `PSDCone(d)` | semidefinite barrier oracle (below) |
| `Reals` | no constraint |

## Semidefinite constraints

`PSDCone` constraints require a strictly feasible (positive definite)
initial point — `IPMAlg` throws an informative `ArgumentError` otherwise.
The PSD barrier is implemented by the optional
[NonconvexSemidefinite.jl](https://github.com/JuliaNonconvex/NonconvexSemidefinite.jl)
package; loading it enables the full oracle suite.

```julia
using NonconvexInteriorPoint, NonconvexSemidefinite

m = Model(x -> -logdet([x[1] x[2]; x[2] x[3]]) + 0.5 * sum((x .- [2.0, 0.5, 2.0]) .^ 2))
addvar!(m, [0.1, -Inf, 0.1], [Inf, Inf, Inf])
add_set_constraint!(m, x -> [x[1] x[2]; x[2] x[3]], PSDCone(2))
r = optimize(m, IPMAlg(), [2.5, 0.5, 2.5])
```

### Barrier types

The semidefinite barrier type is selected with
[`BarrierSelection`](@ref)'s `sd_type`:

- `:logdet`: `-log det X` (the default) — the standard PSD barrier;
- `:inverse`: `tr(X⁻¹)`;
- `:inverse_square`: `tr(X⁻²)/2`.

The non-log-determinant types use analytic gradients and Hessians (no
second-order AD through the factorization) and require the dense oracle.
The matrix-free and trace oracles support `:logdet`.

### Evaluation modes

Each PSD constraint can be evaluated densely, matrix-free, or with
randomized trace estimation, selected per constraint through NonconvexCore
labels — see [Matrix-free and trace-estimated evaluation](matrixfree.md).
