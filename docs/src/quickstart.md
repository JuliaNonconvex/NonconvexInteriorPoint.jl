## Quick start

```julia
using NonconvexInteriorPoint

# Convex QP: min (x-1)² + (y-2)² s.t. x² + y ≤ 3, x + y ≤ 2
m = Model(x -> (x[1] - 1)^2 + (x[2] - 2)^2)
addvar!(m, [-10.0, -10.0], [10.0, 10.0])
add_ineq_constraint!(m, x -> x[1]^2 + x[2] - 3)
add_ineq_constraint!(m, x -> x[1] + x[2] - 2)

r = optimize(m, IPMAlg(), [0.0, 0.0])
@show r.minimizer r.minimum r.status
```

Linear constraints are handled natively when expressed through
`add_linear_constraint!` or detected structurally:

```julia
m = Model(x -> sum(x .^ 2))
addvar!(m, [-1.0, -1.0], [1.0, 1.0])
add_linear_constraint!(m, [1.0 1.0], LessThan(0.5))

r = optimize(m, IPMAlg(), [0.1, 0.1])
```

Semidefinite constraints use NonconvexCore's `PSDCone` set and the optional
NonconvexSemidefinite package for the barrier oracle:

```julia
using NonconvexInteriorPoint
using NonconvexSemidefinite # optional: enables the full PSD barrier oracle

m = Model(x -> x[1] + x[2])
addvar!(m, [-Inf, -Inf], [Inf, Inf])
# [x1 x2; x2 x1] ⪰ I
add_set_constraint!(m, x -> [x[1] x[2]; x[2] x[1]], PSDCone(2))
add_linear_constraint!(m, [1.0 0.0], GreaterThan(1.0))
add_linear_constraint!(m, [0.0 1.0], GreaterThan(0.0))

r = optimize(m, IPMAlg(), [2.0, 0.5])
```
