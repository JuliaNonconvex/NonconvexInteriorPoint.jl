# Matrix-free and trace-estimated evaluation

Semidefinite (and other matrix-valued) constraints can be evaluated without
ever assembling dense matrices, selected per constraint through
NonconvexCore's label mechanism (`flags`) and
[`IPMEvaluationSelection`](@ref):

| Mode | What is computed | Requires |
|:-----|:-----------------|:---------|
| `:dense` | value, gradient, Hessian products via a Cholesky factorization | dense assembly |
| `:matrixfree` | value and products through iterative linear solvers only | matrix-vector products |
| `:trace` | randomized log-determinant estimates + implicit products | matrix-vector products, an RNG |

```julia
m = Model(x -> -sum(abs2, x))  # placeholder objective
addvar!(m, fill(-1.0, 10), fill(1.0, 10))
add_set_constraint!(
    m,
    x -> Symmetric(Matrix{Float64}(I, 10, 10) + x * x'),  # assembled only as products
    PSDCone(10);
    flags = Set([:sparse_sdp]),
)
options = IPMOptions(
    evaluation = IPMEvaluationSelection(trace = [:sparse_sdp]),
    trace_method = :slq, trace_nprobes = 30, trace_lanczos_deg = 40,
    rng = Random.MersenneTwister(0),
)
r = optimize(m, IPMAlg(), zeros(10); options = options)
```

An explicit label selection dominates `matrixfree_all`/`trace_all`; `trace`
dominates `matrixfree`; unselected constraints use the dense strategy. When
any constraint is matrix-free or trace-estimated, `linear_solver = :auto`
selects the matrix-free (Krylov) KKT solver, which needs only Hessian-vector
products — no KKT matrix is assembled.

## Log-determinant estimation

The trace-estimated log-determinant barriers live in
[NonconvexSemidefinite.jl](https://github.com/JuliaNonconvex/NonconvexSemidefinite.jl)
(`NonconvexSemidefinite.logdet_estimate`), and `trace_method` selects the
estimator for `:trace`-labeled PSD constraints:

| `trace_method` | Estimator | Notes |
|:---------------|:----------|:------|
| `:slq` | stochastic Lanczos quadrature (Gauss quadrature of `log`) | default; requires positive definiteness on the Krylov spaces |
| `:chebyshev` | Chebyshev expansion of `log` | needs spectral bounds `lmin`, `lmax` |
| `:hutchpp` | Hutch++-style: Nyström part + deflated-range SLQ | low-rank plus tail |
| `:rational` | rational approximation (`log t = Σ wₖ/(1+tₖ) - ...`) via Gauss-Laguerre | needs shifted solves; coarse on wide spectra |
| `:taylor` | Taylor expansion of `log` | needs spectral bounds |
| `:integration` | integral identity `log x = ∫₀¹ (x-1)/(1+t(x-1)) dt` via Gauss-Legendre | needs shifted solves |
| `:lowrank` | Lanczos low-rank approximation of `A - I` | for near-identity matrices |

Every estimator evaluates only matrix-vector products and shifted linear
solves (conjugate gradients); no factorization is required. The estimators
`slq_logdet`, `chebyshev_logdet`, `hutchpp_logdet`, `rational_logdet`,
`taylor_logdet`, `trace_integration_logdet`, `lowrank_logdet` and
`logdet_estimate_of` are exported by NonconvexSemidefinite, together with
the spectral-bound helper `lanczos_bounds`.

### Differentiating the estimators

The solve-based estimators (`:slq`, `:hutchpp`, `:rational`,
`:integration`, `:lowrank`) are differentiable: their reverse rules
(ChainRules `rrule`s) require only the already-computed linear solves —
never a factorization — so `logdet_estimate_of` composes inside Zygote
gradients. The polynomial methods' (`:chebyshev`, `:taylor`) reverse rules
are not implemented yet (their values remain available); differentiating
them throws an informative `ArgumentError`.

```julia
using NonconvexSemidefinite, Zygote
c = x -> [x[1] x[2]; x[2] x[3]]
x = [4.0, 1.0, 3.0]
g = Zygote.gradient(
    z -> NonconvexSemidefinite.logdet_estimate_of(
        c, z; method = :slq, nprobes = 60, rng = Random.MersenneTwister(0)
    ),
    x,
)[1]
```
