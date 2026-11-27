# A buckling-constrained example

This page walks through the buckling-constrained truss design that the test
suite uses to compare `IPMAlg` with `AugLagAlg` — a nonconvex semidefinite
program with a single **global** buckling constraint.

## The problem

A statically indeterminate three-bar truss: the nodes `A = (0,0)`,
`B = (1,0)` and `D = (0.5,-1)` are fixed, the tip `C = (0.5,1)` is free
(two displacement degrees of freedom), and the bars `A–C`, `B–C`, `D–C`
have cross-sectional areas `x = (x₁, x₂, x₃)` and unit modulus `E = 1`. A
reference load `P = (0,-1)` acts at the tip. The design problem is

```math
minimize \quad \sum_i l_i x_i
\quad s.t. \quad S(x) := K(x) + \bar{c}\, G(x, u(x)) \succeq 0,
\qquad x \ge x_{\min},
```

where `l_i` are the bar lengths, `\bar{c} = 10` the required load factor,
and the buckling constraint is one **global** semidefinite constraint (no
eigenvalue aggregation is used).

## The global buckling LMI

The elastic stiffness matrix is the design-weighted assembly of the bar
stiffnesses,

```math
K(x) = \sum_i \frac{x_i}{l_i}
\begin{bmatrix} c_{x,i}^2 & c_{x,i} c_{y,i} \\ c_{x,i} c_{y,i} & c_{y,i}^2 \end{bmatrix},
```

and the displacement solves `K(x) u = P`. The geometric (stress) stiffness
comes from the axial forces `N_i = x_i (u_b - u_a)·c_i / l_i` — negative in
compression:

```math
G(x, u) = \sum_i \frac{N_i}{l_i}
\begin{bmatrix} c_{y,i}^2 & -c_{x,i} c_{y,i} \\ -c_{x,i} c_{y,i} & c_{x,i}^2 \end{bmatrix}.
```

The generalized eigenvalue problem `K v = λ G v` describes the buckling
modes: a negative `G` direction means the compressive stress softens the
corresponding stiffness mode, and the structure buckles when the softest
mode reaches zero — `λ_min(K, G) ≥ \bar{c} ⟺ K + \bar{c} G ⪰ 0` (with `G`
evaluated at the reference-load displacement). Because the truss is
statically indeterminate, the force distribution — and hence `G` — depends
on the design, which makes the LMI genuinely nonconvex in `x`. (A statically
determinate truss would have design-independent forces and the constraint
would collapse to a linear bound.)

The unconstrained volume minimum `x → x_min` violates the LMI: thin bars
carry the same compressive force at vanishing stiffness, so the buckling
eigenvalue is negative. Thickening the bars restores it — at the optimum
the LMI is exactly active (the cheapest design sits on the buckling
boundary).

## Solving

```julia
using NonconvexInteriorPoint, NonconvexSemidefinite

# the truss constants and the buckling constraint map
buckling_state(x) = begin
    K = foldl(+, [
        x[i] .* ([c[1]^2 c[1]*c[2]; c[1]*c[2] c[2]^2] ./ L[i]) for
        (i, c) in enumerate(dirs)
    ])
    u = K \ P
    G = foldl(+, [
        ((x[i] * dot(u, dirs[i]) / L[i]) / L[i]) .*
        [c[2]^2 -c[1]*c[2]; -c[1]*c[2] c[1]^2] for
        (i, c) in enumerate(dirs)
    ])
    return K + cbar .* G, u
end

m = Model(x -> sum(L .* x))
addvar!(m, fill(0.01, 3), fill(50.0, 3))
add_set_constraint!(m, x -> buckling_state(x)[1], PSDCone(2))

# dense oracle (default)
r = optimize(m, IPMAlg(), [25.0, 25.0, 5.0])

# matrix-free oracle (label-selected: no dense PSD assembly)
m = Model(x -> sum(L .* x))
addvar!(m, fill(0.01, 3), fill(50.0, 3))
add_set_constraint!(m, x -> buckling_state(x)[1], PSDCone(2); flags = Set([:mf]))
r = optimize(
    m, IPMAlg(), [25.0, 25.0, 5.0];
    options = IPMOptions(
        evaluation = IPMEvaluationSelection(matrixfree_all = true),
    ),
)
```

Both the dense and the matrix-free oracle converge from a strictly feasible
start (the PSD barrier requires `S(x₀) ≻ 0`) to the same design, and
`AugLagAlg` agrees:

| algorithm | minimizer | volume | `λ_min(S(x*))` |
|---|---|---|---|
| `IPMAlg` (dense) | `(22.36, 22.36, 0.01)` | `50.02` | `0.0` |
| `IPMAlg` (matrix-free) | `(22.36, 22.36, 0.01)` | `50.02` | `0.0` |
| `AugLagAlg` | `(22.36, 22.36, 0.01)` | `50.02` | `0.0` |

The symmetric optimum puts all the material into the two inclined bars
(`x₃` at its lower bound: the vertical bar carries only tension and does
not drive buckling), and the smallest buckling eigenvalue is exactly zero —
the classic "optimal design sits on the stability boundary" signature of
this problem class (Kočvara–Stingl PBM).

The full version of this example, including the comparison with
`AugLagAlg`, lives in `test/buckling.jl`. The trace-estimated oracle is
deliberately not used here: its log-determinant estimate is only meaningful
for positive definite matrices, and the iterates of this problem approach
the buckling boundary where `S` is singular.
