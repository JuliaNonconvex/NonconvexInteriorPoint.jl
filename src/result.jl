"""
    IPMResult

Result returned by `optimize` with [`IPMAlg`](@ref).

## Fields

- `minimizer`: the best primal solution found,
- `minimum`: the objective value at `minimizer`,
- `status`: `Symbol` summarizing the outcome — `:converged`, `:max_iter` or
  `:diverged`,
- `fcalls`: number of objective/constraint evaluations performed,
- `iterations`: number of interior-point iterations run.
"""
struct IPMResult{TM1,TM2} <: AbstractResult
    minimizer::TM1
    minimum::TM2
    status::Symbol
    fcalls::Int
    iterations::Int
end
