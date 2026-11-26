# Result

`optimize(m, IPMAlg(), x0)` returns an [`IPMResult`](@ref) (a subtype of
`NonconvexCore.AbstractResult`):

```julia
r = optimize(m, IPMAlg(), x0)
r.minimizer   # best primal solution found
r.minimum     # objective value at r.minimizer
r.status      # Symbol summarizing the outcome
r.fcalls      # number of objective/constraint evaluations performed
r.iterations  # number of interior-point iterations run
```

## Statuses

- `:converged` — the termination criterion is satisfied: `μ` is at (or
  below) `mu_min` and the primal feasibility and scaled stationarity
  residuals are within `tol` (or the practical endgame criterion applies,
  see [Theory](theory.md)).
- `:max_iter` — the iteration or time limit was reached first.
- `:diverged` — the line search or KKT solve failed, or the iterates
  diverged.

Check the status before trusting the point:

```julia
if r.status == :converged
    @show r.minimizer
else
    @warn "IPMAlg stopped with status $(r.status)" r.minimizer
end
```
