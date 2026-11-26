# Globalization helpers for the primal barrier form: the line search is a
# plain backtracking (Armijo) on the barrier objective `φ_μ`, whose Newton
# direction is a descent direction by construction.

# Effective evaluation mode (trace → dense in the final phase when enabled).
function _eff_mode(mode::Symbol, mu::Float64, opts::IPMOptions)
    return (mode === :trace && opts.trace_final_exact && mu <= 1e-6) ? :dense : mode
end
