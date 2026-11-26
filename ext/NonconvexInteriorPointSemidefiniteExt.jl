module NonconvexInteriorPointSemidefiniteExt

using NonconvexInteriorPoint
using NonconvexCore: PSDCone, LinearFunction
using NonconvexSemidefinite

import NonconvexInteriorPoint:
    _psd_barrier,
    _psd_barrier_grad!,
    _psd_barrier_hess!,
    _psd_barrier_hessprod!,
    _sd_barrier,
    _sd_barrier_grad!,
    _sd_barrier_hess!,
    _sd_barrier_hessprod!

# Wire NonconvexSemidefinite's PSD barrier oracles (dense, matrix-free and
# trace-estimated) into the IPM barrier protocol. The oracles live in
# NonconvexSemidefinite (semidefinite code stays in that package) and were
# added there in the `ipm_oracles` feature; feature-detect so this extension
# remains loadable against older releases.
if isdefined(NonconvexSemidefinite, :PSDBarrierOracle)
    function _oracle(c, set::PSDCone, mode::Symbol, opts)
        logdet_est =
            (f!, n, rng) -> NonconvexSemidefinite.logdet_estimate(
                f!,
                n;
                method=opts.trace_method,
                nprobes=opts.trace_nprobes,
                lanczos_deg=opts.trace_lanczos_deg,
                cg_tol=opts.trace_cg_tol,
                rng=rng,
            )
        return NonconvexSemidefinite.PSDBarrierOracle(
            Symbol(mode),
            c,
            set.dim,
            opts.trace_nprobes,
            opts.trace_lanczos_deg,
            opts.trace_cg_tol,
            opts.rng,
            logdet_est,
        )
    end

    for mode in (:dense, :matrixfree, :trace)
        mode_v = QuoteNode(mode)
        @eval function _psd_barrier(c, set::PSDCone, x, ::Val{$mode_v}, opts)
            return NonconvexSemidefinite.psd_barrier_value(
                _oracle(c, set, $(QuoteNode(mode)), opts), x
            )
        end
        @eval function _psd_barrier_grad!(g, c, set::PSDCone, x, ::Val{$mode_v}, opts)
            NonconvexSemidefinite.psd_barrier_grad!(
                _oracle(c, set, $(QuoteNode(mode)), opts), x, g
            )
            return g
        end
        @eval function _psd_barrier_hessprod!(
            y, c, set::PSDCone, x, w, ::Val{$mode_v}, opts
        )
            NonconvexSemidefinite.psd_hess_prod!(
                _oracle(c, set, $(QuoteNode(mode)), opts), x, w, y
            )
            return y
        end
        # type-aware hooks: non-log-determinant sd barrier types
        @eval function _sd_barrier(c, set::PSDCone, x, ::Val{$mode_v}, type::Symbol, opts)
            return NonconvexSemidefinite.psd_barrier_value(
                _oracle(c, set, $(QuoteNode(mode)), opts), x, type
            )
        end
        @eval function _sd_barrier_grad!(
            g, c, set::PSDCone, x, ::Val{$mode_v}, type::Symbol, opts
        )
            NonconvexSemidefinite.psd_barrier_grad!(
                _oracle(c, set, $(QuoteNode(mode)), opts), x, g, type
            )
            return g
        end
        @eval function _sd_barrier_hessprod!(
            y, c, set::PSDCone, x, w, ::Val{$mode_v}, type::Symbol, opts
        )
            NonconvexSemidefinite.psd_hess_prod!(
                _oracle(c, set, $(QuoteNode(mode)), opts), x, w, y, type
            )
            return y
        end
    end
    function _psd_barrier_hess!(H, c, set::PSDCone, x, ::Val{:dense}, opts)
        NonconvexSemidefinite.psd_hess_dense!(_oracle(c, set, :dense, opts), x, H)
        return H
    end
    function _sd_barrier_hess!(H, c, set::PSDCone, x, ::Val{:dense}, type::Symbol, opts)
        NonconvexSemidefinite.psd_hess_dense!(_oracle(c, set, :dense, opts), x, H, type)
        return H
    end
end

end # module
