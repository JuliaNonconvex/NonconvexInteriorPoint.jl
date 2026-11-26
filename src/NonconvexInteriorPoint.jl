module NonconvexInteriorPoint

using Reexport
@reexport using NonconvexCore
using NonconvexCore:
    AbstractOptimizer,
    AbstractModel,
    AbstractResult,
    AbstractSet,
    AbstractFunction,
    FunctionInSet,
    PSDCone,
    getobjective,
    geteqconstraints,
    getineqconstraints,
    getsdconstraints,
    getsetconstraints,
    getmin,
    getmax,
    getinit,
    getdim,
    getfunction,
    getfunctions,
    LinearFunction,
    value_gradient,
    value_jacobian,
    value_hessian,
    set_violation,
    components,
    ConstraintCapabilities,
    solver_capabilities,
    VecModel
import NonconvexCore: Workspace, optimize!
using LinearAlgebra
using SparseArrays
using Random
using Zygote: Zygote
using Krylov: Krylov
using DifferentiationInterface: DifferentiationInterface
using ADTypes: ADTypes

const DI = DifferentiationInterface

export IPMAlg, IPMOptions, IPMResult, IPMEvaluationSelection, BarrierSelection

include("options.jl")
include("result.jl")
include("ad.jl")
include("model.jl")
include("barrier.jl")
include("trace_estimation.jl")
include("kkt.jl")
include("line_search.jl")
include("slacked.jl")
include("workspace.jl")

end # module
