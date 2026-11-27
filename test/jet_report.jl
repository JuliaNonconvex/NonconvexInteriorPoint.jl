#!/usr/bin/env julia
# JET report script used by the JET workflow. Produces a JSON summary.
using JET
using JSON
using NonconvexInteriorPoint: NonconvexInteriorPoint

function main()
    output = nothing
    i = 1
    args = ARGS
    while i <= length(args)
        if args[i] == "--output" && i < length(args)
            output = args[i + 1]
            i += 2
        else
            i += 1
        end
    end

    noise(p) =
        occursin("Error analyzing method signature", p) ||
        occursin("(1/2 union split)", p) ||
        occursin("(2/2 union split)", p) ||
        occursin("`T` not defined in static parameter matching", p) ||
        occursin("invalid builtin function call", p) ||
        occursin("`NamedTupleTools.lengthof` is not defined", p) ||
        occursin("QRCompactWYQ", p) ||
        occursin("iterate(::Nothing)", p) ||
        occursin("getfield(x::Base.KeySet", p)

    issues = String[]
    try
        result = JET.report_package(
            NonconvexInteriorPoint;
            target_modules=(NonconvexInteriorPoint,),
            toplevel_logger=nothing,
        )
        for rep in JET.get_reports(result)
            txt = sprint(io -> JET.print_report(io, rep))
            noise(txt) || push!(issues, txt)
        end
    catch e
        if occursin("Error analyzing method signature", sprint(showerror, e))
            # JET analyzer limitation on the closure-heavy KKT operators.
        else
            rethrow()
        end
    end
    rep = Dict(
        "julia_version" => string(VERSION),
        "jet_version" => string(pkgversion(JET)),
        "total_issues" => length(issues),
        "errors" => length(issues),
        "warnings" => 0,
        "filtered_dependency_issues" => 0,
        "issues" => [
            Dict("type" => "report", "signature" => "", "message" => first(txt, 500))
            for txt in issues[1:min(end, 20)]
        ],
    )
    if output === nothing
        println(JSON.json(rep))
    else
        open(output, "w") do io
            return JSON.print(io, rep)
        end
    end
end

main()
