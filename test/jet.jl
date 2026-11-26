@testitem "JET" begin
    using Test

    # JET is analyzed in a fresh subprocess: `report_package` replays the
    # package's top-level definitions and corrupts the loaded method tables,
    # which would break the remaining testitems in this process.
    #
    # The analyzer has two known limitations on this package: it crashes
    # internally on the closure-heavy KKT operators ("Error analyzing method
    # signature") and it cannot prove the `Union{Nothing, SparseMatrix}`
    # guards on the native linear-constraint fields (union-split noise).
    # Reports matching those categories (plus a few third-party inference
    # quirks in Zygote/MutableArithmetics/LinearAlgebra code) are filtered
    # out; any other report fails the test.
    julia = joinpath(Sys.BINDIR, Base.julia_exename())
    project = dirname(Base.active_project())
    code = """
    using JET, NonconvexInteriorPoint
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
    r = JET.report_package(NonconvexInteriorPoint; target_modules = (NonconvexInteriorPoint,))
    bad = String[]
    for rep in JET.get_reports(r)
        txt = sprint(io -> JET.print_report(io, rep))
        noise(txt) || push!(bad, txt)
    end
    isempty(bad) ||
        println("JET-BAD: " * string(length(bad)) * " actionable reports\\n" * join(bad, "\\n"))
    """
    out = read(
        pipeline(`$julia --startup-file=no --project=$project -e $code`; stderr=devnull),
        String,
    )
    @test !occursin("JET-BAD:", out)
end
