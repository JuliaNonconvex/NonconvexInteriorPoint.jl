# Randomized trace estimation utilities: Hutchinson and Hutch++ (Nyström).
# The matrix-free log-determinant estimators (SLQ, Chebyshev, rational, ...)
# live in NonconvexSemidefinite.jl (`logdet_estimate`); the `:trace`
# evaluation mode of the semidefinite barriers consumes them through the
# package extension.

"""
    hutchinson_trace(f!, n, k, rng)

Estimate `tr(A)` where `f!(y, x)` computes `y = A * x` in place, using `k`
Rademacher probes.
"""
function hutchinson_trace(f!, n::Int, k::Int, rng::Random.AbstractRNG)
    z = zeros(n)
    y = Vector{Float64}(undef, n)
    s = 0.0
    for _ in 1:k
        _rademacher!(z, rng)
        f!(y, z)
        s += dot(z, y)
    end
    return s / k
end

"""
    hutchpp_trace(f!, n, k, rng)

Estimate `tr(A)` with the Hutch++ (Nyström + Hutchinson) estimator using `k`
sketch columns and `k` residual probes.
"""
function hutchpp_trace(f!, n::Int, k::Int, rng::Random.AbstractRNG)
    k = min(k, n)
    Ω = randn(rng, n, k)
    Y = Matrix{Float64}(undef, n, k)
    for j in 1:k
        f!(view(Y, :, j), view(Ω, :, j))
    end
    Q = Matrix(qr(Y).Q)
    # tr(Q' A Q)
    AQ = Matrix{Float64}(undef, n, k)
    for j in 1:k
        f!(view(AQ, :, j), view(Q, :, j))
    end
    t_nystrom = sum(Q[:, j] ⋅ AQ[:, j] for j in 1:k)
    # residual Hutchinson on A - Q Q' A Q Q'
    z = zeros(n)
    y = Vector{Float64}(undef, n)
    w = Vector{Float64}(undef, n)
    s = 0.0
    for _ in 1:k
        _rademacher!(z, rng)
        f!(y, z)
        mul!(w, Q, Q' * y)
        s += dot(z, y .- w)
    end
    return t_nystrom + s / k
end

function _rademacher!(z::AbstractVector{Float64}, rng::Random.AbstractRNG)
    @inbounds for i in eachindex(z)
        z[i] = rand(rng, Bool) ? 1.0 : -1.0
    end
    return z
end
