# Automatic differentiation helpers used by the implicit (matrix-free) path.

"""
    hvp(f, x, v)

Hessian-vector product `∇²f(x) v` of the scalar-valued function `f`, computed
with DifferentiationInterface's second-order machinery (forward-over-reverse,
`SecondOrder(AutoForwardDiff(), backend)`) using NonconvexCore's configured
AD backend (`AutoZygote` by default). Falls back to the dense Hessian (and
then to zeros) when the second-order machinery fails.
"""
function hvp(f::Function, x::AbstractVector, v::AbstractVector)
    try
        backend = _second_order_backend()
        prep = DI.prepare_hvp(f, backend, x, (v,))
        return DI.hvp(f, prep, backend, x, (v,))[1]
    catch
        # Fallback: dense Hessian times vector via the second-order backend
        # of NonconvexCore (constant functions give a zero Hessian).
        try
            H = value_hessian(f, x)
            return H * v
        catch
            return zeros(eltype(x), length(x))
        end
    end
end

# Forward-over-reverse second-order backend with NonconvexCore's configured
# reverse backend (AutoZygote by default).
_second_order_backend() = DI.SecondOrder(ADTypes.AutoForwardDiff(), get_ad_backend())

# In-place accumulate: y .+= ∇²f(x) v
function hvp!(y::AbstractVector, f::Function, x::AbstractVector, v::AbstractVector)
    h = hvp(f, x, v)
    y .+= h
    return y
end

"""
    jacprod(f, x, v)

Jacobian-vector product `J_f(x) v` of the vector-valued function `f`.
DifferentiationInterface's Zygote backend does not support pushforwards
(reverse-mode only), so this stays on Zygote's own forward mode — the only
JVP available without materializing the dense Jacobian.
"""
function jacprod(f::Function, x::AbstractVector, v::AbstractVector)
    J = Zygote.forward_jacobian(f, x)[2]
    Jm = reshape(J, :, length(x))
    return vec(Jm * v)
end

"""
    jtprod(f, x, w)

Transposed-Jacobian-vector product `J_f(x)' w` of the vector-valued function
`f`, computed through NonconvexCore's reverse-mode wrapper (which handles
`nothing` gradients of constant functions).
"""
function jtprod(f::Function, x::AbstractVector, w::AbstractVector)
    _, gr = value_gradient(z -> dot(w, f(z)), x)
    return gr === nothing ? zeros(eltype(x), length(x)) : gr
end

# In-place accumulate: y .+= J_f(x)' w
function jtprod!(y::AbstractVector, f::Function, x::AbstractVector, w::AbstractVector)
    g = jtprod(f, x, w)
    y .+= g
    return y
end
