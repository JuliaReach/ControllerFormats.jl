"""
    DenseLayerOp{F, M, B} <: AbstractLayerOp

A dense layer operation is an affine map followed by an activation function.

### Fields

- `weights`    -- weight matrix
- `bias`       -- bias vector
- `activation` -- activation function

### Notes

Conversion from a `Flux.Dense` is supported.
"""
struct DenseLayerOp{F,W,B} <: AbstractLayerOp
    weights::W
    bias::B
    activation::F

    function DenseLayerOp(weights::W, bias::B, activation::F;
                          validate=Val(true)) where {F,W,B}
        if validate isa Val{true} && !_isconsistent_DenseLayerOp(weights, bias)
            throw(ArgumentError("inconsistent dimensions of weights " *
                                "($(size(weights, 1))) and bias ($(length(bias)))"))
        end

        return new{F,W,B}(weights, bias, activation)
    end
end

function _isconsistent_DenseLayerOp(weights, bias)
    return size(weights, 1) == length(bias)
end

# application to a vector
(L::DenseLayerOp)(x) = L.activation.(L.weights * x .+ L.bias)

Base.length(L::DenseLayerOp) = length(L.bias)

function Base.:(==)(L1::DenseLayerOp, L2::DenseLayerOp)
    return L1.weights == L2.weights &&
           L1.bias == L2.bias &&
           L1.activation == L2.activation
end

function Base.:isapprox(L1::DenseLayerOp, L2::DenseLayerOp; atol::Real=0,
                        rtol=nothing)
    if dim_in(L1) != dim_in(L2) || dim_out(L1) != dim_out(L2)
        return false
    end
    if isnothing(rtol)
        if iszero(atol)
            N = promote_type(eltype(L1.weights), eltype(L2.weights),
                             eltype(L1.bias), eltype(L2.bias))
            rtol = Base.rtoldefault(N)
        else
            rtol = zero(atol)
        end
    end
    return isapprox(L1.weights, L2.weights; atol=atol, rtol=rtol) &&
           isapprox(L1.bias, L2.bias; atol=atol, rtol=rtol) &&
           L1.activation == L2.activation
end

function Base.show(io::IO, L::DenseLayerOp)
    str = "$DenseLayerOp with $(dim_in(L)) inputs, $(dim_out(L)) " *
          "outputs, and $(L.activation) activation"
    return print(io, str)
end

dim_in(L::DenseLayerOp) = size(L.weights, 2)

dim_out(L::DenseLayerOp) = length(L.bias)

function load_Flux_convert_Dense_layer()
    return quote
        function Base.convert(::Type{DenseLayerOp}, layer::Flux.Dense)
            act = get(activations_Flux, layer.σ, nothing)
            if isnothing(act)
                throw(ArgumentError("unsupported activation function $(layer.σ)"))
            end
            return DenseLayerOp(layer.weight, layer.bias, act)
        end

        function Base.convert(::Type{Flux.Dense}, layer::DenseLayerOp)
            act = get(activations_Flux, layer.activation, nothing)
            if isnothing(act)
                throw(ArgumentError("unsupported activation function $(layer.activation)"))
            end
            return Flux.Dense(layer.weights, layer.bias, act)
        end
    end
end
