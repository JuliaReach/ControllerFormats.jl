"""
    DenseLayerOp{F, M<:AbstractMatrix, B} <: AbstractLayerOp

A dense layer operation is an affine map followed by an activation function.

### Fields

- `weights`    -- weight matrix
- `bias`       -- bias vector
- `activation` -- activation function

### Notes

Conversion from a `Flux.Dense` is supported.
"""
struct DenseLayerOp{F,M<:AbstractMatrix,B} <: AbstractLayerOp
    weights::M
    bias::B
    activation::F

    function DenseLayerOp(weights::M, bias::B, activation::F;
                          validate=Val(true)) where {F,M<:AbstractMatrix,B}
        if validate isa Val{true} && !_isconsistent(weights, bias)
            throw(ArgumentError("inconsistent dimensions of weights " *
                                "($(size(weights, 1))) and bias ($(length(bias)))"))
        end

        return new{F,M,B}(weights, bias, activation)
    end
end

function _isconsistent(weights, bias)
    return size(weights, 1) == length(bias)
end

(l::DenseLayerOp)(x) = l.activation.(l.weights * x .+ l.bias)

Base.length(L::DenseLayerOp) = length(L.bias)

function Base.:(==)(L1::DenseLayerOp, L2::DenseLayerOp)
    return L1.weights == L2.weights &&
           L1.bias == L2.bias &&
           L1.activation == L2.activation
end

dim_in(L::DenseLayerOp) = size(L.weights, 2)

dim_out(L::DenseLayerOp) = length(L.bias)

function load_Flux_convert_layer()
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
