"""
    ConvolutionalLayerOp{F, M, B} <: AbstractLayerOp

A convolutional layer operation is a series of filters, each of which computes a
small affine map followed by an activation function.

### Fields

- `weights`    -- vector with one weight matrix for each filter
- `bias`       -- vector with one bias value for each filter
- `activation` -- activation function

### Notes

Conversion from a `Flux.Conv` is supported.
"""
struct ConvolutionalLayerOp{F,W,B} <: AbstractLayerOp
    weights::W
    bias::B
    activation::F

    function ConvolutionalLayerOp(weights::W, bias::B, activation::F;
                                  validate=Val(true)) where {F,W,B}
        if validate isa Val{true} && !_isconsistent_ConvolutionalLayerOp(weights, bias)
            throw(ArgumentError("inconsistent filter dimensions: weights " *
                                "($(length(weights))) and biases ($(length(bias)))"))
        end

        return new{F,W,B}(weights, bias, activation)
    end
end

function _isconsistent_ConvolutionalLayerOp(weights, bias)
    if length(weights) != length(bias)
        return false
    elseif length(bias) == 0
        return false
    end
    @inbounds begin
        s = size(first(weights))
        if length(s) != 3 || s[1] == 0 || s[2] == 0 || s[3] == 0
            return false
        end
        for e in weights
            if size(e) != s
                return false
            end
        end
    end
    return true
end

n_filters(L::ConvolutionalLayerOp) = length(L.bias)

kernel(L::ConvolutionalLayerOp) = @inbounds size(first(L.weights))

# application to a tensor
function (L::ConvolutionalLayerOp)(T)
    s = size(T)
    if length(s) != 3
        throw(ArgumentError("a convolutional layer requires at least two dimensions, but got $s"))
    end
    p, q, r = kernel(L)
    @inbounds begin
        if p > s[1] || q > s[2] || r != s[3]
            throw(ArgumentError("convolution with kernel size $(kernel(L)) " *
                                "does not apply to a tensor of dimension $s"))
        end
        d1 = s[1] - p + 1
        d2 = s[2] - q + 1
    end
    t = n_filters(L)
    s = (d1, d2, t)
    O = similar(T, s)
    @inbounds for f in 1:t
        W = L.weights[f]
        b = L.bias[f]
        for k in 1:r
            for j in 1:d2
                for i in 1:d1
                    T′ = view(T, i:(i + p - 1), j:(j + q - 1), k)
                    O[i, j, f] = L.activation(dot(W, T′) + b)
                end
            end
        end
    end
    return O
end

function Base.:(==)(L1::ConvolutionalLayerOp, L2::ConvolutionalLayerOp)
    return L1.weights == L2.weights &&
           L1.bias == L2.bias &&
           L1.activation == L2.activation
end

function Base.isapprox(L1::ConvolutionalLayerOp, L2::ConvolutionalLayerOp;
                       atol::Real=0, rtol=nothing)
    if isnothing(rtol)
        if iszero(atol)
            N = @inbounds promote_type(eltype(first(L1.weights)), eltype(first(L2.weights)),
                                       eltype(L1.bias), eltype(L2.bias))
            rtol = rtoldefault(N)
        else
            rtol = zero(atol)
        end
    end
    return isapprox(L1.weights, L2.weights; atol=atol, rtol=rtol) &&
           isapprox(L1.bias, L2.bias; atol=atol, rtol=rtol) &&
           L1.activation == L2.activation
end

function Base.show(io::IO, L::ConvolutionalLayerOp)
    str = "$(string(ConvolutionalLayerOp)) of $(n_filters(L)) filters with " *
          "kernel size $(kernel(L)) and $(typeof(L.activation)) activation"
    return print(io, str)
end

size(::ConvolutionalLayerOp) = (3, 3)

function load_Flux_convert_Conv_layer()
    return quote
        function Base.convert(::Type{ConvolutionalLayerOp}, layer::Flux.Conv)
            if !all(isone, layer.stride)
                throw(ArgumentError("stride $(layer.stride) != 1 is not supported"))  # COV_EXCL_LINE
            end
            if !all(iszero, layer.pad)
                throw(ArgumentError("pad $(layer.pad) != 0 is not supported"))  # COV_EXCL_LINE
            end
            if !all(isone, layer.dilation)
                throw(ArgumentError("dilation $(layer.dilation) != 1 is not supported"))  # COV_EXCL_LINE
            end
            if !all(isone, layer.groups)
                throw(ArgumentError("groups $(layer.groups) != 1 is not supported"))  # COV_EXCL_LINE
            end
            act = get(activations_Flux, layer.σ, nothing)
            if isnothing(act)
                throw(ArgumentError("unsupported activation function $(layer.σ)"))  # COV_EXCL_LINE
            end
            # Flux stores a 4D matrix instead of a vector of 3D matrices
            weights = @inbounds [layer.weight[:, :, :, i] for i in 1:size(layer.weight, 4)]
            return ConvolutionalLayerOp(weights, layer.bias, act)
        end

        function Base.convert(::Type{Flux.Conv}, layer::ConvolutionalLayerOp)
            act = get(activations_Flux, layer.activation, nothing)
            if isnothing(act)
                throw(ArgumentError("unsupported activation function $(layer.activation)"))  # COV_EXCL_LINE
            end
            # Flux stores a 4D matrix instead of a vector of 3D matrices
            weights = cat(layer.weights...; dims=4)
            return Flux.Conv(weights, layer.bias, act)
        end
    end
end
