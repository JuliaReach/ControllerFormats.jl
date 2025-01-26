module FluxExt

using ControllerFormats.Architecture
using ControllerFormats.Architecture: _id, _relu, _sigmoid

@static if isdefined(Base, :get_extension)
    import Flux
else
    import ..Flux
end

# activation functions
activations_Flux = Dict(Flux.identity => _id,
                        _id => Flux.identity,
                        Flux.relu => _relu,
                        _relu => Flux.relu,
                        Flux.sigmoid => _sigmoid,
                        _sigmoid => Flux.sigmoid)

# convertion between dense layers

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

# convertion between convolutional layers

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

# conversion between neural networks

function Base.convert(::Type{FeedforwardNetwork}, chain::Flux.Chain)
    layers = [convert(DenseLayerOp, layer) for layer in chain.layers]
    return FeedforwardNetwork(layers)
end

function Base.convert(::Type{Flux.Chain}, net::FeedforwardNetwork)
    layers = [convert(Flux.Dense, layer) for layer in net.layers]
    return Flux.Chain(layers)
end

end  # module
