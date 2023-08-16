"""
    FeedforwardNetwork{L} <: AbstractNeuralNetwork

Standard implementation of a feedforward neural network which stores the layer
operations.

### Fields

- `layers` -- vector of layer operations (see [`AbstractLayerOp`](@ref))

### Notes

The field `layers` contains the layer operations, so the number of layers is
`length(layers) + 1`.

Conversion from a `Flux.Chain` is supported.
"""
struct FeedforwardNetwork{L} <: AbstractNeuralNetwork
    layers::L

    function FeedforwardNetwork(layers::L; validate=Val(true)) where {L}
        if validate isa Val{true}
            i = _first_inconsistent_layer(layers)
            i != 0 && throw(ArgumentError("inconsistent layer dimensions at " *
                                          "index $i"))
        end

        return new{L}(layers)
    end
end

function _first_inconsistent_layer(L)
    prev = nothing
    for (i, l) in enumerate(L)
        if !isnothing(prev) && dim_in(l) != dim_out(prev)
            return i
        end
        prev = l
    end
    return 0
end

layers(N::FeedforwardNetwork) = N.layers

function load_Flux_convert_network()
    return quote
        function Base.convert(::Type{FeedforwardNetwork}, chain::Flux.Chain)
            layers = [convert(DenseLayerOp, layer) for layer in chain.layers]
            return FeedforwardNetwork(layers)
        end

        function Base.convert(::Type{Flux.Chain}, net::FeedforwardNetwork)
            layers = [convert(Flux.Dense, layer) for layer in net.layers]
            return Flux.Chain(layers)
        end
    end
end
