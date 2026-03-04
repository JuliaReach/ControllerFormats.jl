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
        if !isnothing(prev) &&
           ((!isnothing(dim_in(l)) && !isnothing(dim_out(prev)) && dim_in(l) != dim_out(prev)) ||
            !_iscompatible(size(prev), size(l)))
            return i
        end
        prev = l
    end
    return 0
end

function _iscompatible(t1::Tuple, t2::Tuple)
    return _iscompatible(t1[2], t2[1])
end

function _iscompatible(i::Int, j::Int)
    return i == j
end

function _iscompatible(::Any, ::Nothing)
    return true
end

layers(N::FeedforwardNetwork) = N.layers

function Base.show(io::IO, N::FeedforwardNetwork)
    str = "$FeedforwardNetwork with $(dim_in(N)) inputs, " *
          "$(dim_out(N)) outputs, and $(length(N)) layers:"
    for l in layers(N)
        str *= "\n- $l"
    end
    return print(io, str)
end
