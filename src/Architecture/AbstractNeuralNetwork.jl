"""
    AbstractNeuralNetwork

Abstract type for neural networks.

### Notes

Subtypes should implement the following method:

- `layers(::AbstractNeuralNetwork)` - return a list of the layers

The following standard methods are implemented:

- `length(::AbstractNeuralNetwork)`
- `getindex(::AbstractNeuralNetwork, indices)`
- `lastindex(::AbstractNeuralNetwork)`
- `==(::AbstractNeuralNetwork, ::AbstractNeuralNetwork)`
"""
abstract type AbstractNeuralNetwork end

"""
    layers(N::AbstractNeuralNetwork)

Return a list of the layers of a neural network.

### Input

- `N` -- neural network

### Output

The list of layers.
"""
function layers(::AbstractNeuralNetwork) end

"""
    dim_in(N::AbstractNeuralNetwork)

Return the input dimension of a neural network.

### Input

- `N` -- neural network

### Output

The dimension of the input layer of `N`.
"""
dim_in(N::AbstractNeuralNetwork) = dim_in(first(layers(N)))

"""
    dim_out(N::AbstractNeuralNetwork)

Return the output dimension of a neural network.

### Input

- `N` -- neural network

### Output

The dimension of the output layer of `N`.
"""
dim_out(N::AbstractNeuralNetwork) = dim_out(last(layers(N)))

"""
    dim(N::AbstractNeuralNetwork)

Return the input and output dimension of a neural network.

### Input

- `N` -- neural network

### Output

The pair ``(i, o)`` where ``i`` is the input dimension and ``o`` is the output
dimension of `N`.
"""
dim(N::AbstractNeuralNetwork) = (dim_in(N), dim_out(N))

Base.length(N::AbstractNeuralNetwork) = length(layers(N))

Base.getindex(N::AbstractNeuralNetwork, i::Int) = layers(N)[i]

function Base.getindex(N::AbstractNeuralNetwork, indices::AbstractVector{Int})
    return [N[i] for i in indices]
end

Base.lastindex(N::AbstractNeuralNetwork) = length(N)

function Base.:(==)(N1::AbstractNeuralNetwork, N2::AbstractNeuralNetwork)
    return layers(N1) == layers(N2)
end

function Base.:isapprox(N1::AbstractNeuralNetwork, N2::AbstractNeuralNetwork;
                        atol::Real=0, rtol=nothing)
    if length(N1) != length(N2)
        return false
    end
    return all(isapprox.(layers(N1), layers(N2); atol=atol, rtol=rtol))
end

(N::AbstractNeuralNetwork)(x) = reduce((a1, a2) -> a2 ∘ a1, layers(N))(x)
