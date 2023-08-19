"""
    read_NNet(filename::String)

Read a neural network stored in [`NNet`](https://github.com/sisl/NNet) format.

### Input

- `filename` -- name of the `NNet` file

### Output

A [`FeedforwardNetwork`](@ref).

### Notes

The format assumes that all layers but the output layer use ReLU activation (the
output layer uses the identity activation).

The format looks like this (each line may optionally be terminated by a comma):

1. Header text, each line beginning with "//"
2. Comma-separated line with four values:
   number of layer operations, number of inputs, number of outputs, maximum
   layer size
3. Comma-separated line with the layer sizes
4. Flag that is no longer used
5. Minimum values of inputs
6. Maximum values of inputs
7. Mean values of inputs and one value for all outputs
8. Range values of inputs and one value for all outputs
9. Blocks of lines describing the weight matrix and bias vector for a layer;
   each matrix row is written as a comma-separated line, and each vector entry
   is written in its own line

The code follows [this implementation](https://github.com/sisl/NeuralVerification.jl/blob/957cb32081f37de57d84d7f0813f708288b56271/src/utils/util.jl#L10).
"""
function read_NNet(filename::String)
    layers = nothing
    open(filename, "r") do io
        line = readline(io)

        # skip header text
        while startswith(line, "//")
            line = readline(io)
        end

        # four numbers: only the first (number of layer operations) is relevant
        n_layer_ops = parse(Int, split(line, ",")[1])

        # layer sizes
        layer_sizes = parse.(Int, split(readline(io), ",")[1:(n_layer_ops + 1)])

        # five lines of irrelevant information
        for i in 1:5
            line = readline(io)
        end

        # read layers except for the output layer (with ReLU activation)
        T = DenseLayerOp{<:ActivationFunction,Matrix{Float32},Vector{Float32}}
        layers = T[_read_layer_NNet(io, dim, Architecture._relu)
                   for dim in layer_sizes[2:(end - 1)]]

        # read output layer (with identity activation)
        return push!(layers, _read_layer_NNet(io, last(layer_sizes), Architecture._id))
    end

    return FeedforwardNetwork(layers)
end

# some complication because lines can optionally be terminated by a comma
function _read_layer_NNet(io::IOStream, output_dim::Int, act)
    # simple parsing as a Vector of Vectors
    weights = [parse.(Float32, filter(!isempty, split(readline(io), ","))) for _ in 1:output_dim]
    weights = vcat(weights'...)
    bias = [parse(Float32, split(readline(io), ",")[1]) for _ in 1:output_dim]
    return DenseLayerOp(weights, bias, act)
end

"""
    write_NNet(N::FeedforwardNetwork, filename::String)

Write a neural network to a file in [`NNet`](https://github.com/sisl/NNet)
format.

### Input

- `N`        -- feedforward neural network
- `filename` -- name of the output file

### Output

`nothing`. The network is written to the output file.

### Notes

The NNet format assumes that all layers but the output layer use ReLU activation
(the output layer uses the identity activation).

Some non-important part of the output (such as the input domain) is not
correctly written and instead set to `0`.

See [`read_NNet`](@ref) for the documentation of the format.
"""
function write_NNet(N::FeedforwardNetwork, filename::String)
    n_layer_ops = length(N.layers)
    n_inputs = dim_in(N)
    n_outputs = dim_out(N)
    n_neurons_max = max(maximum(dim_in, N.layers), n_outputs)
    open(filename, "w") do io
        print(io, string(n_layer_ops), ", ")  # number of layer operations
        print(io, string(n_inputs), ", ")  # number of neurons in input layer
        print(io, string(n_outputs), ", ")  # number of neurons in output layer
        println(io, string(n_neurons_max))  # maximum number of neurons per layer

        # layer sizes
        print(io, string(n_inputs))
        @inbounds for l in N.layers
            print(io, ", ", string(dim_out(l)))
        end
        println(io)

        # five lines of irrelevant information
        for i in 1:5
            println(io, "0")
        end

        # one line for each weight and bias of the hidden and output layers
        @inbounds for (i, layer) in enumerate(N.layers)
            if i == n_layer_ops
                @assert layer.activation isa Id "the NNet format requires an Id activation in " *
                                                "the last layer, but the network contains a " *
                                                "`$(typeof(layer.activation))` activation"
            else
                @assert layer.activation isa ReLU "the NNet format requires ReLU activations " *
                                                  "everywhere but in the last layer, but the " *
                                                  "network contains a " *
                                                  "`$(typeof(layer.activation))` activation"
            end

            _write_layer_NNet(io, layer)
        end
    end
    return nothing
end

# each matrix row is written as a comma-separated line; each vector entry is
# written in its own line; activations are not written
function _write_layer_NNet(io::IOStream, layer)
    W = layer.weights
    m, n = size(W)
    @inbounds for i in 1:m
        print(io, W[i, 1])
        for j in 2:n
            print(io, ", ", W[i, j])
        end
        println(io)
    end
    b = layer.bias
    @inbounds for bi in b
        println(io, bi)
    end
end
