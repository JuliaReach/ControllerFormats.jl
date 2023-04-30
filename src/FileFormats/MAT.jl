"""
    read_MAT(filename::String; act_key::String)

Read a neural network stored in MATLAB's
[`MAT`](https://www.mathworks.com/help/matlab/import_export/load-parts-of-variables-from-mat-files.html)
format. This function requires to load the
[`MAT.jl` library](https://github.com/JuliaIO/MAT.jl).

### Input

- `filename` -- name of the `MAT` file
- `act_key`  -- key used for the activation functions
- `net_key`  -- (optional; default: `nothing`) key used for the neural network

### Output

A [`FeedforwardNetwork`](@ref).

### Notes

The `MATLAB` file encodes a dictionary.
If `net_key` is given, then the dictionary contains another dictionary under
this key.
Otherwise the outer dictionary directly contains the following:

- A vector of weight matrices (under the name `"W"`)
- A vector of bias vectors (under the name `"b"`)
- A vector of strings for the activation functions (under the name passed via
  `act_key`)
"""
function read_MAT(filename::String; act_key::String,
                  net_key::Union{String, Nothing}=nothing)
    require(@__MODULE__, :MAT; fun_name="read_MAT")

    # read data as a Dict
    data = matread(filename)

    # unwrap potential inner dictionary
    if !isnothing(net_key)
        data = data[net_key]
    end

    # read data
    !haskey(data, "W") && throw(ArgumentError("could not find key `'W'`"))
    !haskey(data, "b") && throw(ArgumentError("could not find key `'b'`"))
    weights_vec = data["W"]
    bias_vec = data["b"]
    act_vec = data[act_key]
    n_layer_ops = length(bias_vec)  # number of layer operations

    T = DenseLayerOp{<:ActivationFunction, Matrix{Float64}, Vector{Float64}}
    layers = Vector{T}(undef, n_layer_ops)

    for i in 1:n_layer_ops
        # weights
        W = _mat(weights_vec[i])

        # bias
        b = _vec(bias_vec[i])

        # activation function
        act = available_activations[act_vec[i]]

        layers[i] = DenseLayerOp(W, b, act)
    end

    return FeedforwardNetwork(layers)
end

# convert to a Vector
_vec(A::Vector) = A
_vec(A::AbstractMatrix) = vec(A)
_vec(A::Number) = [A]

# convert to a Matrix
_mat(A::Matrix) = A
_mat(A::Number) = hcat(A)
function _mat(A::Array{<:Number, 4})
    # weights are sometimes stored as a multi-dimensional array with two flat
    # dimensions
    s = size(A)
    if s[3] == 1 && s[4] == 1
        return reshape(A, s[1], s[2])
    else
        throw(ArgumentError("unexpected dimension of the weights matrix: $s"))
    end
end
