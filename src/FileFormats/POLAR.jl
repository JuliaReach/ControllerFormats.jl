"""
    read_POLAR(filename::String)

Read a neural network stored in POLAR format.

### Input

- `filename` -- name of the POLAR file

### Output

A [`FeedforwardNetwork`](@ref).

### Notes

The POLAR format uses the same parameter format as Sherlock (see
[`read_Sherlock`](@ref)) but allows for general activation functions.

In addition, the last two lines are:
```
0.0
1.0
```

The reference parser and writer can be found
[here](https://github.com/ChaoHuang2018/POLAR_Tool/blob/8df333a59321f45227dafc87c367779783b6166c/POLAR/neuralnetwork.py).
"""
function read_POLAR(filename::String)
    function read_activations(io, n_layer_ops)
        activations = [available_activations[readline(io)] for _ in 1:n_layer_ops]
        return i -> activations[i]
    end

    layer_type = DenseLayerOp{<:ActivationFunction,Matrix{Float32},Vector{Float32}}

    return _read_Sherlock_POLAR(filename, read_activations, layer_type, _read_end_POLAR)
end

function _read_end_POLAR(io)
    line1 = readline(io)
    line2 = readline(io)
    if line1 != "0.0" || line2 != "1.0"
        throw(ArgumentError("the POLAR format requires to end with two lines " *
                            "containing `0.0` and `1.0`"))
    end
end

"""
    write_POLAR(N::FeedforwardNetwork, filename::String)

Write a neural network to a file in POLAR format.

### Input

- `N`        -- feedforward neural network
- `filename` -- name of the output file

### Output

`nothing`. The network is written to the output file.
"""
function write_POLAR(N::FeedforwardNetwork, filename::String)
    return _write_Sherlock_POLAR(N, filename, _write_activation_POLAR, _write_end_POLAR)
end

function _write_activation_POLAR(io, layer)
    act = get(activations_POLAR, layer.activation, nothing)
    if isnothing(act)
        throw(ArgumentError("unsupported activation function `$(typeof(layer.activation))`"))
    end
    println(io, act)
    return nothing
end

function _write_end_POLAR(io)
    println(io, "0.0")
    println(io, "1.0")
    return nothing
end

const activations_POLAR = Dict(Architecture._id => "Affine",
                               Architecture._relu => "ReLU",
                               Architecture._sigmoid => "sigmoid",
                               Architecture._tanh => "tanh")
