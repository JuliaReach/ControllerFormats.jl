"""
    read_YAML(filename::String)

Read a neural network stored in [YAML](https://yaml.org/) format. This function
requires to load the [`YAML.jl` library](https://github.com/JuliaData/YAML.jl).

### Input

- `filename` -- name of the `YAML` file

### Output

A [`FeedforwardNetwork`](@ref).
"""
function read_YAML(filename::String)
    # read data as a Dict
    data = _load_YAML(filename)

    # read data
    !haskey(data, "weights") && throw(ArgumentError("could not find key `'weights'`"))
    !haskey(data, "offsets") && throw(ArgumentError("could not find key `'offsets'`"))
    !haskey(data, "activations") && throw(ArgumentError("could not find key `'activations'`"))
    weights_vec = data["weights"]
    bias_vec = data["offsets"]
    act_vec = data["activations"]
    n_layer_ops = length(bias_vec)  # number of layer operations

    T = DenseLayerOp{<:ActivationFunction,Matrix{eltype(eltype(weights_vec[1]))},
                     Vector{eltype(bias_vec[1])}}
    layers = Vector{T}(undef, n_layer_ops)

    for i in 1:n_layer_ops
        W = weights_vec[i]
        W = Matrix(reduce(hcat, W)')
        b = bias_vec[i]
        act = available_activations[act_vec[i]]
        layers[i] = DenseLayerOp(W, b, act)
    end

    return FeedforwardNetwork(layers)
end

# defined in `YAMLExt.jl`
function _load_YAML(filename)
    mod = isdefined(Base, :get_extension) ? Base.get_extension(@__MODULE__, :YAMLExt) : @__MODULE__
    require(mod, :YAML; fun_name="read_YAML")
    return nothing
end
