module ONNXExt

using ControllerFormats
using ControllerFormats.FileFormats: available_activations

@static if isdefined(Base, :get_extension)
    import ONNX
else
    import ..ONNX
end

function ControllerFormats.FileFormats._read_ONNX(filename::String; input_dimension=nothing)
    # parse input dimension if not provided
    if isnothing(input_dimension)
        input_dimension = open(filename) do io
            onnx_raw_model = ONNX.decode(ONNX.ProtoDecoder(io), ONNX.ModelProto)
            input = onnx_raw_model.graph.input
            @assert input isa Vector{ONNX.ValueInfoProto} && length(input) == 1
            dimensions = input[1].var"#type".value.value.shape.dim
            @assert dimensions isa Vector{ONNX.var"TensorShapeProto.Dimension"} &&
                    length(dimensions) == 2 && dimensions[1].value.value == 1
            return dimensions[2].value.value
        end
    end

    # ONNX.jl expects an input, so the user must provide that
    x0 = zeros(Float32, input_dimension)

    # read data
    data = ONNX.load(filename, x0)

    @assert data isa ONNX.Umlaut.Tape{ONNX.ONNXCtx} "`read_ONNX` must be called with " *
                                                    "`ONNX.Umlaut.Tape{ONNX.ONNXCtx}`"

    layer_parameters = []
    ops = data.ops
    @assert ops[1] isa ONNX.Umlaut.Input && iszero(ops[1].val)  # skip input operation
    idx = 2
    @inbounds while idx <= length(ops)
        op = ops[idx]
        if !(op.val isa AbstractMatrix)
            break
        end
        W = permutedims(op.val)
        idx += 1
        op = ops[idx]
        @assert op.val isa AbstractVector "expected a bias vector"
        b = op.val
        push!(layer_parameters, (W, b))
        idx += 1
    end
    n_layers = div(idx - 2, 2)
    # 4 operations per layer +1 for the input operation
    # (-1 potentially for implicit identity activation in the last layer)
    @assert length(ops) == 4 * n_layers ||
            length(ops) == 4 * n_layers + 1 "each layer should consist of 4 " *
                                            "operations (except possibly the last one)"
    T = DenseLayerOp{<:ActivationFunction,Matrix{Float32},Vector{Float32}}
    layers = T[]
    layer = 1
    while idx <= length(ops)
        # affine map (treated implicitly)
        op = ops[idx]
        @assert op isa ONNX.Umlaut.Call "expected an affine map"
        args = op.args
        @assert length(args) == 5
        @assert args[2] == ONNX.onnx_gemm
        @assert args[3]._op.id == (layer == 1 ? 1 : idx - 1)
        @assert args[4]._op.id == 2 * layer
        @assert args[5]._op.id == 2 * layer + 1
        W, b = @inbounds layer_parameters[layer]
        idx += 1

        # activation function
        if idx > length(ops)
            # last layer is assumed to be the identity
            a = Architecture._id
        else
            op = ops[idx]
            @assert op isa ONNX.Umlaut.Call "expected an activation function"
            args = op.args
            if length(args) == 1
                @assert args[1]._op.id == idx - 1
                act = op.fn
            elseif length(args) == 2
                @assert args[2]._op.id == idx - 1
                act = args[1]
            else
                throw(ArgumentError("cannot parse activation $op"))  # COV_EXCL_LINE
            end
            a = available_activations[string(act)]
            idx += 1
        end

        L = DenseLayerOp(W, b, a)
        push!(layers, L)
        layer += 1
    end

    return FeedforwardNetwork(layers)
end

end  # module
