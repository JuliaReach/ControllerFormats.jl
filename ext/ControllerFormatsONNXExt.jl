module ControllerFormatsONNXExt

using ControllerFormats
import ControllerFormats.FileFormats
using ControllerFormats.FileFormats: available_activations

@static if isdefined(Base, :get_extension)
    import ONNX
else
    import ..ONNX
end

function FileFormats._ext_read_ONNX(filename::String; input_dimension=nothing)
    # parse input dimension if not provided
    if isnothing(input_dimension)
        input_dimension = get_input_dimension(filename)
    end

    # ONNX.jl expects an input, so the user must provide that
    x0 = zeros(Float32, input_dimension)

    # read data
    data = ONNX.load(filename, x0)

    @assert data isa ONNX.Umlaut.Tape{ONNX.ONNXCtx} "`read_ONNX` must be called with " *
                                                    "`ONNX.Umlaut.Tape{ONNX.ONNXCtx}`"

    layer_parameters = []
    ops = data.ops
    @assert iszero(ops[1].val)
    if ops[1] isa Union{ONNX.Umlaut.Input,ONNX.Umlaut.Constant}
        idx = 2  # skip empty input/constant operation
    else
        @assert false "unsupported first op $(typeof(ops[1]))"
    end
    @inbounds while idx <= length(ops)
        op = ops[idx]
        # expect weights matrix
        if op.val isa AbstractMatrix
            W_flattened = op.val
        elseif op.val isa AbstractArray
            s = size(op.val)
            if length(s) == 1
                break
            end
            i = -1
            o = -1
            for j in eachindex(s)
                if s[j] == 1
                    continue
                elseif i == -1
                    i = j
                elseif o == -1
                    o = j
                else
                    @assert false "unexpected dimension: $s"
                end
            end
            @assert o != -1 "unexpected dimension: $s"
            @show s
            @show s[i]
            @show s[o]
            W_flattened = reshape(op.val, (s[i], s[o]))
        else
            break
        end
        W = permutedims(W_flattened)
        idx += 1
        op = ops[idx]
        @assert op.val isa AbstractVector "expected a bias vector"
        b = op.val
        push!(layer_parameters, (W, b))
        idx += 1
    end
    @show n_layers = div(idx - 2, 2)
    @show length(ops)
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

function get_input_dimension(filename::String)
    input_dimension = open(filename) do io
        onnx_raw_model = ONNX.decode(ONNX.ProtoDecoder(io), ONNX.ModelProto)
        input = onnx_raw_model.graph.input
        @assert input isa Vector{ONNX.ValueInfoProto} && !isempty(input)
        dimensions = input[1].var"#type".value.value.shape.dim
        @assert dimensions isa Vector{ONNX.var"TensorShapeProto.Dimension"}
        # `dimensions` consists of 1's except for the last entry
        for (i, di) in enumerate(dimensions)
            vi = di.value.value
            if vi != 1
                @assert i == length(dimensions)
            end
        end
        input_dimension = dimensions[end].value.value
    end
    return input_dimension
end

end  # module
