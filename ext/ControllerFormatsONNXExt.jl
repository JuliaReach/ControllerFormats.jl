module ControllerFormatsONNXExt

using ControllerFormats
import ControllerFormats.FileFormats
using ControllerFormats.FileFormats: available_activations

@static if isdefined(Base, :get_extension)
    import ONNX
else
    import ..ONNX
end

const supported_activations = Set([identity, ONNX.relu, ONNX.tanh])

function FileFormats._ext_read_ONNX(filename::String; input_dimension=nothing)
    # parse input dimension if not provided
    if isnothing(input_dimension)
        input_dimension = get_input_dimension(filename)
    end

    # ONNX.jl expects an input vector
    x0 = zeros(Float32, input_dimension)

    # read data
    data = ONNX.load(filename, x0)

    @assert data isa ONNX.Umlaut.Tape{ONNX.ONNXCtx} "`read_ONNX` must be called with " *
                                                    "`ONNX.Umlaut.Tape{ONNX.ONNXCtx}`"

    found_input = false
    T = DenseLayerOp{<:ActivationFunction,Matrix{Float32},Vector{Float32}}
    layers = T[]
    layer = 1
    ops_converted = Vector{Any}(undef, length(data.ops))
    @inbounds for (i, op) in enumerate(data.ops)
        if op isa ONNX.Umlaut.Constant
            if op.typ == Vector{Float32}
                # bias vector
                ops_converted[i] = op.val
            else
                # weights matrix
                @assert op.val isa AbstractArray
                ops_converted[i] = _read_weights_matrix(op)
            end
        elseif op isa ONNX.Umlaut.Call
            # affine map or activation function
            args = op.args
            if length(args) == 5 && args[2] == ONNX.onnx_gemm
                # affine map
                prev_layer = args[3]
                weights = args[4]
                bias = args[5]
                # these assertions are just an observation and can be deactivated
                @assert prev_layer._op.id == (layer == 1 ? 1 : i - 1)
                @assert weights._op.id == 2 * layer
                @assert bias._op.id == 2 * layer + 1

                W = ops_converted[weights._op.id]
                b = ops_converted[bias._op.id]
                ops_converted[i] = (W, b)  # store affine map as a Tuple
            elseif op.fn == broadcast || op.fn ∈ supported_activations
                # activation function
                if length(args) == 1
                    @assert op.fn ∈ supported_activations
                    j = args[1]._op.id
                    # this assertion is just an observation and can be deactivated
                    @assert j == i - 1
                    W, b = ops_converted[j]
                    act = op.fn
                elseif length(args) == 2
                    @assert op.fn == broadcast
                    j = args[2]._op.id
                    # this assertion is just an observation and can be deactivated
                    @assert j == i - 1
                    W, b = ops_converted[j]
                    act = args[1]
                else
                    throw(ArgumentError("unsupported activation $op"))  # COV_EXCL_LINE
                end
                a = available_activations[string(act)]

                # finish this layer
                L = DenseLayerOp(W, b, a)
                ops_converted[i] = L
                push!(layers, L)
                layer += 1
            elseif op.fn == ONNX.sub  # COV_EXCL_LINE
                throw(ArgumentError("unsupported operation sub: $op"))  # COV_EXCL_LINE
            else
                throw(ArgumentError("unsupported operation: $op"))  # COV_EXCL_LINE
            end
        elseif op isa ONNX.Umlaut.Input
            # ignore input
            @assert iszero(op.val) && length(op.val) == input_dimension && !found_input
            found_input = true
            ops_converted[i] = x0
        else
            throw(ArgumentError("unsupported operation: $op"))  # COV_EXCL_LINE
        end
    end

    # typically add last layer with identity activation
    if ops_converted[end] isa Tuple
        W, b = ops_converted[end]
        a = Architecture._id
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
        found_dim_value = false
        for di in dimensions
            vi = di.value
            if !(vi isa ONNX.OneOf{Int64}) || vi.name != :dim_value
                continue
            end
            found_dim_value = true
            if vi.value != 1  # skip values 1
                return vi.value
            end
        end
        @assert found_dim_value
        return 1
    end
    return input_dimension
end

function _read_weights_matrix(op)
    if op.typ == Matrix{Float32}
        W = op.val
    elseif op.val isa AbstractArray
        s = size(op.val)
        @assert length(s) > 1
        # search for two numbers > 1, if any
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
                @assert false "unexpected dimension: $s"  # COV_EXCL_LINE
            end
        end
        # default indices: first and last
        if o == -1
            if i == -1
                i = 1
            end
            o = length(s)
        end
        W = reshape(op.val, (s[i], s[o]))
    else
        @assert false "unsupported operation: $op"  # COV_EXCL_LINE
    end
    return permutedims(W)
end

end  # module
