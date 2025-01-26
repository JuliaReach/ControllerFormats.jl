"""
    read_ONNX(filename::String; [input_dimension=nothing])

Read a neural network stored in [ONNX](https://github.com/onnx/onnx) format.
This function requires to load the
[`ONNX.jl` library](https://github.com/FluxML/ONNX.jl).

### Input

- `filename`        -- name of the `ONNX` file
- `input_dimension` -- (optional; default: `nothing`) input dimension (required
                       by `ONNX.jl` parser); see the notes below

### Output

A [`FeedforwardNetwork`](@ref).

### Notes

This implementation assumes the following structure:
1. First comes the input vector (which is ignored).
2. Next come the weight matrices `W` (transposed) and bias vectors `b` in pairs
   *in the order in which they are applied*.
3. Next come the affine maps and the activation functions *in the order in which
   they are applied*. The last layer does not have an activation function.

Some of these assumptions are currently *not validated*. Hence it may happen
that this function returns a result that is incorrect.

If the argument `input_dimension` is not provided, the file is parsed an
additional time to read the correct number (which is inefficient).
"""
function read_ONNX(filename::String; input_dimension=nothing)
    _read_ONNX(filename; input_dimension)
end

# defined in `ONNXExt.jl`
function _read_ONNX(filename; input_dimension)
    mod = isdefined(Base, :get_extension) ? Base.get_extension(@__MODULE__, :ONNXExt) : @__MODULE__
    require(mod, :ONNX; fun_name="read_ONNX")
    return nothing
end
