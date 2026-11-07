"""
    AbstractLayerOp

Abstract type for layer operations.

### Notes

An `AbstractLayerOp` represents a layer *operation*. A classical example is a
"dense layer operation" with an affine map followed by an activation function.
"""
abstract type AbstractLayerOp end

@doc """
    dim_in(L::AbstractLayerOp)

Return the input dimension of a layer operation.

### Input

- `L` -- layer operation

### Output

The input dimension of `L`.
"""
dim_in(::AbstractLayerOp)
function dim_in end

@doc """
    dim_out(L::AbstractLayerOp)

Return the output dimension of a layer operation.

### Input

- `L` -- layer operation

### Output

The output dimension of `L`.
"""
dim_out(::AbstractLayerOp)
function dim_out end

"""
    dim(L::AbstractLayerOp)

Return the input and output dimension of a layer operation.

### Input

- `N` -- neural network

### Output

The pair ``(i, o)`` where ``i`` is the input dimension and ``o`` is the output
dimension of `N`.

### Notes

This function is not exported due to name conflicts with other related packages.
"""
dim(L::AbstractLayerOp) = (dim_in(L), dim_out(L))
