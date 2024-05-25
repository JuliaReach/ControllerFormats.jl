"""
    FlattenLayerOp <: AbstractLayerOp

A flattening layer operation converts a multidimensional tensor into a vector.

### Notes

The implementation uses row-major ordering for convenience with the
machine-learning literature.

```@jldoctest
julia> T = reshape([1, 3, 2, 4, 5, 7, 6, 8], (2, 2, 2))
2×2×2 Array{Int64, 3}:
[:, :, 1] =
 1  2
 3  4

[:, :, 2] =
 5  6
 7  8

julia> FlattenLayerOp()(T)
8-element Vector{Int64}:
 1
 2
 3
 4
 5
 6
 7
 8
```
"""
struct FlattenLayerOp <: AbstractLayerOp
end

# application to a vector (swap to row-major convention)
function (L::FlattenLayerOp)(T)
    s = size(T)
    if length(s) == 1
        return vec(T)
    end
    return vec(permutedims(T, (2, 1, 3:length(s)...)))
end

size(::FlattenLayerOp) = (nothing, 1)
