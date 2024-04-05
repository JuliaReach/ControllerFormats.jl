"""
    AbstractPoolingLayerOp <: AbstractLayerOp

Abstract type for pooling layer operations.

### Notes

Pooling is an operation on a three-dimensional tensor that iterates over the
first two dimensions in a window and aggregates the values, thus reducing the
output dimension.

### Implementation

The following (unexported) functions should be implemented:

- `window(::AbstractPoolingLayerOp)`   -- return the pair ``(p, q)`` representing the window size
- `aggregation(::AbstractPoolingLayerOp)` -- return the aggregation function (applied to a tensor)
"""
abstract type AbstractPoolingLayerOp <: AbstractLayerOp end

for (type_name, normal_name, agg_function, agg_name) in
    ((:MaxPoolingLayerOp, "max", maximum, "maximum"),
     (:MeanPoolingLayerOp, "mean", mean, "Statistics.mean"))
    @eval begin
        @doc """
            $($type_name) <: AbstractPoolingLayerOp

        A $($normal_name)-pooling layer operation. The aggregation function is
        `$($agg_name)`.

        ### Fields

        - `p` -- horizontal window size
        - `q` -- vertical window size
        """
        struct $type_name <: AbstractPoolingLayerOp
            p::Int
            q::Int

            function $type_name(p::Int, q::Int; validate=Val(true))
                if validate isa Val{true} && (p <= 0 || q <= 0)
                    throw(ArgumentError("inconsistent window size ($p, $q)"))
                end
                return new(p, q)
            end
        end

        window(L::$type_name) = (L.p, L.q)

        aggregation(::$type_name) = $agg_function

        function Base.:(==)(L1::$type_name, L2::$type_name)
            return window(L1) == window(L2)
        end

        function Base.show(io::IO, L::$type_name)
            str = "$(string($type_name)) for $($normal_name)-pooling of window " *
                  "size $(window(L))"
            return print(io, str)
        end
    end
end

# application to a tensor
function (L::AbstractPoolingLayerOp)(T)
    s = size(T)
    if length(s) != 3
        throw(ArgumentError("a pooling layer requires a three-dimensional input, but got $s"))
    end
    p, q = window(L)
    @inbounds begin
        if mod(s[1], p) != 0 || mod(s[2], q) != 0
            throw(ArgumentError("pooling with window size ($p, $q) does " *
                                "not apply to a tensor of dimension $s"))
        end
        d1 = div(s[1], p)
        d2 = div(s[2], q)
        d3 = s[3]
    end
    s = (d1, d2, d3)
    O = similar(T, s)
    aggregate = aggregation(L)
    @inbounds for k in 1:d3
        for j in 1:d2
            for i in 1:d1
                cluster = view(T, ((i - 1) * p + 1):(i * p), ((j - 1) * q + 1):(j * q), k)
                O[i, j, k] = aggregate(cluster)
            end
        end
    end
    return O
end
