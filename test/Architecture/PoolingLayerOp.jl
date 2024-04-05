using ControllerFormats.Architecture: window, aggregation
using Statistics: mean

LTs = (MaxPoolingLayerOp, MeanPoolingLayerOp)
Ls = [LT(2, 3) for LT in LTs]
L1, L2 = Ls

# printing
io = IOBuffer()
for L in Ls
    println(io, L)
end

# invalid inputs
for LT in LTs
    @test_throws ArgumentError LT(0, 1)
    @test_throws ArgumentError LT(1, 0)
end

# window
@test window(L1) == (2, 3)

# aggregation
@test aggregation(L1) == maximum
@test aggregation(L2) == mean

# output for tensor `T`
T = reshape(1:72.0, (4, 6, 3))
@test L1(T) == cat([10 22; 12 24], [34 46; 36 48], [58 70; 60 72]; dims=3)
@test L2(T) == cat([5.5 17.5; 7.5 19.5], [29.5 41.5; 31.5 43.5], [53.5 65.5; 55.5 67.5]; dims=3)
for L in Ls
    @test_throws ArgumentError L(reshape(1:4.0, (2, 2)))
    @test_throws ArgumentError L(reshape(1:8.0, (2, 2, 2)))
end

# equality
for (L, LT) in zip(Ls, LTs)
    @test L == LT(2, 3)
    @test L != LT(2, 2) && L != LT(3, 3)
end
@test L1 != L2
