# 2D input vector
x = [1.0, 1]

# 2x3 layer
W1 = hcat([1 0.5; -0.5 0.5; -1 -0.5])
b1 = [1.0, 0, -2]

# network with a single layer
L1 = DenseLayerOp(W1, b1, ReLU())
N1 = FeedforwardNetwork([L1])
@test layers(N1) == [L1]
@test N1(x) == max.(W1 * x + b1, 0) == [2.5, 0, 0]

# invalid layer combination
@test_throws ArgumentError FeedforwardNetwork([L1, L1])

# 3x2 layer
W2 = hcat([-1 -0.5 0; 0.5 -0.5 0])
b2 = [-1.0, 0]

# network with two layers
L2 = DenseLayerOp(W2, b2, Id())
N2 = FeedforwardNetwork([L1, L2])
@test layers(N2) == [L1, L2]
@test N2(x) == W2 * max.(W1 * x + b1, 0) + b2 == [-3.5, 1.25]

# printing
io = IOBuffer()
println(io, N1)
println(io, N2)

# equality
@test N1 == FeedforwardNetwork([L1])
@test N1 != FeedforwardNetwork([L2])

# approximate equality
@test N1 ≈ FeedforwardNetwork([L1])
@test N1 ≈ FeedforwardNetwork([DenseLayerOp(W1 .+ 1e-10, b1, ReLU())])
@test !(N1 ≈ FeedforwardNetwork([L2]))
@test !(N1 ≈ N2)

# list/array interface
@test length(N1) == 1 && length(N2) == 2
@test N1[1] == L1 && N1[1:1] == [L1] && N2[2] == L2 && N2[1:2] == [L1, L2]
@test N1[end] == L1 && N2[end] == L2

# dimensions
@test dim_in(N1) == 2 && dim_in(N2) == 2
@test dim_out(N1) == 3 && dim_out(N2) == 2
@test dim(N1) == (2, 3) && dim(N2) == (2, 2)

# network with all layer types
L1 = ConvolutionalLayerOp([reshape([1 0; -1 2], (2, 2, 1))], [1], ReLU())
L2 = MaxPoolingLayerOp(1, 1)
L3 = FlattenLayerOp()
W = zeros(2, 9);
W[1, 1] = W[2, 2] = 1;
L4 = DenseLayerOp(W, [1.0 0], ReLU())
N3 = FeedforwardNetwork([L1, L2, L3, L4])
T441 = reshape([0 4 2 1; -1 0 1 -2; 3 1 2 0; 0 1 4 1], (4, 4, 1))
@test N3(T441) == [3.0 2; 8 7]

# incompatible dimensions
@test_throws ArgumentError FeedforwardNetwork([L1, L4])
