import Flux

################
# Dense layers #
################

L1 = Flux.Dense(1, 2, Flux.relu)
L1.weight .= 1, 2
L1.bias .= 3, 4

L2 = Flux.Dense(2, 3, Flux.sigmoid)
L2.weight .= [1 2; 3 4; 5 6]

L3 = Flux.Dense(3, 1)
L3.weight .= [1 2 3;]

L_unsupported = Flux.Dense(1 => 1, Flux.trelu)

c = Flux.Chain(L1, L2, L3)

activations = [ReLU(), Sigmoid(), Id()]

# `==` is not defined for Flux types
function compare_Flux_layer(L1, L2)
    return L1.weight == L2.weight && L1.bias == L2.bias && L1.σ == L2.σ
end

# layer conversion
for (i, L) in enumerate(c.layers)
    op = convert(DenseLayerOp, L)
    @test op.weights == L.weight
    @test op.bias == L.bias
    @test op.activation == activations[i]

    L_back = convert(Flux.Dense, op)
    @test compare_Flux_layer(L, L_back)
end
@test_throws ArgumentError convert(DenseLayerOp, L_unsupported)

# network conversion
net = convert(FeedforwardNetwork, c)
c_back = convert(Flux.Chain, net)
@test length(net.layers) == length(c)
for (i, l) in enumerate(c.layers)
    @test net.layers[i] == convert(DenseLayerOp, l)

    @test compare_Flux_layer(l, c_back.layers[i])
end

# unknown activation function
W = hcat([1 0.5; -0.5 0.5; -1 -0.5])
b = [1.0, 0, -2]
L = DenseLayerOp(W, b, TestActivation())
@test_throws ArgumentError convert(Flux.Dense, L)

########################
# Convolutional layers #
########################

LC = Flux.Conv((2, 2), 1 => 1, Flux.relu)
LC.weight .= reshape([1 0; -1 2], (2, 2, 1, 1))
LC.bias .= 1

# layer conversion
op = convert(ConvolutionalLayerOp, LC)
@test op.weights[1] == LC.weight[:, :, :]
@test op.bias == LC.bias
@test op.activation == ReLU()
L_back = convert(Flux.Conv, op)
@test compare_Flux_layer(LC, L_back)
