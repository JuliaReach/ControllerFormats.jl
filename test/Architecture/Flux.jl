import Flux

l1 = Flux.Dense(1, 2, Flux.relu)
l1.weight .= 1, 2
l1.bias .= 3, 4

l2 = Flux.Dense(2, 3, Flux.sigmoid)
l2.weight .= [1 2; 3 4; 5 6]

l3 = Flux.Dense(3, 1)
l3.weight .= [1 2 3;]

l_unsupported = Flux.Dense(1 => 1, Flux.trelu)

c = Flux.Chain(l1, l2, l3)

activations = [ReLU(), Sigmoid(), Id()]

# `==` is not defined for Flux types
function compare_Flux_layer(l1, l2)
    return l1.weight == l2.weight && l1.bias == l2.bias && l1.σ == l2.σ
end

# layer conversion
for (i, l) in enumerate(c.layers)
    op = convert(DenseLayerOp, l)
    @test op.weights == l.weight
    @test op.bias == l.bias
    @test op.activation == activations[i]

    l_back = convert(Flux.Dense, op)
    @test compare_Flux_layer(l, l_back)
end
@test_throws ArgumentError convert(DenseLayerOp, l_unsupported)

# network conversion
net = convert(FeedforwardNetwork, c)
c_back = convert(Flux.Chain, net)
@test length(net.layers) == length(c)
for (i, l) in enumerate(c.layers)
    @test net.layers[i] == convert(DenseLayerOp, l)

    @test compare_Flux_layer(l, c_back.layers[i])
end
