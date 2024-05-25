using ControllerFormats.Architecture: kernel, n_filters
using ReachabilityBase.Subtypes: subtypes

# 4x4x1 input tensor
T441 = reshape([0 4 2 1; -1 0 1 -2; 3 1 2 0; 0 1 4 1], (4, 4, 1))
O_Id = reshape([2 7 -2; -1 4 0; 6 9 1], (3, 3, 1))
# 2x2x3 input tensor
T223 = reshape(1:12, (2, 2, 3))

W1 = reshape([1 0; -1 2], (2, 2, 1))
b1 = 1
W2 = W1
b2 = 2
# 2x2 kernel and 1 filter
Ws = [W1]
bs = [b1]

# invalid weight/bias combination
@test_throws ArgumentError ConvolutionalLayerOp(Ws, [1, 0], Id())
@test_throws ArgumentError ConvolutionalLayerOp([], [], Id())
@test_throws ArgumentError ConvolutionalLayerOp([W1, hcat(1)], [1, 0], Id())
@test_throws ArgumentError ConvolutionalLayerOp([[1 0; -1 2]], [1], Id())

# one filter
L = ConvolutionalLayerOp(Ws, bs, ReLU())
# two filters
L2 = ConvolutionalLayerOp([W1, W2], [b1, b2], ReLU())

# printing
io = IOBuffer()
println(io, L)

# output for tensors
@test L(T441) == reshape([2 7 0; 0 4 0; 6 9 1], (3, 3, 1))
@test L2(T441) == cat([2 7 0; 0 4 0; 6 9 1], [3 8 0; 0 5 1; 7 10 2]; dims=(3))
@test_throws ArgumentError L(T223)
@test_throws ArgumentError L(reshape(1:4.0, (2, 2)))

# equality
@test L == ConvolutionalLayerOp(Ws, bs, ReLU())
@test L != ConvolutionalLayerOp([W1 .+ 1], bs, ReLU()) &&
      L != ConvolutionalLayerOp(Ws, [b1 .+ 1], ReLU()) &&
      L != ConvolutionalLayerOp(Ws, bs, Id())

# approximate equality
@test L ≈ ConvolutionalLayerOp(Ws, bs, ReLU())
@test L ≈ ConvolutionalLayerOp([W1 .+ 1e-10], bs, ReLU()) &&
      L ≈ ConvolutionalLayerOp(Ws, [b1 .+ 1e-10], ReLU()) &&
      !≈(L, ConvolutionalLayerOp([W1 .+ 1e-10], bs, ReLU()); rtol=1e-12) &&
      !≈(L, ConvolutionalLayerOp(Ws, [b1 .+ 1e-10], ReLU()); rtol=1e-12) &&
      ≈(L, ConvolutionalLayerOp([W1 .+ 1e-1], bs, ReLU()); atol=1) &&
      ≈(L, ConvolutionalLayerOp(Ws, [b1 .+ 1e-1], ReLU()); atol=1) &&
      !(L ≈ ConvolutionalLayerOp([W1 .+ 1], bs, ReLU())) &&
      !(L ≈ ConvolutionalLayerOp(Ws, [b1 .+ 1], ReLU())) &&
      !(L ≈ ConvolutionalLayerOp(Ws, bs, Id()))

# size
@test size(L) == (3, 3)

# kernel size and number of filters
@test kernel(L) == kernel(L2) == (2, 2, 1)
@test n_filters(L) == 1 && n_filters(L2) == 2

# test methods for all activations
function test_layer(L::ConvolutionalLayerOp{Id})
    @test L(T441) == O_Id
end

function test_layer(L::ConvolutionalLayerOp{ReLU})
    @test L(T441) == reshape([2 7 0; 0 4 0; 6 9 1], (3, 3, 1))
end

function test_layer(L::ConvolutionalLayerOp{Sigmoid})
    @test L(float(T441)) ≈ Sigmoid().(O_Id) atol = 1e-3
end

function test_layer(L::ConvolutionalLayerOp{Tanh})
    @test L(float(T441)) ≈ Tanh().(O_Id) atol = 1e-3
end

function test_layer(L::ConvolutionalLayerOp{<:LeakyReLU})
    @test L(T441) == O_Id
end

function test_layer(L::ConvolutionalLayerOp)
    return error("untested activation function: ", typeof(L.activation))
end

# run test with all activations
for act in subtypes(ActivationFunction)
    if act == TestActivation
        continue
    elseif act == LeakyReLU
        act_inst = LeakyReLU(1)
    else
        act_inst = act()
    end
    test_layer(ConvolutionalLayerOp(Ws, bs, act_inst))
end
