using ReachabilityBase.Subtypes: subtypes

# AbstractLayerOp implementation
struct TestLayerOp <: AbstractLayerOp end
L = TestLayerOp()
dim_in(L)
dim_out(L)

# 2D input vector and 2x3 layer
x = [1.0, 1]
W = hcat([1 0.5; -0.5 0.5; -1 -0.5])
b = [1.0, 0, -2]

L = DenseLayerOp(W, b, Id())

# printing
io = IOBuffer()
println(io, L)

# output for `x` under identity activation
@test L(x) == W * x + b == [2.5, 0, -3.5]

# invalid weight/bias combination
@test_throws ArgumentError DenseLayerOp(W, [1.0, 0], Id())

# equality
@test L == DenseLayerOp(W, b, Id())
@test L != DenseLayerOp(W .+ 1, b, Id()) &&
      L != DenseLayerOp(W, b .+ 1, Id()) &&
      L != DenseLayerOp(W, b, ReLU())
@test L != DenseLayerOp(hcat(1), [1], Id())

# approximate equality
@test L ≈ DenseLayerOp(W, b, Id())
@test L ≈ DenseLayerOp(W .+ 1e-10, b, Id()) &&
      L ≈ DenseLayerOp(W, b .+ 1e-10, Id()) &&
      !≈(L, DenseLayerOp(W .+ 1e-10, b, Id()); rtol=1e-12) &&
      !≈(L, DenseLayerOp(W, b .+ 1e-10, Id()); rtol=1e-12) &&
      ≈(L, DenseLayerOp(W .+ 1e-1, b, Id()); atol=1) &&
      ≈(L, DenseLayerOp(W, b .+ 1e-1, Id()); atol=1) &&
      !(L ≈ DenseLayerOp(W .+ 1, b, Id())) &&
      !(L ≈ DenseLayerOp(W, b .+ 1, Id())) &&
      !(L ≈ DenseLayerOp(W, b, ReLU()))
@test !(L ≈ DenseLayerOp(hcat(1), [1], Id()))

# dimensions
@test dim_in(L) == 2
@test dim_out(L) == 3
@test dim(L) == (2, 3)
@test length(L) == 3

# test methods for all activations
function test_layer(L::DenseLayerOp{Id})
    @test L(x) == [2.5, 0, -3.5]
end

function test_layer(L::DenseLayerOp{ReLU})
    @test L(x) == [2.5, 0, 0]
end

function test_layer(L::DenseLayerOp{Sigmoid})
    @test L(x) ≈ [0.924, 0.5, 0.029] atol = 1e-3
end

function test_layer(L::DenseLayerOp{Tanh})
    @test L(x) ≈ [0.986, 0, -0.998] atol = 1e-3
end

function test_layer(L::DenseLayerOp{<:LeakyReLU})
    @test L(x) == [2.5, 0, -0.035]
end

function test_layer(L::DenseLayerOp)
    return error("untested activation function: ", typeof(L.activation))
end

# run test with all activations
for act in subtypes(ActivationFunction)
    if act == TestActivation
        continue
    elseif act == LeakyReLU
        act_inst = LeakyReLU(0.01)
    else
        act_inst = act()
    end
    test_layer(DenseLayerOp(W, b, act_inst))
end

# leaky ReLU on a vector
act = LeakyReLU(0.01)
@test act([-1.0, 0, 1, -100]) == [-0.01, 0, 1, -1]
