# AbstractNeuralNetwork implementation
struct TestNeuralNetwork <: AbstractNeuralNetwork end

function ControllerFormats.layers(::TestNeuralNetwork)
    return [11, 12, 13]
end

N = TestNeuralNetwork()
@test length(N) == 3
@test N[1] == 11 && N[[3, 1]] == [13, 11]
@test N[end] == 13
