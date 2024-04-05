using Test, ControllerFormats

using ControllerFormats.Architecture: dim

import Flux, MAT, ONNX, YAML

struct TestActivation <: ActivationFunction end

@testset "Architecture" begin
    @testset "ActivationFunction" begin
        include("Architecture/ActivationFunction.jl")
    end
    @testset "AbstractLayerOp" begin
        include("Architecture/AbstractLayerOp.jl")
    end
    @testset "DenseLayerOp" begin
        include("Architecture/DenseLayerOp.jl")
    end
    @testset "PoolingLayerOp" begin
        include("Architecture/PoolingLayerOp.jl")
    end
    @testset "AbstractNeuralNetwork" begin
        include("Architecture/AbstractNeuralNetwork.jl")
    end
    @testset "FeedforwardNetwork" begin
        include("Architecture/FeedforwardNetwork.jl")
    end
    @testset "Flux bridge" begin
        include("Architecture/Flux.jl")
    end
end

@testset "FileFormats" begin
    @testset "MAT" begin
        include("FileFormats/MAT.jl")
    end
    @testset "NNet" begin
        include("FileFormats/NNet.jl")
    end
    @testset "ONNX" begin
        include("FileFormats/ONNX.jl")
    end
    @testset "POLAR" begin
        include("FileFormats/POLAR.jl")
    end
    @testset "Sherlock" begin
        include("FileFormats/Sherlock.jl")
    end
    @testset "YAML" begin
        include("FileFormats/YAML.jl")
    end
end

include("Aqua.jl")
