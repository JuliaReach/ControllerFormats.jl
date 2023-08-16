using Test, ControllerFormats

using ControllerFormats: dim

import Flux, MAT, ONNX, YAML

@testset "Architecture" begin
    @testset "DenseLayerOp" begin
        include("Architecture/DenseLayerOp.jl")
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
