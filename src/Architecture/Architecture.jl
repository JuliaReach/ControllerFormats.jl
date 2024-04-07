"""
    Architecture

Module containing data structures to represent controllers.
"""
module Architecture

using Requires
using LinearAlgebra: dot
using Statistics: mean

export AbstractNeuralNetwork, FeedforwardNetwork,
       AbstractLayerOp, DenseLayerOp, ConvolutionalLayerOp, FlattenLayerOp,
       AbstractPoolingLayerOp, MaxPoolingLayerOp, MeanPoolingLayerOp,
       layers, dim_in, dim_out,
       ActivationFunction, Id, ReLU, Sigmoid, Tanh, LeakyReLU

include("ActivationFunction.jl")
include("LayerOps/AbstractLayerOp.jl")
include("LayerOps/DenseLayerOp.jl")
include("LayerOps/ConvolutionalLayerOp.jl")
include("LayerOps/FlattenLayerOp.jl")
include("LayerOps/PoolingLayerOp.jl")
include("NeuralNetworks/AbstractNeuralNetwork.jl")
include("NeuralNetworks/FeedforwardNetwork.jl")

include("init.jl")

end  # module
