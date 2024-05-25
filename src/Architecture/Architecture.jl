"""
    Architecture

Module containing data structures to represent controllers.
"""
module Architecture

using Requires

export AbstractNeuralNetwork, FeedforwardNetwork,
       AbstractLayerOp, DenseLayerOp,
       layers, dim_in, dim_out,
       ActivationFunction, Id, ReLU, Sigmoid, Tanh, LeakyReLU

include("ActivationFunction.jl")
include("LayerOps/AbstractLayerOp.jl")
include("LayerOps/DenseLayerOp.jl")
include("NeuralNetworks/AbstractNeuralNetwork.jl")
include("NeuralNetworks/FeedforwardNetwork.jl")

include("init.jl")

end  # module
