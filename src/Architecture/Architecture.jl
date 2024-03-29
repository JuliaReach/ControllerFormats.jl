"""
    Architecture

Module containing data structures to represent controllers.
"""
module Architecture

using Requires

export AbstractNeuralNetwork, AbstractLayerOp,
       FeedforwardNetwork, DenseLayerOp,
       layers, dim_in, dim_out,
       ActivationFunction, Id, ReLU, Sigmoid, Tanh, LeakyReLU

include("AbstractNeuralNetwork.jl")
include("AbstractLayerOp.jl")
include("ActivationFunction.jl")
include("DenseLayerOp.jl")
include("FeedforwardNetwork.jl")

include("init.jl")

end  # module
