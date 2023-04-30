```@meta
DocTestSetup = :(using ControllerFormats)
CurrentModule = ControllerFormats
```

# Architecture module

```@docs
Architecture
```

```@contents
Pages = ["Architecture.md"]
Depth = 3
```

## Neural networks

An artificial neural network can be used as a controller.

### General interface

```@docs
AbstractNeuralNetwork
dim_in(::AbstractNeuralNetwork)
dim_out(::AbstractNeuralNetwork)
ControllerFormats.dim(::AbstractNeuralNetwork)
```

#### Implementation

```@docs
FeedforwardNetwork
```

### Layer operations

```@docs
AbstractLayerOp
dim_in(::AbstractLayerOp)
dim_out(::AbstractLayerOp)
ControllerFormats.dim(::AbstractLayerOp)
```

#### Implementation

```@docs
DenseLayerOp
```

### Activation functions

```@docs
ActivationFunction
Id
ReLU
Sigmoid
Tanh
```

The following strings can be parsed as activation functions:

```@example
using ControllerFormats  # hide
ControllerFormats.FileFormats.available_activations
```
