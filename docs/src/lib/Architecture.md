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
layers(::AbstractNeuralNetwork)
```

The following non-standard methods are implemented:

```@docs
dim_in(::AbstractNeuralNetwork)
dim_out(::AbstractNeuralNetwork)
ControllerFormats.Architecture.dim(::AbstractNeuralNetwork)
```

#### Implementation

```@docs
FeedforwardNetwork
```

### Layer operations

```@docs
AbstractLayerOp
```

The following non-standard methods are useful to implement:

```@docs
dim_in(::AbstractLayerOp)
dim_out(::AbstractLayerOp)
ControllerFormats.Architecture.dim(::AbstractLayerOp)
```

#### More specific layer interfaces

```@docs
AbstractPoolingLayerOp
```

#### Implementation

```@docs
DenseLayerOp
ConvolutionalLayerOp
FlattenLayerOp
MaxPoolingLayerOp
MeanPoolingLayerOp
```

### Activation functions

```@docs
ActivationFunction
Id
ReLU
Sigmoid
Tanh
LeakyReLU
```

The following strings can be parsed as activation functions:

```@example
using ControllerFormats  # hide
ControllerFormats.FileFormats.available_activations
```
