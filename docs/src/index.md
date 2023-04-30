# ControllerFormats.jl

This light-weight [Julia](http://julialang.org) library contains basic
representations of controllers (currently deep neural networks) as well as
functionality to parse them from various file formats like MAT, YAML and ONNX.

The library originated from the package
[ClosedLoopReachability](https://github.com/JuliaReach/ClosedLoopReachability.jl),
which performs formal analysis of a given trained neural network.
This motivates that `ControllerFormats.jl` does not provide support for typical
other tasks such as network training, and some of the supported file formats are
only used by some similar analysis tool.

## Related packages

- [Flux.jl](https://github.com/FluxML/Flux.jl/) is a comprehensive Julia
  framework for machine learning. It also offers a representation of neural
  networks.
- [MLJ.jl](https://github.com/alan-turing-institute/MLJ.jl) is a large Julia
  library of machine-learning models such as neural networks and decision trees.
- [NNet](https://github.com/sisl/NNet) offers a representation of neural
  networks and a parser for the NNet format.
