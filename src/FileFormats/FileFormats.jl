"""
    FileFormats

Module to parse and write file formats of controllers.
"""
module FileFormats

using ..Architecture
using ReachabilityBase.Require

export read_MAT,
       read_NNet, write_NNet,
       read_ONNX,
       read_POLAR, write_POLAR,
       read_Sherlock, write_Sherlock,
       read_YAML

include("available_activations.jl")

include("MAT.jl")
include("NNet.jl")
include("ONNX.jl")
include("POLAR.jl")
include("Sherlock.jl")
include("YAML.jl")

include("init.jl")

end  # module
