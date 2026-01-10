"""
    ControllerFormats

Module for representations of controllers.

### Submodules

- [`Architecture`](@ref) -- data structures for controllers
- [`FileFormats`](@ref)  -- IO of file representations of controllers
"""
module ControllerFormats

using Reexport: @reexport

include("Architecture/Architecture.jl")

include("FileFormats/FileFormats.jl")

@reexport using .Architecture
@reexport using .FileFormats

end  # module
