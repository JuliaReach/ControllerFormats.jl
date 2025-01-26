# optional dependencies
@static if !isdefined(Base, :get_extension)
    using Requires
end

@static if !isdefined(Base, :get_extension)
    function __init__()
        @require Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c" include("../../ext/FluxExt.jl")
    end
end
