# optional dependencies
@static if !isdefined(Base, :get_extension)
    using Requires
end

@static if !isdefined(Base, :get_extension)
    function __init__()
        @require MAT = "23992714-dd62-5051-b70f-ba57cb901cac" include("../../ext/MATExt.jl")

        @require ONNX = "d0dd6a25-fac6-55c0-abf7-829e0c774d20" include("../../ext/ONNXExt.jl")

        @require YAML = "ddb6d928-2868-570f-bddf-ab3f9cf99eb6" include("../../ext/YAMLExt.jl")
    end
end
