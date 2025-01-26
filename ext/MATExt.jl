module MATExt

import ControllerFormats.FileFormats

@static if isdefined(Base, :get_extension)
    import MAT
else
    import ..MAT
end

FileFormats._load_MAT(filename::String) = MAT.matread(filename)

end  # module
