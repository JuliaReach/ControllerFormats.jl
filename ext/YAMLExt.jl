module YAMLExt

import ControllerFormats.FileFormats

@static if isdefined(Base, :get_extension)
    import YAML
else
    import ..YAML
end

FileFormats._load_YAML(filename::String) = YAML.load_file(filename)

end  # module
