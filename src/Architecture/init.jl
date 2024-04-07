# optional dependencies
function __init__()
    @require Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c" begin
        eval(load_Flux_activations())
        eval(load_Flux_convert_Dense_layer())
        eval(load_Flux_convert_network())
    end
end
