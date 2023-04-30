using Documenter, ControllerFormats

DocMeta.setdocmeta!(ControllerFormats, :DocTestSetup,
                    :(using ControllerFormats); recursive=true)

makedocs(
    sitename = "ControllerFormats.jl",
    modules = [ControllerFormats],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        assets = ["assets/aligned.css"]),
    pages = [
        "Home" => "index.md",
        "About" => "about.md"
    ],
    strict = true
)

deploydocs(
    repo = "github.com/JuliaReach/ControllerFormats.jl.git",
    push_preview = true
)
