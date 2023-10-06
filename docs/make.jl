using Documenter, ControllerFormats

DocMeta.setdocmeta!(ControllerFormats, :DocTestSetup,
                    :(using ControllerFormats); recursive=true)

makedocs(; sitename="ControllerFormats.jl",
         modules=[ControllerFormats],
         format=Documenter.HTML(; prettyurls=get(ENV, "CI", nothing) == "true",
                                assets=["assets/aligned.css"]),
         pagesonly=true,
         pages=["Home" => "index.md",
                "Library" => ["ControllerFormats module" => "lib/ControllerFormats.md",
                              "Architecture module" => "lib/Architecture.md",
                              "FileFormats module" => "lib/FileFormats.md"],
                "About" => "about.md"])

deploydocs(; repo="github.com/JuliaReach/ControllerFormats.jl.git",
           push_preview=true)
