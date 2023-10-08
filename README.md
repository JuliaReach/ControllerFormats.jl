# ControllerFormats.jl

| **Documentation** | **Status** | **Community** | **License** |
|:-----------------:|:----------:|:-------------:|:-----------:|
| [![docs-dev][dev-img]][dev-url] | [![CI][ci-img]][ci-url] [![codecov][cov-img]][cov-url] | [![zulip][chat-img]][chat-url] | [![license][lic-img]][lic-url] |

[dev-img]: https://img.shields.io/badge/docs-latest-blue.svg
[dev-url]: https://juliareach.github.io/ControllerFormats.jl/dev/
[ci-img]: https://github.com/JuliaReach/ControllerFormats.jl/workflows/CI/badge.svg
[ci-url]: https://github.com/JuliaReach/ControllerFormats.jl/actions/workflows/ci.yml
[cov-img]: https://codecov.io/github/JuliaReach/ControllerFormats.jl/coverage.svg
[cov-url]: https://app.codecov.io/github/JuliaReach/ControllerFormats.jl
[chat-img]: https://img.shields.io/badge/zulip-join_chat-brightgreen.svg
[chat-url]: https://julialang.zulipchat.com/#narrow/stream/278609-juliareach
[lic-img]: https://img.shields.io/github/license/mashape/apistatus.svg
[lic-url]: https://github.com/JuliaReach/ControllerFormats.jl/blob/master/LICENSE

`ControllerFormats` is a [Julia](http://julialang.org) package for working with
various formats for different types of controllers.
Currently the package supports neural-network controllers.

## Installing

This package requires Julia v1.6 or later.
We refer to the [official documentation](https://julialang.org/downloads) on how
to install and run Julia on your system.

Depending on your needs, choose an appropriate command from the following list
and enter it in Julia's REPL.
To activate the `pkg` mode, type `]` (and to leave it, type `<backspace>`).

#### [Install the latest release version](https://julialang.github.io/Pkg.jl/v1/managing-packages/#Adding-registered-packages-1)

```julia
pkg> add ControllerFormats
```

#### Install the latest development version

```julia
pkg> add ControllerFormats#master
```

#### [Clone the package for development](https://julialang.github.io/Pkg.jl/v1/managing-packages/#Developing-packages-1)

```julia
pkg> dev ControllerFormats
```
