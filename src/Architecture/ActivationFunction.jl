"""
    ActivationFunction

Abstract type for activation functions.
"""
abstract type ActivationFunction end

"""
    Id

Identity activation.

```math
    f(x) = x
```
"""
struct Id <: ActivationFunction end

(::Id)(x) = x

Base.show(io::IO, ::Id) = print(io, Id)

"""
    ReLU

Rectified linear unit (ReLU) activation.

```math
    f(x) = max(x, 0)
```
"""
struct ReLU <: ActivationFunction end

(::ReLU)(x) = max.(x, zero(eltype(x)))

Base.show(io::IO, ::ReLU) = print(io, ReLU)

"""
    Sigmoid

Sigmoid activation.

```math
    f(x) = \\frac{1}{1 + e^{-x}}
```
"""
struct Sigmoid <: ActivationFunction end

(::Sigmoid)(x) = @. 1 / (1 + exp(-x))

Base.show(io::IO, ::Sigmoid) = print(io, Sigmoid)

"""
    Tanh

Hyperbolic tangent activation.

```math
    f(x) = tanh(x) = \\frac{e^x - e^{-x}}{e^x + e^{-x}}
```
"""
struct Tanh <: ActivationFunction end

(::Tanh)(x) = tanh.(x)

Base.show(io::IO, ::Tanh) = print(io, Tanh)

"""
    LeakyReLU{N<:Number}

Leaky ReLU activation.

```math
    fₐ(x) = x > 0 ? x : a x
```
where ``a`` is the parameter.

### Fields

- `slope` -- parameter for negative inputs
"""
struct LeakyReLU{N<:Number} <: ActivationFunction
    slope::N
end

(lr::LeakyReLU)(x::Number) = x >= zero(x) ? x : lr.slope * x
(lr::LeakyReLU)(x::AbstractVector) = lr.(x)

Base.show(io::IO, lr::LeakyReLU) = print(io, "$LeakyReLU($(lr.slope))")

# constant instances of each activation function
const _id = Id()
const _relu = ReLU()
const _sigmoid = Sigmoid()
const _tanh = Tanh()

function load_Flux_activations()
    return quote
        activations_Flux = Dict(identity => _id,
                                _id => identity,
                                Flux.relu => _relu,
                                _relu => Flux.relu,
                                Flux.sigmoid => _sigmoid,
                                _sigmoid => Flux.sigmoid)
    end
end
