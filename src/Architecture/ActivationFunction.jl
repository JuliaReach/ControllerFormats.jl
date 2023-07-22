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

"""
    ReLU

Rectified linear unit (ReLU) activation.

```math
    f(x) = max(x, 0)
```
"""
struct ReLU <: ActivationFunction end

(::ReLU)(x) = max.(x, zero(eltype(x)))

"""
    Sigmoid

Sigmoid activation.

```math
    f(x) = \\frac{1}{1 + e^{-x}}
```
"""
struct Sigmoid <: ActivationFunction end

(::Sigmoid)(x) = @. 1 / (1 + exp(-x))

"""
    Tanh

Hyperbolic tangent activation.

```math
    f(x) = tanh(x) = \\frac{e^x - e^{-x}}{e^x + e^{-x}}
```
"""
struct Tanh <: ActivationFunction end

(::Tanh)(x) = tanh.(x)

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

# constant instances of each activation function
const _id = Id()
const _relu = ReLU()
const _sigmoid = Sigmoid()
const _tanh = Tanh()

function load_Flux_activations()
    return quote
        activations_Flux = Dict(Flux.identity => _id,
                                _id => Flux.identity,
                                Flux.relu => _relu,
                                _relu => Flux.relu,
                                Flux.sigmoid => _sigmoid,
                                _sigmoid => Flux.sigmoid)
    end
end
