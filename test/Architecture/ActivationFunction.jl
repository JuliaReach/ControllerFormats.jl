# printing
io = IOBuffer()
for act in (Id(), ReLU(), Sigmoid(), Tanh(), LeakyReLU(0.1))
    println(io, act)
end
