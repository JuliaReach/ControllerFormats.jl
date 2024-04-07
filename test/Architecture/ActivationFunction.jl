# printing
io = IOBuffer()
for act in (Id(), ReLU(), Sigmoid(), Tanh(), LeakyReLU(0.1))
    println(io, act)
end

# leaky ReLU on a vector
act = LeakyReLU(0.01)
@test act([-1.0, 0, 1, -100]) == [-0.01, 0, 1, -1]
