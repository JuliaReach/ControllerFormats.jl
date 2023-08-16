# valid sample file
file = joinpath(@__DIR__, "sample_POLAR")

# parse file
N = read_POLAR(file)

@test length(N.layers) == 4

# write network back to file and re-read it
file = joinpath(@__DIR__, "sample_POLAR_output")

write_POLAR(N, file)
N2 = read_POLAR(file)
rm(file)

@test N == N2

# unknown activation function
W = hcat([1 0.5; -0.5 0.5; -1 -0.5])
b = [1.0, 0, -2]
N = FeedforwardNetwork([DenseLayerOp(W, b, TestActivation())])
@test_throws ArgumentError write_POLAR(N, file)
rm(file)
