# valid sample file; features:
# - comma-separated lists may contain whitespaces
# - lines may end with a whitespace
# - line 7 contains one element too much, which must be ignored
file = joinpath(@__DIR__, "sample_NNet.nnet")

# parse file
N = read_NNet(file)

@test length(N.layers) == 3

# write network back to file and re-read it
file = joinpath(@__DIR__, "sample_NNet_output.nnet")

write_NNet(N, file)
N2 = read_NNet(file)
rm(file)

@test N == N2
