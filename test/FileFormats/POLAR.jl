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
