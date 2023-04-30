# valid sample file from Sherlock manual
file = joinpath(@__DIR__, "sample_Sherlock")

# parse file
N = read_Sherlock(file)

@test dim_in(N) == 2
@test dim_out(N.layers[1]) == 2
@test dim_out(N) == 1


# write network back to file and re-read it
file = joinpath(@__DIR__, "sample_Sherlock_output")

write_Sherlock(N, file)
N2 = read_Sherlock(file)
rm(file)

@test N == N2
