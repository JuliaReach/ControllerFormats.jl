# valid sample file
file = joinpath(@__DIR__, "sample_MAT.mat")

# parse file
N = read_MAT(file; act_key="act_fcns")

@test length(N.layers) == 4

# alternative file with 4D weights
file = joinpath(@__DIR__, "sample_MAT2.mat")

# parse file
N = read_MAT(file; act_key="act_fcns")

# alternative file with 4D weights but invalid dimensions
file = joinpath(@__DIR__, "sample_MAT_invalid.mat")

# parse file
@test_throws ArgumentError read_MAT(file; act_key="act_fcns")

@test length(N.layers) == 4

# alternative file with nested dictionary
file = joinpath(@__DIR__, "sample_MAT3.mat")

N = read_MAT(file; act_key="activation_fcns", net_key="network")

@test length(N.layers) == 4
