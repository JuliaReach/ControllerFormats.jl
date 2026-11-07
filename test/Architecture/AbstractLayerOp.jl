# AbstractLayerOp implementation
struct TestLayerOp <: AbstractLayerOp end

ControllerFormats.dim_in(::TestLayerOp) = 1
ControllerFormats.dim_out(::TestLayerOp) = 2

L = TestLayerOp()
@test dim(L) == (1, 2)
