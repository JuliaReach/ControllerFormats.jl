# AbstractLayerOp implementation
struct TestLayerOp <: AbstractLayerOp end
L = TestLayerOp()
dim_in(L)
dim_out(L)
