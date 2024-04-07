L = FlattenLayerOp()

# output for tensor `T`
@test L([1 2; 3 4]) == [1, 2, 3, 4]
@test L([1]) == [1]

# equality
@test L == FlattenLayerOp()
