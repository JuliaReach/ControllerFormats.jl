const available_activations = Dict(
                                   # Id
                                   "Id" => Architecture._id,
                                   "linear" => Architecture._id,
                                   "Linear" => Architecture._id,
                                   "Affine" => Architecture._id,
                                   # ReLU
                                   "relu" => Architecture._relu,
                                   "ReLU" => Architecture._relu,
                                   # Sigmoid
                                   "sigmoid" => Architecture._sigmoid,
                                   "Sigmoid" => Architecture._sigmoid,
                                   "σ" => Architecture._sigmoid,
                                   "logsig" => Architecture._sigmoid,
                                   # Tanh
                                   "tanh" => Architecture._tanh,
                                   "Tanh" => Architecture._tanh)
