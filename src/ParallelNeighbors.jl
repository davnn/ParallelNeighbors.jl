module ParallelNeighbors

using Distances

const input_type = AbstractMatrix{<:AbstractFloat}
export knn_pointwise, knn_batch, knn_full

include("./distance.jl")
include("./knn.jl")

end
