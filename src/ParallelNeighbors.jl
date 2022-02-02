module ParallelNeighbors

using Distances: PreMetric, Euclidean, SqEuclidean

const input_type = AbstractMatrix{<:AbstractFloat}
export knn_pointwise, knn_batch, knn_full

include("./distance.jl")
include("./knn.jl")

end
