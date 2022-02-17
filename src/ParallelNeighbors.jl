module ParallelNeighbors

using Distances: PreMetric, Euclidean, SqEuclidean

const input_type = AbstractMatrix{<:AbstractFloat}

export knn

include("./distance.jl")
include("./knn.jl")
include("./interface.jl")

end
