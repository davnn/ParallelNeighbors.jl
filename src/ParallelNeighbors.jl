module ParallelNeighbors

using CUDA
using Distances: PreMetric, Euclidean, SqEuclidean

const host_input = Matrix{<:AbstractFloat}
const device_input = CuMatrix{<:AbstractFloat}

export knn

include("./distance.jl")
include("./knn.jl")
include("./interface.jl")

function __init__()
    if get(ENV, "JULIA_REGISTRYCI_AUTOMERGE", "false") != "true"
        @assert CUDA.functional(true)
    end
end

end
