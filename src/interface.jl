guess_batchsize(X, k) = max(trunc(Int, size(X, 2)^(1 / sqrt(2))), k)

"""
    knn(Xtrain, Xtest, k, batch_size; metric, convert, algorithm)

Interface to all parallel implementations to determine the nearest neighbors of a number of points, where each column
in `Xtrain` and `Xtest` corresponds to a point.

Parameters
----------
$shared_parameters

    algorithm::Symbol
The specific algorithm to use to determine the nearest neighbors.
"""
function knn(Xtrain::M, Xtest::M, k::Int, batch_size::Int = guess_batchsize(Xtest, k);
    metric::PreMetric = Euclidean(), convert::Function = identity, pin::Function = identity,
    algorithm = :hybrid_batch_all) where {M<:input_type}
    valid_algorithms = [:hybrid_batch_all, :hybrid_batch_test, :full]

    if algorithm === :hybrid_batch_all
        return knn_hybrid_batch_all(Xtrain, Xtest, k, batch_size; metric, convert, pin)
    elseif algorithm === :hybrid_batch_test
        return knn_hybrid_batch_test(Xtrain, Xtest, k, batch_size; metric, convert, pin)
    elseif algorithm === :full
        return knn_full(Xtrain, Xtest, k; metric, convert)
    else
        throw(AssertionError("Argument 'algorithm' must be one of $valid_algorithms"))
    end
end
