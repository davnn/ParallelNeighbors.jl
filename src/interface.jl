guess_batchsize(X, k) = max(trunc(Int, size(X, 2)^(1 / sqrt(2))), k)

"""
    knn(Xtrain, Xtest, k, batch_size; metric, algorithm)

Interface to all parallel implementations to determine the nearest neighbors of a number of points, where each column
in `Xtrain` and `Xtest` corresponds to a point.

Parameters
----------
$shared_parameters

    algorithm::Symbol
The specific algorithm to use to determine the nearest neighbors.
"""
function knn(Xtrain::M, Xtest::M, k::Int, batch_size::Int=guess_batchsize(Xtest, k);
    metric::PreMetric=Euclidean(), algorithm=:hybrid_batch_all) where {M<:Union{host_input,device_input}}
    valid_algorithms = [:hybrid_batch_all, :hybrid_batch_test, :full]
    train_rows, train_cols = size(Xtrain)
    test_rows, test_cols = size(Xtest)

    err_matrix(train) = "The $(train ? "train" : "test") input matrix must contain at least one feature and sample."
    @assert train_rows > 0 && train_cols > 0 err_matrix(true)
    @assert test_rows > 0 && test_cols > 0 err_matrix(false)

    if algorithm === :hybrid_batch_all
        return knn_hybrid_batch_all(Xtrain, Xtest, k, batch_size; metric)
    elseif algorithm === :hybrid_batch_test
        return knn_hybrid_batch_test(Xtrain, Xtest, k, batch_size; metric)
    elseif algorithm === :full
        return knn_full(Xtrain, Xtest, k; metric)
    else
        throw(AssertionError("Argument 'algorithm' must be one of $valid_algorithms"))
    end
end
