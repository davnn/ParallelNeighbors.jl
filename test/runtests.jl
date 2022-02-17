using Test
import ParallelNeighbors
using Random: MersenneTwister, shuffle!
using NearestNeighbors: BruteTree, knn, Euclidean, SqEuclidean
using LinearAlgebra: LowerTriangular, Diagonal
using MultivariateStats: fit, predict, MDS
using Suppressor: @suppress_err

# TODO: change to `as` import once we drop Julia 1.5 support
const knn_parallel = ParallelNeighbors.knn

function create_unique_data(dim, size, rng = MersenneTwister(0))
    # Use MDS to calculate samples that are of pairwise-unique distance to each other
    # https://www.stat.pitt.edu/sungkyu/course/2221Fall13/lec8_mds_combined.pdf
    base_vector = Float64.(collect(1:size*size))
    unique_dists = shuffle!(rng, base_vector) # unique distances
    Dfull = reshape(unique_dists, size, size)
    D = LowerTriangular(Dfull)
    D = D + D' - Diagonal(D)
    # we don't care if some of the eigenvalues are degenerate in testing
    mds = @suppress_err fit(MDS, D; distances = true, maxoutdim = dim)
    return predict(mds)
end

function create_unique_split(dim, n_train, n_test)
    X = create_unique_data(dim, n_train + n_test)
    return X[:, 1:n_train], X[:, n_train+1:end]
end

function reference_knn(Xtrain, Xtest, k; metric = Euclidean())
    knn(BruteTree(Xtrain, metric), Xtest, k, true)
end

@testset "ParallelNeighbors.jl" begin
    ks = 1:5
    dims = 1:3
    batch_sizes = 1:5
    train_sizes = 1:8
    test_sizes = 1:8

    # we cannot sample less than k training points (otherwise how should we determine the k nearest neighbors)
    for k in ks, dim in dims,
        batch_size in batch_sizes,
        test_size in test_sizes,
        train_size in train_sizes[k:end]

        # test distances with discrete data and indices with continuous data
        Xtrain, Xtest = create_unique_split(dim, train_size, test_size)

        pointwise_batch_size = min(batch_size, test_size)
        batched_batch_size = max(min(batch_size, train_size), k)

        # evaluate with euclidean distance
        results_euclidean = (reference_knn(Xtrain, Xtest, k; metric = Euclidean()),
            knn_parallel(Xtrain, Xtest, k; metric = Euclidean(), algorithm = :full),
            knn_parallel(Xtrain, Xtest, k, pointwise_batch_size; metric = Euclidean(), algorithm = :hybrid_batch_test),
            knn_parallel(Xtrain, Xtest, k, batched_batch_size; metric = Euclidean(), algorithm = :hybrid_batch_all))

        # TODO: evaluate with squared euclidean distance once the following pull request is merged
        # https://github.com/KristofferC/NearestNeighbors.jl/pull/137
        # results_sqeuclidean = (reference_knn(Xtrain, Xtest, k; metric = SqEuclidean()),
        #     knn_full(Xtrain, Xtest, k; metric = SqEuclidean()),
        #     knn_pointwise(Xtrain, Xtest, k, pointwise_batch_size; metric = SqEuclidean()),
        #     knn_batch(Xtrain, Xtest, k, batched_batch_size; metric = SqEuclidean()))

        ref_idx, full_idx, point_idx, batch_idx = map(first, results_euclidean)
        @test all(ref_idx .== point_idx)
        @test all(ref_idx .== batch_idx)
        @test all(ref_idx .== full_idx)

        ref_dist, full_dist, point_dist, batch_dist = map(last, results_euclidean)
        @test all(ref_dist .≈ point_dist)
        @test all(ref_dist .≈ batch_dist)
        @test all(ref_dist .≈ full_dist)
    end
end
