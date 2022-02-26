shared_parameters = """    Xtrain::M where M <: AbstractMatrix{<:AbstractFloat}
Matrix of training points.

    Xtest::M where M <: AbstractMatrix{<:AbstractFloat}
Matrix of test points.

    k::Int
Nuumber of nearest neighbors to find in the training points.

    batch_size::Int
Batch size to split the training and/or test data.

    metric::PreMetric
(Pre-) Metric to use for distance computation.

    convert::Function
Conversion function to convert the input data (`Xtrain` and `Xtest`) to an appropriate format, e.g. `CUDA.cu`."""

"""
    knn_hybrid_batch_test(Xtrain, Xtest, k, batch_size; metric, convert)

A parallel KNN implementation to determine the nearest neighbors of a number of points, where each column in  `Xtrain`
and `Xtest` corresponds to a point. This implementation batches points of `Xtest`, and calculates the nearest neighbors
to all points in `Xtrain`.

Parameters
----------
$shared_parameters
"""
function knn_hybrid_batch_test(Xtrain::M, Xtest::M, k::Int, batch_size::Int = guess_batchsize(Xtest, k);
    metric::PreMetric = Euclidean(), convert::Function = identity, pin::Function = identity) where {M<:input_type}
    n_train = size(Xtrain, 2)
    n_test = size(Xtest, 2)
    @assert batch_size <= n_test "Batch size must be smaller or equal the number of test points."

    batches = Iterators.partition(1:n_test, batch_size)
    idxs = [Vector{Int}(undef, k) for _ in 1:n_test]
    dists = [Vector{eltype(Xtrain)}(undef, k) for _ in 1:n_test]
    workspace_idxs = [collect(1:n_train) for _ in 1:batch_size]

    # Convert train points according to conversion function, e.g. `cu`
    Xtrain = convert(Xtrain)

    mcpu_odd, mcpu_even = [pin(Matrix{eltype(Xtrain)}(undef, n_train, batch_size)) for _ in 1:2]
    mgpu_odd, mgpu_even = [similar(Xtrain, n_train, batch_size) for _ in 1:2]
    sort_finished = Task(() -> nothing) |> schedule

    # Convert the metric to a kernel
    distance_kernel = metric_to_kernel(metric)

    for (i, batch) in enumerate(batches)
        n_batch = length(batch)
        mgpu, mcpu = iseven(i) ? (mgpu_even, mcpu_even) : (mgpu_odd, mcpu_odd)
        mgpu[:, 1:n_batch] .= distance_kernel(convert(Xtest[:, batch]), Xtrain)
        wait(sort_finished)
        sort_finished = @async begin
            copyto!(mcpu, mgpu)
            cpu_colsort!(idxs, dists, workspace_idxs, mcpu, batch_size, n_batch, i, k)
        end
    end
    wait(sort_finished)
    return idxs, dists
end

"""
    knn_hybrid_batch_all(Xtrain, Xtest, k, batch_size; metric, convert)

A mixed GPU/CPU implementation to determine the nearest neighbors of a number of points, where each column in  `Xtrain`
and `Xtest` corresponds to a point. This implementation batches points of `Xtrain` and `Xtest` simultaneously, saves the
temporary nearest neighbors and reduces the temporary neighbors after all other points in `Xtrain` have been observed.

Parameters
----------
$shared_parameters
"""
function knn_hybrid_batch_all(Xtrain::M, Xtest::M, k::Int, batch_size::Int = guess_batchsize(Xtest, k);
    metric::PreMetric = Euclidean(), convert::Function = identity, pin::Function = identity) where {M<:input_type}
    n_train = size(Xtrain, 2)
    n_test = size(Xtest, 2)
    @assert batch_size >= k "Batch size must be larger or equal to k."
    @assert batch_size <= n_train "Batch size must be smaller or equal the number of training points."

    # generate the partitions
    train_batches = Iterators.partition(1:n_train, batch_size)
    test_batches = Iterators.partition(1:n_test, batch_size)

    # get sizes to preallocate the temporary store of all possible nearest neighbors
    temp_length = map(train_batches) do batch
        len = length(batch)
        len > k ? k : len
    end |> sum

    # pre-allocate the final outputs
    idxs = [Vector{Int}(undef, k) for _ in 1:n_test]
    dists = [Vector{eltype(Xtrain)}(undef, k) for _ in 1:n_test]

    # pre-allocate the temporary store of the distance matrix
    mcpu_odd, mcpu_even = [pin(Matrix{eltype(Xtrain)}(undef, batch_size, batch_size)) for _ in 1:2]
    mgpu_odd, mgpu_even = [similar(convert(Xtrain[1:1]), batch_size, batch_size) for _ in 1:2]

    # pre-allocate the temporary store of the indices and distances
    temp_idxs = Matrix{Int}(undef, temp_length, n_test)
    temp_dists = Matrix{eltype(Xtrain)}(undef, temp_length, n_test)

    # pre-allocate the temporary sorting indices (there are never more than batch_size indices needed)
    outer_workspace = [collect(1:temp_length) for _ in 1:n_test]
    inner_workspace = [collect(1:batch_size) for _ in 1:batch_size]

    # this will hold a task that determinies if sorting is already done
    sort_finished = Task(() -> nothing) |> schedule
    reduction_finished = Task(() -> nothing) |> schedule

    # Convert the metric to a kernel
    distance_kernel = metric_to_kernel(metric)

    for batch_test in test_batches
        n_batch_test = length(batch_test)
        for (idx_train, batch_train) in enumerate(train_batches)
            n_train_batch = length(batch_train)
            mgpu, mcpu = iseven(idx_train) ? (mgpu_even, mcpu_even) : (mgpu_odd, mcpu_odd)

            # calculate the distance matrix, it's important that we restrict the pre-allocated matrix to the
            # possible points in the batch (which might be less then the full `batchsize x batchsize` matrix)
            mgpu[1:n_train_batch, 1:n_batch_test] .=
                distance_kernel(convert(Xtest[:, batch_test]), convert(Xtrain[:, batch_train]))

            # we calculate the distances in parallel, but we need to wait for the sorting before we can copy
            wait(sort_finished)

            sort_finished = @async begin
                copyto!(mcpu, mgpu)
                # prepare ranges of indices for the currently calculated neighbors, batches might be smaller than `k`
                nbatch_or_k = n_train_batch > k ? k : n_train_batch
                range_lower = (idx_train - 1) * k + 1
                range_upper = range_lower + nbatch_or_k - 1

                # we are working with a fixed-size square matrix of `batch_size x batch_size`, but only
                # some values might be used for the last batches (that are smaller than batch_size)
                cpu_batchsort!(temp_idxs, temp_dists, inner_workspace, mcpu,
                    range_lower:range_upper, batch_train, batch_test, n_train_batch, n_batch_test, nbatch_or_k)

                nothing # JuliaLang/julia#40626
            end
        end
        # Note: It's important to wait for the tasks in the corresponding loops the are started from, otherwise (only
        # waiting after the loop) it is not guaranteed that all tasks finished, but only the last one.
        wait(sort_finished)
        wait(reduction_finished)
        reduction_finished = async_cpu_reduction!(idxs, dists, temp_idxs, temp_dists, outer_workspace, batch_test, k)
    end
    wait(reduction_finished)
    return idxs, dists
end

"""
    knn_full(Xtrain, Xtest, k, batch_size, distance_kernel)

A trivial implementation to determine the nearest neighbors of a number of points, where each column in  `Xtrain` and
`Xtest` corresponds to a point. This implementation calculates all distances at once, sorting afterwards. Note: because
there is no `partialsortperm` GPU kernel available, this implementation inefficiently uses `sortperm` to sort the
resulting distance matrix.

Parameters
----------
    Xtrain::M where M <: AbstractGPUArray{<:AbstractFloat}
Matrix of training points.

    Xtest::M where M <: AbstractGPUArray{<:AbstractFloat}
Matrix of test points.

    k::Int
Nuumber of nearest neighbors to find in the training points.

    metric::PreMetric
Metric to use for distance computation.

    convert::Function
Conversion function to convert the input data (`Xtrain` and `Xtest`) to an appropriate format, e.g. `CUDA.cu`. In this
implementation, both `Xtrain` and `Xtest` are converted beforehand.
"""
function knn_full(Xtrain::M, Xtest::M, k::Int;
    metric::PreMetric = Euclidean(), convert::Function = identity) where {M<:input_type}
    n_test = size(Xtest, 2)
    idxs = [Vector{Int}(undef, k) for _ in 1:n_test]
    dists = [Vector{eltype(convert(Xtrain[1:1]))}(undef, k) for _ in 1:n_test]
    distance_kernel = metric_to_kernel(metric)
    distance_matrix = distance_kernel(convert(Xtest), convert(Xtrain))
    colsort!(idxs, dists, distance_matrix, k)
    return idxs, dists
end

function colsort!(idxs, dists, distance_matrix, k)
    for (i, col) in enumerate(eachcol(distance_matrix))
        # TODO: currently, there is no partialsortperm kernel available
        sortres = view(sortperm(col), 1:k)
        copyto!(idxs[i], sortres)
        copyto!(dists[i], col[sortres])
    end
end

function cpu_colsort!(idxs, dists, workspace, distance_matrix, batch_size, n_batch, current_batch, k)
    return Threads.@threads for col in 1:n_batch
        idx = ((current_batch - 1) * batch_size) + col
        sorted_idxs = partialsortperm!(workspace[col],
            view(distance_matrix, :, col), 1:k, rev = false, initialized = true)
        copyto!(idxs[idx], sorted_idxs)
        copyto!(dists[idx], view(distance_matrix, sorted_idxs, col))
        nothing # JuliaLang/julia#40626
    end
end

function cpu_batchsort!(idxs, dists, workspace, distance_matrix, idx_range, batch_train, batch_test,
    n_batch_train, n_batch_test, n_or_k_j)
    return Threads.@threads for col in 1:n_batch_test
        # TODO: this sort could probably be sped up, we currently have to rely on resizing and initializing
        # because not all batches must have the same size, if all have the same size, however we can assume
        # `initalized = true` and don't need to resize
        idx = partialsortperm!(resize!(workspace[col], n_batch_train),
            view(distance_matrix, 1:n_batch_train, col), 1:n_or_k_j, rev = false, initialized = false)
        @inbounds idxs[idx_range, batch_test[col]] .= view(batch_train, idx)
        @inbounds dists[idx_range, batch_test[col]] .= view(distance_matrix, idx, col)
        nothing # JuliaLang/julia#40626
    end
end

function async_cpu_reduction!(idxs, dists, temp_idxs, temp_dists, workspace, batch_test, k)
    return @async Threads.@threads for i in batch_test
        idx = partialsortperm!(workspace[i], view(temp_dists, :, i), 1:k, rev = false, initialized = true)
        copyto!(idxs[i], temp_idxs[idx, i])
        copyto!(dists[i], temp_dists[idx, i])
        nothing # JuliaLang/julia#40626
    end
end
