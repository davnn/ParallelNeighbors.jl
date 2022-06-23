const ERR_DIM_EQUAL = "The dimensionality of the training and test data must be equal"
return_type(Xtrain) = Tuple{Vector{Vector{Int}},Vector{Vector{eltype(Xtrain)}}}

# generate the partitions
function pin_partitions(partitions, data)
    v = Vector{typeof(data)}(undef, length(partitions))
    Threads.@threads for i in eachindex(partitions)
        v[i] = CUDA.Mem.pin(data[:, partitions[i]])
    end
    return v
end

# return the length of the temporary index and distance arrays
function calc_temp_length(train_batches, k)
    temp_length = map(train_batches) do batch
        len = length(batch)
        len > k ? k : len
    end |> sum
    return temp_length
end

# pre-allocate the data structures for the results
function allocate_results(data_type, k, n_test)
    idxs = Threads.@spawn [Vector{Int}(undef, k) for _ in 1:n_test]
    dists = Threads.@spawn [Vector{data_type}(undef, k) for _ in 1:n_test]
    return idxs, dists
end

# pre-allocate the temporary store of the indices and distances
function allocate_temp_results(data_type, temp_length, n_test)
    temp_idxs = Threads.@spawn Matrix{Int}(undef, temp_length, n_test)
    temp_dists = Threads.@spawn Matrix{data_type}(undef, temp_length, n_test)
    return temp_idxs, temp_dists
end

# pre-allocate sorting workspace
function allocate_workspace(n_train, batch_size)
    workspace_idxs = Threads.@spawn [collect(1:n_train) for _ in 1:batch_size]
    return workspace_idxs
end

# pre-allocate the temporary sorting indices (there are never more than batch_size indices needed)
function allocate_workspace(temp_length, n_test, batch_size)
    outer_workspace = Threads.@spawn [collect(1:temp_length) for _ in 1:n_test]
    inner_workspace = Threads.@spawn [collect(1:batch_size) for _ in 1:batch_size]
    return outer_workspace, inner_workspace
end

# pre-allocate the distance matrices
function allocate_distance_matrices(n_dim1, n_dim2, data_type)
    mcpus = [CUDA.Mem.pin(Matrix{data_type}(undef, n_dim1, n_dim2)) for _ in 1:2]
    mgpus = [CUDA.CuMatrix{data_type}(undef, n_dim1, n_dim2) for _ in 1:2]
    return mcpus, mgpus
end

function assert_batch_test(batch_size, n_test, dim_train, dim_test)
    @assert batch_size <= n_test "Batch size must be smaller or equal the number of test points."
    @assert dim_train == dim_test ERR_DIM_EQUAL
end

function assert_batch_all(batch_size, k, n_train, dim_train, dim_test)
    @assert batch_size >= k "Batch size must be larger or equal to k."
    @assert batch_size <= n_train "Batch size must be smaller or equal the number of training points."
    @assert dim_train == dim_test ERR_DIM_EQUAL
end

shared_parameters = """    Xtrain::Union{CuMatrix{<:AbstractFloat}, Matrix{<:AbstractFloat}}
Matrix of training points.

    Xtest::Union{CuMatrix{<:AbstractFloat}, Matrix{<:AbstractFloat}}
Matrix of test points.

    k::Int
Nuumber of nearest neighbors to find in the training points.

    batch_size::Int
Batch size to split the training and/or test data.

    metric::PreMetric
(Pre-) Metric to use for distance computation."""

"""
    knn_hybrid_batch_test(Xtrain, Xtest, k, batch_size; metric, convert)

A parallel KNN implementation to determine the nearest neighbors of a number of points, where each column in  `Xtrain`
and `Xtest` corresponds to a point. This implementation batches points of `Xtest`, and calculates the nearest neighbors
to all points in `Xtrain`.

Parameters
----------
$shared_parameters
"""
function knn_hybrid_batch_test(
    Xtrain::M,
    Xtest::M,
    k::Int,
    batch_size::Int=guess_batchsize(Xtest, k);
    metric::PreMetric=Euclidean()
)::return_type(Xtrain) where {M<:host_input}
    (dim_train, n_train), (dim_test, n_test), data_type = size(Xtrain), size(Xtest), eltype(Xtrain)
    assert_batch_test(batch_size, n_test, dim_train, dim_test)

    # convert training data to GPU
    Xtrain = CUDA.cu(CUDA.Mem.pin(Xtrain))
    Xtest = CUDA.Mem.pin(Xtest)

    # pre-allocate result data structures
    idxs, dists = allocate_results(data_type, k, n_test)

    # pre-allocate sorting workspace
    workspace_idxs = allocate_workspace(n_train, batch_size)

    # pre-allocate distance matrices
    (mcpu_odd, mcpu_even), (mgpu_odd, mgpu_even) = allocate_distance_matrices(n_train, batch_size, data_type)

    # convert the metric to a kernel
    distance_kernel = metric_to_kernel(metric)

    # batch and pin the test data
    test_batches = Iterators.partition(1:n_test, batch_size) |> collect

    # pin the memory of the partitions
    test_batch_data = pin_partitions(test_batches, Xtest)

    # start the global tasks
    convert_test_finished = @async CUDA.@sync CUDA.cu(first(test_batch_data))
    sort_finished = @async nothing

    for (idx_test, batch) in enumerate(test_batches)
        n_batch = length(batch)
        mgpu, mcpu = iseven(idx_test) ? (mgpu_even, mcpu_even) : (mgpu_odd, mcpu_odd)

        # wait and fetch previous gpu-converted data and async start new conversion
        Xtest_prev = fetch(convert_test_finished)
        convert_test_finished = @async CUDA.@sync CUDA.cu(test_batch_data[idx_test+1])

        # calculate distances and save to pre-allocated memory
        CUDA.@sync @inbounds mgpu[:, 1:n_batch] .= distance_kernel(Xtest_prev, Xtrain)
        CUDA.unsafe_free!(Xtest_prev)

        # wait until sorting is finished and start next sort
        wait(sort_finished)
        sort_finished = @async begin
            copyto!(mcpu, mgpu)
            cpu_colsort!(fetch(idxs), fetch(dists), fetch(workspace_idxs), mcpu, batch_size, n_batch, idx_test, k)
        end
    end
    wait(sort_finished)
    return fetch(idxs), fetch(dists)
end
function knn_hybrid_batch_test(
    Xtrain::M,
    Xtest::M,
    k::Int,
    batch_size::Int=guess_batchsize(Xtest, k);
    metric::PreMetric=Euclidean()
)::return_type(Xtrain) where {M<:device_input}
    (dim_train, n_train), (dim_test, n_test), data_type = size(Xtrain), size(Xtest), eltype(Xtrain)
    assert_batch_test(batch_size, n_test, dim_train, dim_test)

    # pre-allocate result data structures
    idxs, dists = allocate_results(data_type, k, n_test)

    # pre-allocate sorting workspace
    workspace_idxs = allocate_workspace(n_train, batch_size)

    # pre-allocate distance matrices
    (mcpu_odd, mcpu_even), (mgpu_odd, mgpu_even) = allocate_distance_matrices(n_train, batch_size, data_type)

    # convert the metric to a kernel
    distance_kernel = metric_to_kernel(metric)

    # Batch the test data
    test_batches = Iterators.partition(1:n_test, batch_size) |> collect

    # Prepare the tasks
    sort_finished = @async nothing

    for (idx_test, batch) in enumerate(test_batches)
        n_batch = length(batch)
        mgpu, mcpu = iseven(idx_test) ? (mgpu_even, mcpu_even) : (mgpu_odd, mcpu_odd)

        # calculate distances and save to pre-allocated memory
        CUDA.@sync @inbounds mgpu[:, 1:n_batch] .= distance_kernel(Xtest[:, batch], Xtrain)

        # wait until sorting is finished and start next sort
        wait(sort_finished)
        sort_finished = @async begin
            copyto!(mcpu, mgpu)
            cpu_colsort!(fetch(idxs), fetch(dists), fetch(workspace_idxs), mcpu, batch_size, n_batch, idx_test, k)
        end
    end
    wait(sort_finished)
    return fetch(idxs), fetch(dists)
end

"""
    knn_hybrid_batch_all(Xtrain, Xtest, k, batch_size; metric)

A mixed GPU/CPU implementation to determine the nearest neighbors of a number of points, where each column in  `Xtrain`
and `Xtest` corresponds to a point. This implementation batches points of `Xtrain` and `Xtest` simultaneously, saves the
temporary nearest neighbors and reduces the temporary neighbors after all other points in `Xtrain` have been observed.

Parameters
----------
$shared_parameters
"""
function knn_hybrid_batch_all(Xtrain::M, Xtest::M, k::Int, batch_size::Int=guess_batchsize(Xtest, k);
    metric::PreMetric=Euclidean())::return_type(Xtrain) where {M<:host_input}
    (dim_train, n_train), (dim_test, n_test), data_type = size(Xtrain), size(Xtest), eltype(Xtrain)
    assert_batch_all(batch_size, k, n_train, dim_train, dim_test)

    # partition the train and test data according to the given batch size
    train_batches = Iterators.partition(1:n_train, batch_size) |> collect
    test_batches = Iterators.partition(1:n_test, batch_size) |> collect

    # pin the memory of the partitions
    test_batch_data = pin_partitions(test_batches, Xtest)
    train_batch_data = pin_partitions(train_batches, Xtrain)

    # pre-allocate result data structures
    idxs, dists = allocate_results(data_type, k, n_test)

    # calculate the size for the temporary data structures
    temp_length = calc_temp_length(train_batches, k)

    # pre-allocate the temporary result data structures
    temp_idxs, temp_dists = allocate_temp_results(data_type, temp_length, n_test)

    # pre-allocate sorting workspace
    outer_workspace, inner_workspace = allocate_workspace(temp_length, n_test, batch_size)

    # pre-allocate distance matrices
    (mcpu_odd, mcpu_even), (mgpu_odd, mgpu_even) = allocate_distance_matrices(batch_size, batch_size, data_type)

    # convert the metric to a kernel
    distance_kernel = metric_to_kernel(metric)

    # pre-calculate some tasks used throughout the algorithm
    sort_finished = @async nothing
    reduction_finished = @async nothing
    convert_test_finished = @async CUDA.@sync CUDA.cu(first(test_batch_data))
    convert_train_finished = @async CUDA.@sync CUDA.cu(first(train_batch_data))
    n_train_batches = length(train_batch_data)

    # Convert the metric to a kernel
    distance_kernel = metric_to_kernel(metric)

    for (idx_test, batch_test) in enumerate(test_batches)
        n_batch_test = length(batch_test)
        Xtest_prev = fetch(convert_test_finished)
        convert_test_finished = @async CUDA.@sync CUDA.cu(test_batch_data[idx_test+1]) # last value is never fetched

        for (idx_train, batch_train) in enumerate(train_batches)
            n_batch_train = length(batch_train)
            mgpu, mcpu = iseven(idx_train) ? (mgpu_even, mcpu_even) : (mgpu_odd, mcpu_odd)

            # begin async copy of next train batch data
            Xtrain_prev = fetch(convert_train_finished)
            convert_train_finished = @async begin
                idx_train == n_train_batches && return CUDA.@sync CUDA.cu(first(train_batch_data))
                return CUDA.@sync CUDA.cu(train_batch_data[idx_train+1])
            end

            # calculate the distance matrix, it's important that we restrict the pre-allocated matrix to the
            # possible points in the batch (which might be less then the full `batchsize x batchsize` matrix)
            CUDA.@sync @inbounds mgpu[1:n_batch_train, 1:n_batch_test] .= distance_kernel(Xtest_prev, Xtrain_prev)
            CUDA.unsafe_free!(Xtrain_prev)

            # we calculate the distances in parallel, but we need to wait for the sorting before we can copy
            wait(sort_finished)

            sort_finished = @async begin
                copyto!(mcpu, mgpu) # DtoH copy automatically syncs the device
                cpu_batchsort!(fetch(temp_idxs), fetch(temp_dists), fetch(inner_workspace), mcpu,
                    batch_train, batch_test, n_batch_train, n_batch_test, k, idx_train)
            end
        end
        CUDA.unsafe_free!(Xtest_prev)

        # Note: It's important to wait for the tasks in the corresponding loops the are started from, otherwise (only
        # waiting after the loop) it is not guaranteed that all tasks finished, but only the last one.
        wait(sort_finished)
        wait(reduction_finished)
        reduction_finished = async_cpu_reduction!(fetch(idxs), fetch(dists), fetch(temp_idxs),
            fetch(temp_dists), fetch(outer_workspace), batch_test, k)
    end
    wait(reduction_finished)
    return fetch(idxs), fetch(dists)
end
function knn_hybrid_batch_all(Xtrain::M, Xtest::M, k::Int, batch_size::Int=guess_batchsize(Xtest, k);
    metric::PreMetric=Euclidean())::return_type(Xtrain) where {M<:device_input}
    (dim_train, n_train), (dim_test, n_test), data_type = size(Xtrain), size(Xtest), eltype(Xtrain)
    assert_batch_all(batch_size, k, n_train, dim_train, dim_test)

    # partition the train and test data according to the given batch size
    train_batches = Iterators.partition(1:n_train, batch_size) |> collect
    test_batches = Iterators.partition(1:n_test, batch_size) |> collect

    # pre-allocate result data structures
    idxs, dists = allocate_results(data_type, k, n_test)

    # calculate the size for the temporary data structures
    temp_length = calc_temp_length(train_batches, k)

    # pre-allocate the temporary result data structures
    temp_idxs, temp_dists = allocate_temp_results(data_type, temp_length, n_test)

    # pre-allocate sorting workspace
    outer_workspace, inner_workspace = allocate_workspace(temp_length, n_test, batch_size)

    # pre-allocate distance matrices
    (mcpu_odd, mcpu_even), (mgpu_odd, mgpu_even) = allocate_distance_matrices(batch_size, batch_size, data_type)

    # convert the metric to a kernel
    distance_kernel = metric_to_kernel(metric)

    # pre-calculate some tasks used throughout the algorithm
    sort_finished = @async nothing
    reduction_finished = @async nothing

    # Convert the metric to a kernel
    distance_kernel = metric_to_kernel(metric)

    for batch_test in test_batches
        n_batch_test = length(batch_test)

        for (idx_train, batch_train) in enumerate(train_batches)
            n_batch_train = length(batch_train)
            mgpu, mcpu = iseven(idx_train) ? (mgpu_even, mcpu_even) : (mgpu_odd, mcpu_odd)

            # calculate the distance matrix, it's important that we restrict the pre-allocated matrix to the
            # possible points in the batch (which might be less then the full `batchsize x batchsize` matrix)
            CUDA.@sync @inbounds mgpu[1:n_batch_train, 1:n_batch_test] .=
                distance_kernel(Xtest[:, batch_test], Xtrain[:, batch_train])

            # we calculate the distances in parallel, but we need to wait for the sorting before we can copy
            wait(sort_finished)

            sort_finished = @async begin
                copyto!(mcpu, mgpu) # DtoH copy automatically syncs the device
                cpu_batchsort!(fetch(temp_idxs), fetch(temp_dists), fetch(inner_workspace), mcpu,
                    batch_train, batch_test, n_batch_train, n_batch_test, k, idx_train)
            end
        end

        # Note: It's important to wait for the tasks in the corresponding loops the are started from, otherwise (only
        # waiting after the loop) it is not guaranteed that all tasks finished, but only the last one.
        wait(sort_finished)
        wait(reduction_finished)
        reduction_finished = async_cpu_reduction!(fetch(idxs), fetch(dists), fetch(temp_idxs),
            fetch(temp_dists), fetch(outer_workspace), batch_test, k)
    end
    wait(reduction_finished)
    return fetch(idxs), fetch(dists)
end

"""
    knn_full(Xtrain, Xtest, k, batch_size, distance_kernel)

A trivial implementation to determine the nearest neighbors of a number of points, where each column in  `Xtrain` and
`Xtest` corresponds to a point. This implementation calculates all distances at once, sorting afterwards. Note: because
there is no `partialsortperm` GPU kernel available, this implementation inefficiently uses `sortperm` to sort the
resulting distance matrix.

Parameters
----------
$shared_parameters
"""
function knn_full(Xtrain::M, Xtest::M, k::Int;
    metric::PreMetric=Euclidean())::return_type(Xtrain) where {M<:host_input}
    Xtrain, Xtest = CUDA.cu(Xtrain), CUDA.cu(Xtest)
    knn_full(Xtrain, Xtest, k; metric)
end
function knn_full(Xtrain::M, Xtest::M, k::Int;
    metric::PreMetric=Euclidean())::return_type(Xtrain) where {M<:device_input}
    n_test = size(Xtest, 2)
    idxs = [Vector{Int}(undef, k) for _ in 1:n_test]
    dists = [Vector{eltype(Xtrain)}(undef, k) for _ in 1:n_test]
    distance_kernel = metric_to_kernel(metric)
    distance_matrix = distance_kernel(Xtest, Xtrain)
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
            view(distance_matrix, :, col), 1:k, rev=false, initialized=true)
        copyto!(idxs[idx], sorted_idxs)
        copyto!(dists[idx], view(distance_matrix, sorted_idxs, col))
        nothing # JuliaLang/julia#40626
    end
end

function cpu_batchsort!(idxs, dists, workspace, distance_matrix, batch_train, batch_test,
    n_batch_train, n_batch_test, k, idx_train)
    # Note: we are working with a fixed-size square matrix of `batch_size x batch_size`, but only
    # some values might be used for the last batches (that are smaller than batch_size)

    # prepare ranges of indices for the currently calculated neighbors, batches might be smaller than `k`
    nbatch_or_k = n_batch_train > k ? k : n_batch_train
    range_lower = (idx_train - 1) * k + 1
    range_upper = range_lower + nbatch_or_k - 1
    idx_range = range_lower:range_upper

    return Threads.@threads for col in 1:n_batch_test
        # TODO: this sort could probably be sped up, we currently have to rely on resizing and initializing
        # because not all batches must have the same size, if all have the same size, however we can assume
        # `initalized = true` and don't need to resize
        idx = partialsortperm!(resize!(workspace[col], n_batch_train),
            view(distance_matrix, 1:n_batch_train, col), 1:nbatch_or_k, rev=false, initialized=false)
        @inbounds idxs[idx_range, batch_test[col]] .= view(batch_train, idx)
        @inbounds dists[idx_range, batch_test[col]] .= view(distance_matrix, idx, col)
        nothing # JuliaLang/julia#40626
    end
end

# TODO: should we launch a thread here?
function async_cpu_reduction!(idxs, dists, temp_idxs, temp_dists, workspace, batch_test, k)
    return @async Threads.@threads for i in batch_test
        idx = partialsortperm!(workspace[i], view(temp_dists, :, i), 1:k, rev=false, initialized=true)
        copyto!(idxs[i], temp_idxs[idx, i])
        copyto!(dists[i], temp_dists[idx, i])
        nothing # JuliaLang/julia#40626
    end
end
