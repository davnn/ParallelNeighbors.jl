# ParallelNeighbors.jl

[![Build Status](https://github.com/davnn/ParallelNeighbors.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/davnn/ParallelNeighbors.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Coverage](https://codecov.io/gh/davnn/ParallelNeighbors.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/davnn/ParallelNeighbors.jl)

`ParallelNeighbors.jl` is a Julia package to perform high-performance exact nearest neighbor searches in high-dimensionsal spaces. Unlike [NearestNeighbors.jl](https://github.com/KristofferC/NearestNeighbors.jl), this package solely focuses on massively-parallel brute-force search, which becomes necessary once the data dimensionality becomes large.

Currently, the package is *experimental*, but it should already be usable for most cases. Things that are not yet supported are distance functions other than `Euclidean` and `SqEuclidean`.

While the package does currently not depend on [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) or [AMDGPU.jl](https://github.com/JuliaGPU/AMDGPU.jl), *it is intended to be used with a graphics processing unit (GPU)*, see the examples below.

## Installation

It is recommended to use [Pkg.jl](https://julialang.github.io/Pkg.jl) for installation. Follow the command below to install the latest official release or use `] add ParallelNeighbors` in the Julia REPL.

```julia
import Pkg;
Pkg.add("ParallelNeighbors")
```

If you would like to modify the package locally, you can use `Pkg.develop("ParallelNeighbors")` or `] dev ParallelNeighbors` in the Julia REPL. This fetches a full clone of the package to `~/.julia/dev/` (the path can be changed by setting the environment variable `JULIA_PKG_DEVDIR`).

## Usage

As the name implies, `ParallelNeighbors` is all about parallelization of your nearest neighbors searches. It provides a simple interface to perform massively-parallel nearest neighbors searches: `knn(Xtrain, Xtest, k, batch_size; metric, convert, algorithm)`. The interface is similar to the one provided by [NearestNeighbors.jl](https://github.com/KristofferC/NearestNeighbors.jl), yielding two vectors containing the indices and distances of the nearest neighbors.

```julia
using ParallelNeighbors

k = 5; # number of neighbors to search
Xtrain = rand(Float32, 1000, 1000);
Xtest = rand(Float32, 1000, 100);

# using CPU-only in the following examples
idxs, dists = knn(Xtrain, Xtest, k)
```

Assuming that you have [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) installed and a CUDA-compatible device available, you would use the package as follows (if `Xtrain` and `Xtest` is not already a `CuMatrix`). Note, that different algorithm are available depending on your requirements.

```julia
using CUDA

# copy batches of train and test data to the GPU
idxs, dists = knn(Xtrain, Xtest, k; convert = cu, algorithm=:hybrid_batch_all)

# copy the full train data and batches of the test data to the GPU
idxs, dists = knn(Xtrain, Xtest, k; convert = cu, algorithm=:hybrid_batch_test)

# copy the full train and test data to the GPU using `convert`
idxs, dists = knn(Xtrain, Xtest, k; convert = cu, algorithm=:full)
```

The difference between the algorithms is:

- `full` calculates the full distance matrix, then ranks the nearest neighbors, which is currently the slowest method, but might become useful once better specialized GPU kernels are available. In this case all calculations happen on the same device (CPU or GPU).
- `hybrid_batch_test` calculates the distance matrix for a batch of test points to all train points, and sorts the batch *(n - 1)* in parallel (CPU) while the distance for batch *n* is calculated (GPU).
- `hybrid_batch_all` is the most versatile function, batching distance calculation for the train and test points (GPU), again sorting in parallel (CPU).

## Performance

The default algorithm is `hybrid_batch_all` with a default batch size of `max(trunc(Int, n^(1 / sqrt(2))), k)` and should be the method of choice for most use cases. You can tune the `batch_size` argument such that it perfectly fits your use case. You should always try to fit all the data on the GPU beforehand as in the following example (reusing the data from above example). If you cannot fit all the data on the GPU beforehand you have to use the `convert` keyword argument and provide a conversion function that converts your CPU matrix to a GPU matrix of choice.

```julia
using BenchmarkTools: @benchmark

batch_size = 500
Xtrain_cu, Xtest_cu = cu(Xtrain), cu(Xtest)

@benchmark knn($Xtrain_cu, $Xtest_cu, $k, $batch_size)
```

If you are using `convert`, because you cannot fit all data on the GPU, you should either use `hybrid_batch_test` if you can fit the entire training data on the GPU, or use a large `batch_size` for `hybrid_batch_all`.

## Contributing

`ParallelNeighbors.jl` is a community effort and your help is extremely welcome! Please open an issue or pull request if you find a bug or would like to contribute to the project.
