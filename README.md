# ParallelNeighbors.jl

[![Build Status](https://github.com/davnn/ParallelNeighbors.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/davnn/ParallelNeighbors.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Coverage](https://codecov.io/gh/davnn/ParallelNeighbors.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/davnn/ParallelNeighbors.jl)

`ParallelNeighbors.jl` is a Julia package to perform high-performance exact nearest neighbor searches in high-dimensionsal spaces. Other than [NearestNeighbors.jl](https://github.com/KristofferC/NearestNeighbors.jl), this package solely focuses on massively-parallel brute-force search, which becomes necessary once the data dimensionality becomes large.

Currently, the package is *experimental*, but it should already be usable for most cases. Things that are not yet supported are distance functions other than `Euclidean` and `SqEuclidean`.

While the package does currently not depend on [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) or [AMDGPU.jl](https://github.com/JuliaGPU/AMDGPU.jl), *it is intended to be used with a graphical processor unit (GPU)*, see the examples below.

## Installation

It is recommended to use [Pkg.jl](https://julialang.github.io/Pkg.jl) for installation. Follow the command below to install the latest official release or use `] add OutlierDetection` in the Julia REPL.

```julia
import Pkg;
Pkg.add("ParallelNeighbors")
```

If you would like to modify the package locally, you can use `Pkg.develop(OutlierDetection)` or `] dev OutlierDetection` in the Julia REPL. This fetches a full clone of the package to `~/.julia/dev/` (the path can be changed by setting the environment variable `JULIA_PKG_DEVDIR`).

## Usage

As the name implies, `ParallelNeighbors` is all about parallelization of your nearest neighbors searches. It provides three basic functions to perform nearest neighbors searches: `knn_full`, `knn_pointwise` and `knn_batch`. The interface is very similar to the interface provided by [NearestNeighbors.jl](https://github.com/KristofferC/NearestNeighbors.jl), with each function yielding two vectors containing the indices and distances of the nearest neighbors.

```julia
using ParallelNeighbors

k = 5; # number of neighbors to search
Xtrain = rand(Float32, 1000, 1000);
Xtest = rand(Float32, 1000, 100);

# using CPU-only in the following three examples
idxs_full, dists_full = knn_full(Xtrain, Xtest, k)
idxs_point, dists_point = knn_pointwise(Xtrain, Xtest, k)
idxs_batch, dists_batch = knn_batch(Xtrain, Xtest, k)
```

Assuming that you have [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) installed and a CUDA-compatible device available, you would use the package as follows.

```julia
using CUDA

# copy the full train and test data to the GPU using `convert`
idxs_full, dists_full = knn_full(Xtrain, Xtest, k; convert = cu)

# copy the full train data and test batches to the GPU
idxs_point, dists_point = knn_pointwise(Xtrain, Xtest, k; convert = cu)

# copy batches of train and test data to the GPU
idxs_batch, dists_batch = knn_batch(Xtrain, Xtest, k; convert = cu)
```

The difference between the functions is:

- `knn_full` calculates the full distance matrix, then ranks the nearest neighbors, which is currently the slowest method, but might become useful once better specialized GPU kernels are available.
- `knn_pointwise` calculates the distance matrix for a batch of test points to all train points, and sorts the batch *(n - 1)* in parallel while the distance for batch *n* is calculated.
- `knn_batch` is the most versatile function, batching both the train and test points, again sorting in parallel.

## Performance

`knn_batch` should be the method of choice for most use cases. You can tune the `batch_size` keyword argument such that it perfectly fits your use case. You should always try to fit all the data on the GPU as in the following example (reusing the data from above example). If you cannot fit all the data on the GPU beforehand you have to use the `convert` keyword argument and provide a conversion function that converts your CPU matrix to a GPU matrix of choice.

```julia
using BenchmarkTools: @benchmark

batch_size = 500
Xtrain_cu, Xtest_cu = cu(Xtrain), cu(Xtest)

@benchmark knn_batch(Xtrain_cu, Xtest_cu, k, batch_size)
```

If you are using `convert`, you should either use `knn_pointwise` if you can fit the entire training data on the GPU, or use a large `batch_size` for `knn_batch`.

## Contributing

`ParallelNeighbors.jl` is a community effort and your help is extremely welcome! Please open an issue or pull request if you find a bug or would like to contribute to the project.
