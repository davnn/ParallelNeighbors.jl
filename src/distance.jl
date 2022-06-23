# Calculate a squared euclidean distance matrix using the identity
# (x - y)^t (x - y) = x^t x - 2 y^t x + y^t y
sqeuclidean_kernel(X::M, Y::M) where {M<:device_input} =
    sum(abs2, X, dims=1) .- 2 * Y' * X .+ sum(abs2, Y, dims=1)'

# Calculate the euclidean distance matrix using the absolute squared
# euclidean distance matrix to discard negative rounding errors
euclidean_kernel(X::M, Y::M) where {M<:device_input} =
    sqrt.(abs.(sqeuclidean_kernel(X, Y)))

function metric_to_kernel(metric::PreMetric)
    assert_thres(metric) = @assert metric.thresh == 0.0 "Currently, no threshold other than 0.0 is supported."
    if typeof(metric) <: Euclidean
        assert_thres(metric)
        return euclidean_kernel
    elseif typeof(metric) <: SqEuclidean
        assert_thres(metric)
        return sqeuclidean_kernel
    end
end
