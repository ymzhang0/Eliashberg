# Solvers/types.jl

# Approximation levels
abstract type ApproximationLevel end
Base.show(io::IO, a::ApproximationLevel) = print(io, string(typeof(a)))

