# Solvers/types.jl

# Approximation levels
abstract type ApproximationLevel end
struct ExactTrLn <: ApproximationLevel end
struct RPA <: ApproximationLevel end

