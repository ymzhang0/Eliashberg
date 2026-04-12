# src/Exceptions.jl
abstract type EliashbergException <: Exception end

# 1. Physics/Configuration related errors
struct PhysicalParameterError <: EliashbergException
    param_name::String
    value::Any
    reason::String
end

function Base.showerror(io::IO, e::PhysicalParameterError)
    print(io, "PhysicalParameterError: Invalid value `$(e.value)` for parameter `$(e.param_name)`. $(e.reason)")
end

# 2. Numerical/Algorithmic related errors
struct ConvergenceError <: EliashbergException
    solver_name::String
    iteration::Int
    last_error::Float64
    msg::String
end

function Base.showerror(io::IO, e::ConvergenceError)
    print(io, "ConvergenceError in $(e.solver_name) at iteration $(e.iteration): $(e.msg) (Last error: $(e.last_error))")
end

# 3. IO/Parsing related errors
struct ParsingError <: EliashbergException
    file::String
    msg::String
end

function Base.showerror(io::IO, e::ParsingError)
    print(io, "ParsingError in `$(e.file)`: $(e.msg)")
end

struct ConfigurationError <: EliashbergException
    param_name::String
    value::Any
    reason::String
end

function Base.showerror(io::IO, e::ConfigurationError)
    print(io, "ConfigurationError: Invalid configuration for `$(e.param_name)` (value: `$(e.value)`). $(e.reason)")
end