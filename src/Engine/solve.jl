using LinearAlgebra
using SparseArrays

"""
    AssemblySpectrum{V,M}

Normalized eigensystem container returned by the engine solve layer. It stores
eigenvalues and eigenvectors regardless of whether the underlying eigensolver
was the dense default path or a user-provided sparse hook.
"""
struct AssemblySpectrum{V,M}
    values::V
    vectors::M
end
function Base.show(io::IO, spectrum::AssemblySpectrum)
    v = spectrum.values
    if isempty(v)
        print(io, "AssemblySpectrum(0 eigenvalues)")
    else
        print(io, "AssemblySpectrum(", length(v), " eigenvalues, min=", minimum(v), ", max=", maximum(v), ")")
    end
end

Base.length(::AssemblySpectrum) = 2

function Base.iterate(spectrum::AssemblySpectrum, state::Int=1)
    state == 1 && return (spectrum.values, 2)
    state == 2 && return (spectrum.vectors, 3)
    return nothing
end

"""
    DenseEigenSolver()

Default eigensolver adapter for dense matrices. The solve layer materializes a
dense matrix and delegates to `LinearAlgebra.eigen`.
"""
struct DenseEigenSolver end

"""
    SparseEigenSolverHook(f)

Wrapper for a user-supplied sparse eigensolver callback. The callback receives
the assembled sparse matrix plus forwarded keyword arguments and may return
either an `Eigen` object, a `(values, vectors)` tuple, or a named tuple with
`values` and `vectors`.
"""
struct SparseEigenSolverHook{F}
    solve::F
end

function SparseEigenSolverHook()
    return SparseEigenSolverHook(_missing_sparse_eigensolver_hook)
end

"""
    solve_assembled_eigensystem(matrix; solver=nothing, kwargs...)

Solve an assembled eigensystem with a backend chosen from the matrix storage.
Dense matrices use `DenseEigenSolver()` by default, while sparse matrices
require an explicit `SparseEigenSolverHook` unless the caller opts into dense
fallback by passing `solver=DenseEigenSolver()`.
"""
function solve_assembled_eigensystem(matrix::AbstractMatrix; solver=DenseEigenSolver(), kwargs...)
    return _with_stage_log(
        "Solve assembled eigensystem";
        context=(matrix=matrix, solver=string(typeof(solver))),
        summarize_result=identity,
    ) do
        @timeit TO "Eigensystem Solve" return _solve_assembled_eigensystem(solver, matrix; kwargs...)
    end
end

function solve_assembled_eigensystem(matrix::SparseMatrixCSC; solver=SparseEigenSolverHook(), kwargs...)
    solver isa DenseEigenSolver && @warn "Dense eigensolver requested for sparse assembled matrix; materializing a dense copy." matrix=matrix
    return _with_stage_log(
        "Solve assembled eigensystem";
        context=(matrix=matrix, solver=string(typeof(solver))),
        summarize_result=identity,
    ) do
        @timeit TO "Eigensystem Solve" return _solve_assembled_eigensystem(solver, matrix; kwargs...)
    end
end

function _solve_assembled_eigensystem(::DenseEigenSolver, matrix::AbstractMatrix; kwargs...)
    !_isfinite_value(matrix) && @warn "Assembled matrix contains non-finite entries before dense eigensolve." matrix=matrix nonfinite_entries=_count_nonfinite(matrix)
    eig = eigen(Matrix(matrix))
    !_isfinite_value(eig.values) && @warn "Dense eigensolver returned non-finite eigenvalues." nonfinite_values=_count_nonfinite(eig.values) matrix=matrix
    return AssemblySpectrum(eig.values, eig.vectors)
end

function _solve_assembled_eigensystem(solver::SparseEigenSolverHook, matrix::SparseMatrixCSC; kwargs...)
    return _normalize_assembly_spectrum(solver.solve(matrix; kwargs...))
end

function _normalize_assembly_spectrum(result::AssemblySpectrum)
    return result
end

function _normalize_assembly_spectrum(result::LinearAlgebra.Eigen)
    return AssemblySpectrum(result.values, result.vectors)
end

function _normalize_assembly_spectrum(result::Tuple)
    length(result) == 2 || throw(ArgumentError("Sparse eigensolver hook must return a 2-tuple `(values, vectors)`."))
    return AssemblySpectrum(result[1], result[2])
end

function _normalize_assembly_spectrum(result::NamedTuple)
    haskey(result, :values) || throw(ArgumentError("Sparse eigensolver hook named tuple result must contain `values`."))
    haskey(result, :vectors) || throw(ArgumentError("Sparse eigensolver hook named tuple result must contain `vectors`."))
    return AssemblySpectrum(result.values, result.vectors)
end

function _missing_sparse_eigensolver_hook(::SparseMatrixCSC; kwargs...)
    throw(ArgumentError(
        "No sparse eigensolver hook configured. Pass `solver=SparseEigenSolverHook(f)` " *
        "or explicitly request dense fallback with `solver=DenseEigenSolver()`."
    ))
end

function resolve_assembled_eigensolver(::AbstractMatrix, eigensolver; sparse_warning::Union{Nothing,AbstractString}=nothing)
    return isnothing(eigensolver) ? DenseEigenSolver() : eigensolver
end

function resolve_assembled_eigensolver(::SparseMatrixCSC, eigensolver; sparse_warning::Union{Nothing,AbstractString}=nothing)
    isnothing(eigensolver) && !isnothing(sparse_warning) && @warn sparse_warning
    return isnothing(eigensolver) ? DenseEigenSolver() : eigensolver
end
