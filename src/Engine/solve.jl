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
    return _solve_assembled_eigensystem(solver, matrix; kwargs...)
end

function solve_assembled_eigensystem(matrix::SparseMatrixCSC; solver=SparseEigenSolverHook(), kwargs...)
    return _solve_assembled_eigensystem(solver, matrix; kwargs...)
end

function _solve_assembled_eigensystem(::DenseEigenSolver, matrix::AbstractMatrix; kwargs...)
    eig = eigen(Matrix(matrix))
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
