module Engine

using Distributed
using Logging
using TimerOutputs
using ..Eliashberg: AbstractKGrid, TO

_eliashberg_parent() = parentmodule(@__MODULE__)
_with_stage_log(args...; kwargs...) = getproperty(_eliashberg_parent(), :with_stage_log)(args...; kwargs...)
_grid_summary(args...) = getproperty(_eliashberg_parent(), :grid_summary)(args...)
_axis_summary(args...) = getproperty(_eliashberg_parent(), :axis_summary)(args...)
_layout_summary(args...) = getproperty(_eliashberg_parent(), :layout_summary)(args...)
_matrix_summary(args...) = getproperty(_eliashberg_parent(), :matrix_summary)(args...)
_spectrum_summary(args...) = getproperty(_eliashberg_parent(), :spectrum_summary)(args...)
_isfinite_value(args...) = getproperty(_eliashberg_parent(), :isfinite_value)(args...)
_count_nonfinite(args...) = getproperty(_eliashberg_parent(), :count_nonfinite)(args...)

include("reduce.jl")
include("map.jl")
include("assembly.jl")
include("solve.jl")

export GridSample, BlockAxisLayout, UniformBlockLayout, VariableBlockLayout, AssemblySpectrum, DenseEigenSolver, SparseEigenSolverHook, bootstrap_engine_workers!, grid_samples, assemble_grid_vector, assemble_grid_matrix, assemble_sparse_grid_matrix, assemble_block_grid_matrix, assemble_sparse_block_grid_matrix, assemble_block_diagonal_matrix, assemble_sparse_block_diagonal_matrix, solve_assembled_eigensystem, integrate_grid, distributed_map_grid

end
