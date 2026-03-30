module Engine

using Distributed
using ..Eliashberg: AbstractKGrid

include("reduce.jl")
include("map.jl")
include("assembly.jl")
include("solve.jl")

export GridSample, BlockAxisLayout, UniformBlockLayout, VariableBlockLayout, AssemblySpectrum, DenseEigenSolver, SparseEigenSolverHook, bootstrap_engine_workers!, grid_samples, assemble_grid_vector, assemble_grid_matrix, assemble_sparse_grid_matrix, assemble_block_grid_matrix, assemble_sparse_block_grid_matrix, solve_assembled_eigensystem, integrate_grid, distributed_map_grid

end
