struct SampledHamiltonianAssembly{M,L,S}
    matrix::M
    layout::L
    samples::S
end

struct LocalHamiltonianBlockTask{M}
    dispersion::M
end

function (task::LocalHamiltonianBlockTask)(sample::GridSample)
    return Matrix(ε(sample.value, task.dispersion))
end

"""
    assemble_sampled_hamiltonian(kgrid, dispersion; matrix_format=:dense, sparse_atol=0.0, ...)

Assemble the direct-sum Hamiltonian over a sampled parameter grid. Each sample
contributes one local matrix block `ε(k, dispersion)` and the global operator is
packed as a block-diagonal matrix using `VariableBlockLayout`.

This wrapper is physical-model agnostic at the block level, so it naturally
covers multi-orbital Bloch Hamiltonians and BdG mean-field Hamiltonians.
"""
function assemble_sampled_hamiltonian(
    kgrid::AbstractKGrid{D},
    dispersion::ElectronicDispersion{D};
    matrix_format::Symbol=:dense,
    sparse_atol::Real=0.0,
    bootstrap_workers::Bool=false,
    n_workers::Integer=max(0, Threads.nthreads() - 1),
    project::Union{Nothing,AbstractString}=Base.active_project(),
    restrict::Bool=true
) where {D}
    samples = Engine.grid_samples(kgrid)
    layout = _sampled_hamiltonian_layout(samples, dispersion)
    block_task = LocalHamiltonianBlockTask(dispersion)

    matrix = _assemble_sampled_hamiltonian_matrix(
        matrix_format,
        block_task,
        samples,
        layout;
        sparse_atol=sparse_atol,
        bootstrap_workers=bootstrap_workers,
        n_workers=n_workers,
        project=project,
        restrict=restrict
    )

    return SampledHamiltonianAssembly(matrix, layout, samples)
end

"""
    solve_sampled_hamiltonian(kgrid, dispersion; matrix_format=:dense, sparse_atol=0.0, eigensolver=nothing, ...)

Assemble and diagonalize the sampled direct-sum Hamiltonian. Dense assembly uses
the default dense eigensolver path, while sparse assembly accepts a user-defined
`SparseEigenSolverHook` through `eigensolver`.
"""
function solve_sampled_hamiltonian(
    kgrid::AbstractKGrid{D},
    dispersion::ElectronicDispersion{D};
    matrix_format::Symbol=:dense,
    sparse_atol::Real=0.0,
    eigensolver=nothing,
    bootstrap_workers::Bool=false,
    n_workers::Integer=max(0, Threads.nthreads() - 1),
    project::Union{Nothing,AbstractString}=Base.active_project(),
    restrict::Bool=true
) where {D}
    assembly = assemble_sampled_hamiltonian(
        kgrid,
        dispersion;
        matrix_format=matrix_format,
        sparse_atol=sparse_atol,
        bootstrap_workers=bootstrap_workers,
        n_workers=n_workers,
        project=project,
        restrict=restrict
    )

    spectrum = Engine.solve_assembled_eigensystem(
        assembly.matrix;
        solver=_resolve_sampled_hamiltonian_eigensolver(assembly.matrix, eigensolver)
    )
    return spectrum
end

function _sampled_hamiltonian_layout(samples, dispersion::ElectronicDispersion)
    block_sizes = [size(ε(sample.value, dispersion), 1) for sample in samples]
    return VariableBlockLayout(block_sizes, block_sizes)
end

function _assemble_sampled_hamiltonian_matrix(
    ::Val{:dense},
    block_task,
    samples,
    layout::VariableBlockLayout;
    bootstrap_workers::Bool=false,
    n_workers::Integer=max(0, Threads.nthreads() - 1),
    project::Union{Nothing,AbstractString}=Base.active_project(),
    restrict::Bool=true,
    kwargs...
)
    return Engine.assemble_block_diagonal_matrix(
        block_task,
        samples,
        layout;
        bootstrap_workers=bootstrap_workers,
        n_workers=n_workers,
        project=project,
        restrict=restrict
    )
end

function _assemble_sampled_hamiltonian_matrix(
    ::Val{:sparse},
    block_task,
    samples,
    layout::VariableBlockLayout;
    sparse_atol::Real=0.0,
    bootstrap_workers::Bool=false,
    n_workers::Integer=max(0, Threads.nthreads() - 1),
    project::Union{Nothing,AbstractString}=Base.active_project(),
    restrict::Bool=true,
    kwargs...
)
    return Engine.assemble_sparse_block_diagonal_matrix(
        block_task,
        samples,
        layout;
        atol=sparse_atol,
        bootstrap_workers=bootstrap_workers,
        n_workers=n_workers,
        project=project,
        restrict=restrict
    )
end

function _assemble_sampled_hamiltonian_matrix(
    matrix_format::Symbol,
    block_task,
    samples,
    layout::VariableBlockLayout;
    sparse_atol::Real=0.0,
    bootstrap_workers::Bool=false,
    n_workers::Integer=max(0, Threads.nthreads() - 1),
    project::Union{Nothing,AbstractString}=Base.active_project(),
    restrict::Bool=true
)
    if matrix_format == :dense
        return _assemble_sampled_hamiltonian_matrix(
            Val(:dense),
            block_task,
            samples,
            layout;
            bootstrap_workers=bootstrap_workers,
            n_workers=n_workers,
            project=project,
            restrict=restrict
        )
    elseif matrix_format == :sparse
        return _assemble_sampled_hamiltonian_matrix(
            Val(:sparse),
            block_task,
            samples,
            layout;
            sparse_atol=sparse_atol,
            bootstrap_workers=bootstrap_workers,
            n_workers=n_workers,
            project=project,
            restrict=restrict
        )
    end

    throw(ArgumentError("Unsupported matrix_format `$matrix_format`. Expected `:dense` or `:sparse`."))
end

function _resolve_sampled_hamiltonian_eigensolver(::SparseMatrixCSC, eigensolver)
    return isnothing(eigensolver) ? Engine.DenseEigenSolver() : eigensolver
end

function _resolve_sampled_hamiltonian_eigensolver(::AbstractMatrix, eigensolver)
    return isnothing(eigensolver) ? Engine.DenseEigenSolver() : eigensolver
end
