
function renormalized_dispersion(
    disp::ElectronicDispersion{D},
    self_energy::SelfEnergy,
    k::SVector{D,Float64},
    omega::Float64
) where {D}
    bare_H = ε(k, disp)
    sigma_H = Σ(k, self_energy)
    return Hermitian(bare_H + sigma_H)
end

struct BCSKineticAssemblyTask{M}
    dispersion::M
end

function (task::BCSKineticAssemblyTask)(sample::GridSample)
    return real(band_structure(task.dispersion, sample.value).values[1])
end

struct BCSPairingAssemblyTask{I,M}
    interaction::I
    dispersion::M
end

function (task::BCSPairingAssemblyTask)(row_sample::GridSample, col_sample::GridSample)
    return V(row_sample.value, col_sample.value, task.interaction, task.dispersion) * col_sample.weight
end

"""
    solve_bcs(kgrid::AbstractKGrid{D}, dispersion_model::ElectronicDispersion, interaction_model::Interaction) where {D}

Solve the generalized linearized BCS gap equation over a `KGrid`. 
Assumes a single dominant band closest to the Fermi level for the constructed N x N eigenvalue problem.
"""
function solve_bcs(
    kgrid::AbstractKGrid{D},
    dispersion_model::ElectronicDispersion{D},
    interaction_model::Interaction,
    ;
    matrix_format::Symbol=:dense,
    sparse_atol::Real=0.0,
    eigensolver=nothing,
    bootstrap_workers::Bool=false,
    n_workers::Integer=max(0, Threads.nthreads() - 1),
    project::Union{Nothing,AbstractString}=Base.active_project(),
    restrict::Bool=true
) where {D}
    samples = Engine.grid_samples(kgrid)
    kinetic_vector = Engine.assemble_grid_vector(
        BCSKineticAssemblyTask(dispersion_model),
        samples;
        bootstrap_workers=bootstrap_workers,
        n_workers=n_workers,
        project=project,
        restrict=restrict
    )
    pairing_matrix = _assemble_bcs_pairing_matrix(
        matrix_format,
        samples,
        interaction_model,
        dispersion_model;
        sparse_atol=sparse_atol,
        bootstrap_workers=bootstrap_workers,
        n_workers=n_workers,
        project=project,
        restrict=restrict
    )

    H = _materialize_bcs_matrix(pairing_matrix)
    H[diagind(H)] .+= kinetic_vector

    spectrum = Engine.solve_assembled_eigensystem(H; solver=_resolve_bcs_eigensolver(H, eigensolver))
    return spectrum.values, spectrum.vectors
end

function _assemble_bcs_pairing_matrix(
    ::Val{:dense},
    samples,
    interaction_model::Interaction,
    dispersion_model::ElectronicDispersion;
    bootstrap_workers::Bool=false,
    n_workers::Integer=max(0, Threads.nthreads() - 1),
    project::Union{Nothing,AbstractString}=Base.active_project(),
    restrict::Bool=true
)
    return Engine.assemble_grid_matrix(
        BCSPairingAssemblyTask(interaction_model, dispersion_model),
        samples,
        samples;
        bootstrap_workers=bootstrap_workers,
        n_workers=n_workers,
        project=project,
        restrict=restrict
    )
end

function _assemble_bcs_pairing_matrix(
    ::Val{:sparse},
    samples,
    interaction_model::Interaction,
    dispersion_model::ElectronicDispersion;
    sparse_atol::Real=0.0,
    bootstrap_workers::Bool=false,
    n_workers::Integer=max(0, Threads.nthreads() - 1),
    project::Union{Nothing,AbstractString}=Base.active_project(),
    restrict::Bool=true
)
    return Engine.assemble_sparse_grid_matrix(
        BCSPairingAssemblyTask(interaction_model, dispersion_model),
        samples,
        samples;
        atol=sparse_atol,
        bootstrap_workers=bootstrap_workers,
        n_workers=n_workers,
        project=project,
        restrict=restrict
    )
end

function _assemble_bcs_pairing_matrix(
    matrix_format::Symbol,
    samples,
    interaction_model::Interaction,
    dispersion_model::ElectronicDispersion;
    sparse_atol::Real=0.0,
    bootstrap_workers::Bool=false,
    n_workers::Integer=max(0, Threads.nthreads() - 1),
    project::Union{Nothing,AbstractString}=Base.active_project(),
    restrict::Bool=true
)
    if matrix_format == :dense
        return _assemble_bcs_pairing_matrix(
            Val(:dense),
            samples,
            interaction_model,
            dispersion_model;
            bootstrap_workers=bootstrap_workers,
            n_workers=n_workers,
            project=project,
            restrict=restrict
        )
    elseif matrix_format == :sparse
        return _assemble_bcs_pairing_matrix(
            Val(:sparse),
            samples,
            interaction_model,
            dispersion_model;
            sparse_atol=sparse_atol,
            bootstrap_workers=bootstrap_workers,
            n_workers=n_workers,
            project=project,
            restrict=restrict
        )
    end

    throw(ArgumentError("Unsupported matrix_format `$matrix_format`. Expected `:dense` or `:sparse`."))
end

_materialize_bcs_matrix(matrix::AbstractMatrix) = matrix
_materialize_bcs_matrix(matrix::SparseMatrixCSC) = copy(matrix)

function _resolve_bcs_eigensolver(::SparseMatrixCSC, eigensolver)
    return isnothing(eigensolver) ? Engine.DenseEigenSolver() : eigensolver
end

function _resolve_bcs_eigensolver(::AbstractMatrix, eigensolver)
    return isnothing(eigensolver) ? Engine.DenseEigenSolver() : eigensolver
end
