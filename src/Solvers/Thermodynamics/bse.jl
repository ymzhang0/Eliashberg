
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
    return with_stage_log(
        "Solve BCS";
        context=(
            grid=kgrid,
            model=dispersion_model,
            interaction=interaction_model,
            matrix_format=matrix_format,
            sparse_atol=Float64(sparse_atol),
            bootstrap_workers=bootstrap_workers,
            requested_workers=Int(n_workers),
        ),
        summarize_result=result -> (
            n_eigenvalues=length(result[1]),
            min_value=isempty(result[1]) ? nothing : minimum(result[1]),
            max_value=isempty(result[1]) ? nothing : maximum(result[1]),
        ),
    ) do
        @timeit TO "BCS Solver" begin
            samples = Engine.grid_samples(kgrid)
            kinetic_vector = Engine.assemble_grid_vector(
                BCSKineticAssemblyTask(dispersion_model),
                samples;
                bootstrap_workers=bootstrap_workers,
                n_workers=n_workers,
                project=project,
                restrict=restrict
            )
            if !isfinite_value(kinetic_vector)
                @error "BCS kinetic vector contains non-finite entries." nonfinite_entries = count_nonfinite(kinetic_vector) grid = kgrid model = dispersion_model
                throw(PhysicalParameterError("kinetic_vector", "NaN/Inf", "Evaluated BCS kinetic vector contains non-finite entries. Check if dispersion model or grid produces singularities."))
            end
            pairing_matrix = @timeit TO "Pairing Matrix Assembly" _assemble_bcs_pairing_matrix(
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

            H = Engine.prepare_matrix_for_diagonal_update(pairing_matrix)
            H[diagind(H)] .+= kinetic_vector
            if !isfinite_value(H)
                @error "BCS matrix contains non-finite entries before eigensolve." nonfinite_entries = count_nonfinite(H) matrix = H
                throw(PhysicalParameterError("BCS Matrix", "NaN/Inf", "Assembled Hamiltonian contains non-finite entries. Check model parameters or sparse_atol."))
            end

            spectrum = Engine.solve_assembled_eigensystem(
                H;
                solver=Engine.resolve_assembled_eigensolver(
                    H,
                    eigensolver;
                    sparse_warning="Sparse BCS matrix is using dense eigensolver fallback. Provide `eigensolver=Engine.SparseEigenSolverHook(...)` for large problems.",
                ),
            )
            !isfinite_value(spectrum.values) && @warn "BCS eigenspectrum contains non-finite eigenvalues." nonfinite_values = count_nonfinite(spectrum.values) matrix_format = matrix_format grid = kgrid
            return spectrum.values, spectrum.vectors
        end
    end
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
    task = BCSPairingAssemblyTask(interaction_model, dispersion_model)
    dense_builder = () -> Engine.assemble_grid_matrix(
        task,
        samples,
        samples;
        bootstrap_workers=bootstrap_workers,
        n_workers=n_workers,
        project=project,
        restrict=restrict
    )
    sparse_builder = () -> Engine.assemble_sparse_grid_matrix(
        task,
        samples,
        samples;
        atol=sparse_atol,
        bootstrap_workers=bootstrap_workers,
        n_workers=n_workers,
        project=project,
        restrict=restrict
    )
    return Engine.assemble_by_storage(matrix_format, dense_builder, sparse_builder)
end
