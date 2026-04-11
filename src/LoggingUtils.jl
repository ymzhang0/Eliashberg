const _LOG_STAGE_GROUP = :eliashberg_stage

function _stage_log(level::LogLevel, message::AbstractString; kwargs...)
    logger = current_logger()
    Logging.handle_message(
        logger,
        level,
        message,
        @__MODULE__,
        _LOG_STAGE_GROUP,
        nothing,
        @__FILE__,
        0;
        kwargs...,
    )
    return nothing
end

function with_stage_log(f::F, stage::AbstractString; level::LogLevel=Logging.Info, context::NamedTuple=NamedTuple(), summarize_result=nothing) where {F}
    start_ns = time_ns()
    _stage_log(level, "$(stage) started"; stage, status=:start, context)

    try
        result = f()
        elapsed_s = (time_ns() - start_ns) / 1.0e9
        summary = isnothing(summarize_result) ? nothing : summarize_result(result)
        _stage_log(level, "$(stage) finished"; stage, status=:finish, elapsed_s, context, summary)
        return result
    catch err
        elapsed_s = (time_ns() - start_ns) / 1.0e9
        _stage_log(Logging.Error, "$(stage) failed"; stage, status=:error, elapsed_s, context, exception=(err, catch_backtrace()))
        rethrow()
    end
end

_safe_weight_sum(weights) = isempty(weights) ? 0.0 : sum(weights)

function grid_summary(grid::AbstractKGrid{D}) where {D}
    return (
        type=string(typeof(grid)),
        dim=D,
        n_points=length(grid),
        weight_sum=_safe_weight_sum(grid.weights),
    )
end

function samples_summary(samples::AbstractVector{<:GridSample})
    return (
        n_samples=length(samples),
        index_range=isempty(samples) ? nothing : (first(samples).index, last(samples).index),
        weight_sum=sum(sample.weight for sample in samples),
        value_type=isempty(samples) ? "unknown" : string(typeof(first(samples).value)),
    )
end

function axis_summary(axis)
    return (
        type=string(typeof(axis)),
        length=length(axis),
    )
end

function kpath_summary(kpath::KPath{D}) where {D}
    node_indices, node_labels = path_node_metadata(kpath)
    return (
        dim=D,
        n_points=length(path_points(kpath)),
        n_branches=length(path_branch_ranges(kpath)),
        n_nodes=length(node_indices),
        node_labels=collect(node_labels),
    )
end

function matrix_summary(matrix::SparseMatrixCSC)
    n_entries = prod(size(matrix))
    density = n_entries == 0 ? 0.0 : nnz(matrix) / n_entries
    return (
        storage=:sparse,
        size=size(matrix),
        eltype=string(eltype(matrix)),
        nnz=nnz(matrix),
        density=density,
    )
end

function matrix_summary(matrix::AbstractMatrix)
    return (
        storage=:dense,
        size=size(matrix),
        eltype=string(eltype(matrix)),
    )
end

function model_summary(model::TightBinding{D}) where {D}
    return (
        type="TightBinding",
        dim=D,
        n_hoppings=length(model.hoppings),
        EF=model.EF,
    )
end

function model_summary(model::MultiOrbitalTightBinding{D}) where {D}
    return (
        type="MultiOrbitalTightBinding",
        dim=D,
        periodicity=model.periodicity,
        n_orbitals=model.num_orbitals,
        n_hoppings=length(model.hoppings),
        EF=model.EF,
    )
end

function model_summary(model::SpinorDispersion{D}) where {D}
    return (
        type="SpinorDispersion",
        dim=D,
        bare=model_summary(model.bare),
    )
end

function model_summary(model::ElectronicDispersion{D}) where {D}
    return (
        type=string(typeof(model)),
        dim=D,
    )
end

function field_summary(field::AuxiliaryField)
    summary = (type=string(typeof(field)),)

    if hasproperty(field, :q)
        summary = (; summary..., q=getproperty(field, :q))
    end

    if hasproperty(field, :h)
        summary = (; summary..., h=getproperty(field, :h))
    end

    return summary
end

function field_summary(field::CompositeField)
    return (
        type=string(typeof(field)),
        n_fields=length(field),
        fields=[string(typeof(component)) for component in field.fields],
    )
end

function interaction_summary(interaction::CompositeInteraction)
    return (
        type=string(typeof(interaction)),
        n_terms=length(interaction),
        terms=[string(typeof(component)) for component in interaction.interactions],
    )
end

function interaction_summary(interaction::Interaction)
    return (
        type=string(typeof(interaction)),
    )
end

function approx_summary(approx::ApproximationLevel)
    return (
        type=string(typeof(approx)),
    )
end

function band_data_summary(data::BandStructureData)
    return (
        n_kpoints=size(data.bands, 1),
        n_bands=size(data.bands, 2),
    )
end

function spectrum_summary(spectrum)
    if spectrum isa AssemblySpectrum
        values = spectrum.values
        return (
            n_eigenvalues=length(values),
            min_value=isempty(values) ? nothing : minimum(values),
            max_value=isempty(values) ? nothing : maximum(values),
        )
    end

    return (type=string(typeof(spectrum)),)
end
