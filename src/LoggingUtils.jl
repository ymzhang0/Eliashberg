const _LOG_STAGE_GROUP = :eliashberg_stage
const _LOG_STAGE_DEPTH_KEY = :eliashberg_stage_depth

function _stage_log(level::LogLevel, message::AbstractString; kwargs...)
    logger = current_logger()
    Logging.min_enabled_level(logger) <= level || return nothing
    Logging.shouldlog(logger, level, @__MODULE__, _LOG_STAGE_GROUP, nothing) || return nothing
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
    depth = get(task_local_storage(), _LOG_STAGE_DEPTH_KEY, 0)
    task_local_storage(_LOG_STAGE_DEPTH_KEY, depth + 1)
    _stage_log(level, "$(stage) started"; stage, status=:start, context)

    try
        result = f()
        elapsed_s = (time_ns() - start_ns) / 1.0e9
        summary = isnothing(summarize_result) ? nothing : summarize_result(result)
        _stage_log(level, "$(stage) finished"; stage, status=:finish, elapsed_s, context, summary)
        return result
    catch err
        elapsed_s = (time_ns() - start_ns) / 1.0e9
        if depth == 0
            _stage_log(Logging.Error, "$(stage) failed"; stage, status=:error, elapsed_s, context, exception=(err, catch_backtrace()))
        else
            _stage_log(
                Logging.Error,
                "$(stage) failed";
                stage,
                status=:error,
                elapsed_s,
                context,
                error_type=string(typeof(err)),
                error_message=sprint(showerror, err),
            )
        end
        rethrow()
    finally
        task_local_storage(_LOG_STAGE_DEPTH_KEY, depth)
    end
end

_safe_weight_sum(weights) = isempty(weights) ? 0.0 : sum(weights)

isfinite_value(x::Real) = isfinite(x)
isfinite_value(x::Complex) = isfinite(real(x)) && isfinite(imag(x))
isfinite_value(values::AbstractArray) = all(isfinite_value, values)
count_nonfinite(values) = count(value -> !isfinite_value(value), values)

function grid_summary(grid::AbstractKGrid{D}) where {D}
    return (
        type=string(typeof(grid)),
        dim=D,
        n_points=length(grid),
        weight_sum=_safe_weight_sum(grid.weights),
    )
end

function cell_summary(cell_like)
    vectors = primitive_vectors(cell_like)
    summary = (
        dim=size(vectors, 1),
        lattice_norms=[norm(vectors[:, axis]) for axis in axes(vectors, 2)],
    )

    if applicable(periodicity, cell_like)
        summary = (; summary..., periodicity=periodicity(cell_like))
    end

    return summary
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

function layout_summary(layout::UniformBlockLayout)
    return (
        type="UniformBlockLayout",
        row_block_size=layout.row_block_size,
        col_block_size=layout.col_block_size,
    )
end

function layout_summary(layout::BlockAxisLayout)
    return (
        n_blocks=length(layout.block_sizes),
        total_size=layout.total_size,
        min_block=minimum(layout.block_sizes),
        max_block=maximum(layout.block_sizes),
    )
end

function layout_summary(layout::VariableBlockLayout)
    return (
        type="VariableBlockLayout",
        row_axis=layout_summary(layout.row_axis),
        col_axis=layout_summary(layout.col_axis),
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

function comparison_summary(comparison)
    if hasproperty(comparison, :shift) && hasproperty(comparison, :rmse) && hasproperty(comparison, :max_abs_error)
        return (
            shift=comparison.shift,
            rmse=comparison.rmse,
            max_abs_error=comparison.max_abs_error,
        )
    end

    return (type=string(typeof(comparison)),)
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
