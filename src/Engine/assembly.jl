using SparseArrays

"""
    GridSample{V,W}

Generic weighted sample descriptor used by the assembly engine. A sample keeps
the payload value, its quadrature weight, and a stable index so assembly
kernels can express diagonal or neighborhood structure without depending on any
physics-specific coordinate names.
"""
struct GridSample{V,W}
    value::V
    weight::W
    index::Int
end

"""
    UniformBlockLayout

Uniform block layout descriptor for dense or sparse block-matrix assembly.
Every sample on the row axis contributes `row_block_size` rows and every sample
on the column axis contributes `col_block_size` columns.
"""
struct UniformBlockLayout
    row_block_size::Int
    col_block_size::Int

    function UniformBlockLayout(row_block_size::Integer, col_block_size::Integer)
        row_block_size > 0 || throw(ArgumentError("row_block_size must be positive."))
        col_block_size > 0 || throw(ArgumentError("col_block_size must be positive."))
        return new(Int(row_block_size), Int(col_block_size))
    end
end

"""
    BlockAxisLayout

Axis-local block layout descriptor for non-uniform block assembly. Each entry in
`block_sizes` defines the block extent contributed by the corresponding sample
on that axis.
"""
struct BlockAxisLayout
    block_sizes::Vector{Int}
    offsets::Vector{Int}
    total_size::Int

    function BlockAxisLayout(block_sizes::AbstractVector{<:Integer})
        isempty(block_sizes) && throw(ArgumentError("block_sizes must be non-empty."))
        normalized_sizes = Int[]
        offsets = Int[]
        next_offset = 1

        for block_size in block_sizes
            block_size > 0 || throw(ArgumentError("block sizes must be positive."))
            push!(normalized_sizes, Int(block_size))
            push!(offsets, next_offset)
            next_offset += Int(block_size)
        end

        return new(normalized_sizes, offsets, next_offset - 1)
    end
end

"""
    VariableBlockLayout

Non-uniform block layout descriptor for dense or sparse block-matrix assembly.
Row and column axes may each carry their own per-sample block sizes.
"""
struct VariableBlockLayout
    row_axis::BlockAxisLayout
    col_axis::BlockAxisLayout
end

VariableBlockLayout(row_sizes::AbstractVector{<:Integer}, col_sizes::AbstractVector{<:Integer}) =
    VariableBlockLayout(BlockAxisLayout(row_sizes), BlockAxisLayout(col_sizes))

"""
    grid_samples(grid::AbstractKGrid)

Materialize a weighted sample axis from a grid so higher-level assembly kernels
can operate on generic sample descriptors instead of raw coordinate arrays.
"""
function grid_samples(grid::AbstractKGrid)
    return [GridSample(grid.points[idx], grid.weights[idx], idx) for idx in eachindex(grid.points)]
end

"""
    assemble_grid_vector(f::F, axis; kwargs...) where {F}

Assemble a dense vector by mapping a local kernel over a single discretized
sample axis.
"""
function assemble_grid_vector(f::F, axis; kwargs...) where {F}
    return distributed_map_grid(f, axis; kwargs...)
end

"""
    assemble_grid_matrix(f::F, row_axis, col_axis; kwargs...) where {F}

Assemble a dense matrix by mapping a local kernel over the Cartesian product of
two discretized sample axes.
"""
function assemble_grid_matrix(f::F, row_axis, col_axis; kwargs...) where {F}
    return distributed_map_grid(f, row_axis, col_axis; kwargs...)
end

"""
    assemble_sparse_grid_matrix(f::F, row_axis, col_axis; atol=0.0, kwargs...) where {F}

Assemble a sparse scalar matrix by mapping a local kernel over the Cartesian
product of two sample axes and materializing only entries whose magnitude is
strictly larger than `atol`.
"""
function assemble_sparse_grid_matrix(
    f::F,
    row_axis,
    col_axis;
    atol::Real=0.0,
    kwargs...
) where {F}
    row_values = collect(row_axis)
    col_values = collect(col_axis)
    entries = _map_parameter_task(
        SparseScalarAssemblyTask(f, row_values, col_values, atol),
        (length(row_values), length(col_values));
        kwargs...
    )
    return _sparse_matrix_from_entries(entries, length(row_values), length(col_values))
end

"""
    assemble_block_grid_matrix(f::F, row_axis, col_axis, layout::UniformBlockLayout; kwargs...) where {F}

Assemble a dense block matrix by mapping a block-valued local kernel over the
Cartesian product of two sample axes and packing the blocks into a single dense
matrix according to `layout`.
"""
function assemble_block_grid_matrix(
    f::F,
    row_axis,
    col_axis,
    layout::UniformBlockLayout;
    kwargs...
) where {F}
    blocks = distributed_map_grid(f, row_axis, col_axis; kwargs...)
    return _dense_block_matrix(blocks, layout)
end

function assemble_block_grid_matrix(
    f::F,
    row_axis,
    col_axis,
    layout::VariableBlockLayout;
    kwargs...
) where {F}
    blocks = distributed_map_grid(f, row_axis, col_axis; kwargs...)
    return _dense_block_matrix(blocks, layout)
end

"""
    assemble_sparse_block_grid_matrix(f::F, row_axis, col_axis, layout::UniformBlockLayout; atol=0.0, kwargs...) where {F}

Assemble a sparse block matrix by mapping a block-valued local kernel over the
Cartesian product of two sample axes and materializing only block entries whose
magnitude exceeds `atol`.
"""
function assemble_sparse_block_grid_matrix(
    f::F,
    row_axis,
    col_axis,
    layout::UniformBlockLayout;
    atol::Real=0.0,
    kwargs...
) where {F}
    row_values = collect(row_axis)
    col_values = collect(col_axis)
    entries = _map_parameter_task(
        SparseBlockAssemblyTask(f, row_values, col_values, layout, atol),
        (length(row_values), length(col_values));
        kwargs...
    )
    return _sparse_block_matrix_from_entries(entries, length(row_values), length(col_values), layout, atol)
end

function assemble_sparse_block_grid_matrix(
    f::F,
    row_axis,
    col_axis,
    layout::VariableBlockLayout;
    atol::Real=0.0,
    kwargs...
) where {F}
    row_values = collect(row_axis)
    col_values = collect(col_axis)
    entries = _map_parameter_task(
        SparseBlockAssemblyTask(f, row_values, col_values, layout, atol),
        (length(row_values), length(col_values));
        kwargs...
    )
    return _sparse_block_matrix_from_entries(entries, length(row_values), length(col_values), layout, atol)
end

struct SparseScalarAssemblyEntry{T}
    row::Int
    col::Int
    value::T
end

struct SparseBlockAssemblyEntry{B}
    block_row::Int
    block_col::Int
    block::B
end

struct SparseScalarAssemblyTask{F,R,C,T}
    f::F
    row_axis::R
    col_axis::C
    atol::T
end

function (task::SparseScalarAssemblyTask)(index::CartesianIndex{2})
    row_idx = index[1]
    col_idx = index[2]
    value = task.f(task.row_axis[row_idx], task.col_axis[col_idx])
    _is_significant(value, task.atol) || return nothing
    return SparseScalarAssemblyEntry(row_idx, col_idx, value)
end

struct SparseBlockAssemblyTask{F,R,C,L,T}
    f::F
    row_axis::R
    col_axis::C
    layout::L
    atol::T
end

function (task::SparseBlockAssemblyTask)(index::CartesianIndex{2})
    row_idx = index[1]
    col_idx = index[2]
    block = _coerce_block(task.f(task.row_axis[row_idx], task.col_axis[col_idx]), task.layout, row_idx, col_idx)
    _block_has_significant_entry(block, task.atol) || return nothing
    return SparseBlockAssemblyEntry(row_idx, col_idx, block)
end

function _dense_block_matrix(blocks::AbstractMatrix, layout::UniformBlockLayout)
    n_block_rows, n_block_cols = size(blocks)
    result_type = _dense_block_eltype(blocks, layout)
    n_rows = n_block_rows * layout.row_block_size
    n_cols = n_block_cols * layout.col_block_size
    matrix = Matrix{result_type}(undef, n_rows, n_cols)

    for block_row in 1:n_block_rows
        row_range = _block_range(block_row, layout.row_block_size)
        for block_col in 1:n_block_cols
            col_range = _block_range(block_col, layout.col_block_size)
            matrix[row_range, col_range] = _coerce_block(blocks[block_row, block_col], layout)
        end
    end

    return matrix
end

function _dense_block_matrix(blocks::AbstractMatrix, layout::VariableBlockLayout)
    _validate_layout_axes(size(blocks), layout)
    n_block_rows, n_block_cols = size(blocks)
    result_type = _dense_block_eltype(blocks, layout)
    matrix = Matrix{result_type}(undef, layout.row_axis.total_size, layout.col_axis.total_size)

    for block_row in 1:n_block_rows
        row_range = _block_range(layout.row_axis, block_row)
        for block_col in 1:n_block_cols
            col_range = _block_range(layout.col_axis, block_col)
            matrix[row_range, col_range] = _coerce_block(blocks[block_row, block_col], layout, block_row, block_col)
        end
    end

    return matrix
end

function _dense_block_eltype(blocks::AbstractMatrix, layout::UniformBlockLayout)
    result_type = Union{}

    for block in blocks
        result_type = promote_type(result_type, eltype(_coerce_block(block, layout)))
    end

    return result_type === Union{} ? Float64 : result_type
end

function _dense_block_eltype(blocks::AbstractMatrix, layout::VariableBlockLayout)
    _validate_layout_axes(size(blocks), layout)
    result_type = Union{}

    for block_row in axes(blocks, 1)
        for block_col in axes(blocks, 2)
            result_type = promote_type(result_type, eltype(_coerce_block(blocks[block_row, block_col], layout, block_row, block_col)))
        end
    end

    return result_type === Union{} ? Float64 : result_type
end

function _sparse_matrix_from_entries(entries, n_rows::Int, n_cols::Int)
    kept_entries = filter(!isnothing, entries)
    isempty(kept_entries) && return spzeros(Float64, n_rows, n_cols)

    value_type = Union{}
    for entry in kept_entries
        value_type = promote_type(value_type, typeof(entry.value))
    end

    row_indices = Int[]
    col_indices = Int[]
    values = Vector{value_type}(undef, 0)

    for entry in kept_entries
        push!(row_indices, entry.row)
        push!(col_indices, entry.col)
        push!(values, entry.value)
    end

    return sparse(row_indices, col_indices, values, n_rows, n_cols)
end

function _sparse_block_matrix_from_entries(
    entries,
    n_block_rows::Int,
    n_block_cols::Int,
    layout::UniformBlockLayout,
    atol::Real
)
    kept_entries = filter(!isnothing, entries)
    n_rows = n_block_rows * layout.row_block_size
    n_cols = n_block_cols * layout.col_block_size
    isempty(kept_entries) && return spzeros(Float64, n_rows, n_cols)

    value_type = Union{}
    for entry in kept_entries
        value_type = promote_type(value_type, eltype(_coerce_block(entry.block, layout)))
    end

    row_indices = Int[]
    col_indices = Int[]
    values = Vector{value_type}(undef, 0)

    for entry in kept_entries
        row_offset = (entry.block_row - 1) * layout.row_block_size
        col_offset = (entry.block_col - 1) * layout.col_block_size
        block = _coerce_block(entry.block, layout)

        for local_row in 1:layout.row_block_size
            for local_col in 1:layout.col_block_size
                value = block[local_row, local_col]
                _is_significant(value, atol) || continue
                push!(row_indices, row_offset + local_row)
                push!(col_indices, col_offset + local_col)
                push!(values, value)
            end
        end
    end

    return sparse(row_indices, col_indices, values, n_rows, n_cols)
end

function _sparse_block_matrix_from_entries(
    entries,
    n_block_rows::Int,
    n_block_cols::Int,
    layout::VariableBlockLayout,
    atol::Real
)
    _validate_layout_axes((n_block_rows, n_block_cols), layout)
    kept_entries = filter(!isnothing, entries)
    isempty(kept_entries) && return spzeros(Float64, layout.row_axis.total_size, layout.col_axis.total_size)

    value_type = Union{}
    for entry in kept_entries
        value_type = promote_type(value_type, eltype(_coerce_block(entry.block, layout, entry.block_row, entry.block_col)))
    end

    row_indices = Int[]
    col_indices = Int[]
    values = Vector{value_type}(undef, 0)

    for entry in kept_entries
        row_range = _block_range(layout.row_axis, entry.block_row)
        col_range = _block_range(layout.col_axis, entry.block_col)
        block = _coerce_block(entry.block, layout, entry.block_row, entry.block_col)

        for (local_row, global_row) in enumerate(row_range)
            for (local_col, global_col) in enumerate(col_range)
                value = block[local_row, local_col]
                _is_significant(value, atol) || continue
                push!(row_indices, global_row)
                push!(col_indices, global_col)
                push!(values, value)
            end
        end
    end

    return sparse(row_indices, col_indices, values, layout.row_axis.total_size, layout.col_axis.total_size)
end

function _coerce_block(block::AbstractMatrix, layout::UniformBlockLayout)
    size(block) == (layout.row_block_size, layout.col_block_size) ||
        throw(DimensionMismatch("Block size $(size(block)) does not match layout ($(layout.row_block_size), $(layout.col_block_size))."))
    return block
end

_coerce_block(block::AbstractMatrix, layout::UniformBlockLayout, ::Int, ::Int) = _coerce_block(block, layout)

function _coerce_block(value::Number, layout::UniformBlockLayout)
    layout.row_block_size == 1 && layout.col_block_size == 1 ||
        throw(DimensionMismatch("Scalar block values require a 1x1 UniformBlockLayout."))
    return reshape([value], 1, 1)
end

_coerce_block(value::Number, layout::UniformBlockLayout, ::Int, ::Int) = _coerce_block(value, layout)

function _coerce_block(block::AbstractMatrix, layout::VariableBlockLayout, block_row::Int, block_col::Int)
    expected_shape = _block_shape(layout, block_row, block_col)
    size(block) == expected_shape ||
        throw(DimensionMismatch("Block size $(size(block)) does not match layout $expected_shape at block ($block_row, $block_col)."))
    return block
end

function _coerce_block(value::Number, layout::VariableBlockLayout, block_row::Int, block_col::Int)
    expected_shape = _block_shape(layout, block_row, block_col)
    expected_shape == (1, 1) ||
        throw(DimensionMismatch("Scalar block values require a 1x1 block at ($block_row, $block_col), got $expected_shape."))
    return reshape([value], 1, 1)
end

_is_significant(value, atol::Real) = abs(value) > atol

function _block_has_significant_entry(block::AbstractMatrix, atol::Real)
    for value in block
        _is_significant(value, atol) && return true
    end

    return false
end

function _block_range(block_index::Int, block_size::Int)
    start_idx = (block_index - 1) * block_size + 1
    return start_idx:(start_idx + block_size - 1)
end

function _block_range(axis_layout::BlockAxisLayout, block_index::Int)
    start_idx = axis_layout.offsets[block_index]
    block_size = axis_layout.block_sizes[block_index]
    return start_idx:(start_idx + block_size - 1)
end

function _block_shape(layout::VariableBlockLayout, block_row::Int, block_col::Int)
    return (layout.row_axis.block_sizes[block_row], layout.col_axis.block_sizes[block_col])
end

function _validate_layout_axes(dims::Tuple{Int,Int}, layout::VariableBlockLayout)
    dims[1] == length(layout.row_axis.block_sizes) ||
        throw(DimensionMismatch("Row block count $(dims[1]) does not match variable row layout length $(length(layout.row_axis.block_sizes))."))
    dims[2] == length(layout.col_axis.block_sizes) ||
        throw(DimensionMismatch("Column block count $(dims[2]) does not match variable column layout length $(length(layout.col_axis.block_sizes))."))
end
