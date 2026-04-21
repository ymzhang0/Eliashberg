# src/Visualization/plot_helpers.jl

# --- Shared Path Plotting Helpers (Makie Dependent) ---

function _branch_ticks(
    distances::AbstractVector{<:Real},
    node_indices::AbstractVector{<:Integer},
    node_labels::AbstractVector{<:AbstractString},
    range::UnitRange{Int},
)
    ticks = Float64[]
    labels = String[]

    for (idx, label) in zip(node_indices, node_labels)
        if first(range) <= idx <= last(range)
            push!(ticks, distances[idx])
            push!(labels, label)
        end
    end

    return ticks, labels
end

function _branch_width(distances::AbstractVector{<:Real}, range::UnitRange{Int})
    return max(distances[last(range)] - distances[first(range)], eps(Float64))
end

function _band_path_axes!(
    grid::GridLayout,
    distances::AbstractVector{<:Real},
    node_indices::AbstractVector{<:Integer},
    node_labels::AbstractVector{<:AbstractString},
    branch_ranges::AbstractVector{<:UnitRange{Int}};
    ylabel::AbstractString="",
    title::AbstractString="",
    axis=(;),
)
    branch_axes = Axis[]
    nbranches = length(branch_ranges)

    for (branch_idx, range) in enumerate(branch_ranges)
        is_first = branch_idx == 1
        is_last = branch_idx == nbranches
        branch_ticks, branch_tick_labels = _branch_ticks(distances, node_indices, node_labels, range)

        # Ensure axis is a NamedTuple to avoid spreading positional arguments
        axis_kwargs = axis isa NamedTuple ? axis : (;)
        
        ax = Axis(
            grid[1, branch_idx];
            xticks=(branch_ticks, branch_tick_labels),
            ylabel=is_first ? ylabel : "",
            title=is_first ? title : "",
            xgridvisible=false,
            yticklabelsvisible=is_first,
            yticksvisible=is_first,
            leftspinevisible=is_first,
            rightspinevisible=is_last,
            axis_kwargs...,
        )
        push!(branch_axes, ax)

        xlims!(ax, distances[first(range)], distances[last(range)])
        !isempty(branch_ticks) && vlines!(ax, branch_ticks, color=(:gray, 0.5), linestyle=:dot, linewidth=1.5)

        if !is_first
            hideydecorations!(ax, grid=false)
        end

        colsize!(grid, branch_idx, Auto(_branch_width(distances, range)))
    end

    colgap!(grid, 12)

    for ax in branch_axes[2:end]
        linkyaxes!(branch_axes[1], ax)
    end

    return branch_axes
end
