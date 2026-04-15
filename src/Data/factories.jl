# src/Data/factories.jl

function compute_landscape_line_data(qs::AbstractVector{<:Real}, values::AbstractVector{<:Real})
    return LandscapeLineData(qs, values)
end

function compute_landscape_line_data(grid::AbstractKGrid{1}, values::AbstractVector{<:Real})
    return compute_landscape_line_data([point[1] for point in grid.points], values)
end

function compute_landscape_surface_data(
    qxs::AbstractVector{<:Real},
    qys::AbstractVector{<:Real},
    landscape_matrix::AbstractMatrix{<:Real},
)
    return LandscapeSurfaceData(qxs, qys, landscape_matrix)
end

function compute_landscape_surface_data(grid::AbstractKGrid{2}, landscape_matrix::AbstractMatrix{<:Real})
    qxs = unique(sort([point[1] for point in grid.points]))
    qys = unique(sort([point[2] for point in grid.points]))
    return compute_landscape_surface_data(qxs, qys, landscape_matrix)
end
