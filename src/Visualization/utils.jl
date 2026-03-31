# src/Visualization/utils.jl

# ============================================================================
# Dimensionality Trait / Interface
# ============================================================================

"""
    dimensionality(::Dispersion) -> Int

Returns the spatial dimensionality of a dispersion model.
This trait enables dispatching the correct visualization routing.
"""
dimensionality(::T) where {T<:Dispersion} = _extract_D(T)
_extract_D(::Type{<:Dispersion}) = error("Could not infer dimensionality. Please define `dimensionality(::MyModel)` explicitly.")

# Assuming the struct signature has D as the first parameter for these:
_extract_D(::Type{<:FreeElectron{D}}) where D = D
_extract_D(::Type{<:TightBinding{D}}) where D = D
_extract_D(::Type{<:RenormalizedDispersion{D}}) where D = D
_extract_D(::Type{<:MeanFieldDispersion{D}}) where D = D
_extract_D(::Type{<:KagomeLattice}) = 2
_extract_D(::Type{<:Graphene}) = 2
_extract_D(::Type{<:SSHModel}) = 1


# ============================================================================
# K-Grid Integration (Defaults)
# ============================================================================

# Generate appropriate k-points for visualization if not explicitly provided
default_kgrid(::Val{1}) = KGrid([SVector{1}(k) for k in range(-π, π, length=200)], ones(200) / 200.0)

function default_kgrid(::Val{2})
    ks = [SVector{2}(kx, ky) for kx in range(-π, π, length=100) for ky in range(-π, π, length=100)]
    return KGrid(ks, ones(length(ks)) / length(ks))
end

function default_kgrid(::Val{3})
    path = [
        [SVector{3}(k, 0.0, 0.0) for k in range(0, π, length=50)]...,
        [SVector{3}(π, k, 0.0) for k in range(0, π, length=50)]...,
        [SVector{3}(π - k, π - k, 0.0) for k in range(0, π, length=50)]...
    ]
    return KGrid(path, ones(length(path)) / length(path))
end

"""
    path_distances(path::KPath)

Return the cumulative path-length coordinate associated with each point in a
piecewise-linear path.
"""
function path_distances(path::KPath)
    distances = zeros(length(path.points))
    dist = 0.0
    for idx in 2:length(path.points)
        dist += norm(path.points[idx] - path.points[idx - 1])
        distances[idx] = dist
    end
    return distances
end
