module Visualization

using Makie
using StaticArrays
using LinearAlgebra
using ..Eliashberg: Dispersion, ElectronicDispersion, KGrid, AbstractKGrid, band_structure, ε, FreeElectron, TightBinding, RenormalizedDispersion

export visualize_dispersion, dimensionality

# ============================================================================
# 1. Dimensionality Trait / Interface
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


# ============================================================================
# 3. K-Grid Integration (Defaults)
# ============================================================================

# Generate appropriate k-points for visualization if not explicitly provided
default_kgrid(::Val{1}) = KGrid([SVector{1}(k) for k in range(-π, π, length=200)], ones(200)/200.0)
function default_kgrid(::Val{2})
    ks = [SVector{2}(kx, ky) for kx in range(-π, π, length=100) for ky in range(-π, π, length=100)]
    return KGrid(ks, ones(length(ks))/length(ks))
end
function default_kgrid(::Val{3})
    # For 3D we typically want a high-symmetry path. 
    # Example placeholder: simple cubic Γ -> X -> M -> Γ
    path = [
        [SVector{3}(k, 0.0, 0.0) for k in range(0, π, length=50)]...,
        [SVector{3}(π, k, 0.0) for k in range(0, π, length=50)]...,
        [SVector{3}(π-k, π-k, 0.0) for k in range(0, π, length=50)]...
    ]
    return KGrid(path, ones(length(path))/length(path))
end


# ============================================================================
# 2. Generic Plotting Pipeline
# ============================================================================

"""
    visualize_dispersion(disp::Dispersion, [kgrid::AbstractKGrid]; kwargs...)

Generates the appropriate Makie plot for the given dispersion model.
Automatically infers the dimension via the `dimensionality(disp)` trait.
"""
function visualize_dispersion(disp::Dispersion, kgrid::Union{AbstractKGrid, Nothing} = nothing; kwargs...)
    D = dimensionality(disp)
    grid = isnothing(kgrid) ? default_kgrid(Val(D)) : kgrid
    return visualize_dispersion(Val(D), disp, grid; kwargs...)
end

# ----------------- 1D Dispatch -----------------
function visualize_dispersion(::Val{1}, disp::Dispersion, kgrid::AbstractKGrid; axis=(;), kwargs...)
    k_vals = [k[1] for k in kgrid.points]
    
    # Sort for continuous line plotting
    perm = sortperm(k_vals)
    k_sorted = k_vals[perm]
    
    # Evaluate bands
    # band_structure(disp, k) returns `Eigen`, we take `.values`
    bands = [real(band_structure(disp, kgrid.points[i]).values) for i in perm]
    num_bands = length(bands[1])
    
    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel = "k", ylabel = "E(k)", title="1D Dispersion", axis...)
    
    for b in 1:num_bands
        E_b = [E[b] for E in bands]
        lines!(ax, k_sorted, E_b; label="Band $b", kwargs...)
    end
    
    # Add Fermi Level at E=0 (assuming E is relative to EF)
    hlines!(ax, [0.0], color=:black, linestyle=:dash, label="Fermi Level")
    
    return fig
end

# ----------------- 2D Dispatch -----------------
function visualize_dispersion(::Val{2}, disp::Dispersion, kgrid::AbstractKGrid; axis=(;), kwargs...)
    # Heatmap + Contour
    kxs = unique(sort([k[1] for k in kgrid.points]))
    kys = unique(sort([k[2] for k in kgrid.points]))
    
    # If the user passed an unstructured grid, this reshapes it. For robust 2D,
    # evaluate over the outer product of `kxs` and `kys`.
    E_matrix = zeros(Float64, length(kxs), length(kys))
    
    for (i, kx) in enumerate(kxs)
        for (j, ky) in enumerate(kys)
            # Taking the ground state / lowest band or assuming 1 band for heatmap
            vals = real(band_structure(disp, SVector(kx, ky)).values)
            # (If multiple bands, could plot multiple figures or a slider. For skeleton we take band 1)
            E_matrix[i, j] = vals[1] 
        end
    end
    
    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel = "kx", ylabel = "ky", title="2D Dispersion", axis...)
    hm = heatmap!(ax, kxs, kys, E_matrix; colormap=:viridis, kwargs...)
    Colorbar(fig[1, 2], hm, label="Energy")
    
    # Fermi surface contour E=0
    contour!(ax, kxs, kys, E_matrix; levels=[0.0], color=:red, linewidth=2, labels=true)
    
    # Brillouin Zone boundaries (placeholder box)
    lines!(ax, [-π, π, π, -π, -π], [-π, -π, π, π, -π], color=:white, linestyle=:dash)
    
    return fig
end

# ----------------- 3D Dispatch -----------------
function visualize_dispersion(::Val{3}, disp::Dispersion, kgrid::AbstractKGrid; axis=(;), kwargs...)
    # We assume `kgrid.points` forms an ordered 1D path through 3D space
    dist = 0.0
    distances = zeros(length(kgrid.points))
    for i in 2:length(kgrid.points)
        dist += norm(kgrid.points[i] - kgrid.points[i-1])
        distances[i] = dist
    end
    
    bands = [real(band_structure(disp, k).values) for k in kgrid.points]
    num_bands = length(bands[1])
    
    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel = "Path Distance", ylabel = "E(k)", title="3D Band Structure", axis...)
    
    for b in 1:num_bands
        E_b = [E[b] for E in bands]
        lines!(ax, distances, E_b; kwargs...)
    end
    
    hlines!(ax, [0.0], color=:black, linestyle=:dash)
    
    return fig
end

# ============================================================================
# 4. "Zero-Boilerplate" Extensibility Example
# ============================================================================
# Suppose we want to define a Kagome Lattice tomorrow. Here's all the user has to do.
#
# struct KagomeLattice <: ElectronicDispersion
#     t::Float64
#     EF::Float64
# end
# 
# # 1. Define the Dispersion trait
# Visualization.dimensionality(::KagomeLattice) = 2
# 
# # 2. Define the physics method `ε(k, model)` Returning a Hermitian Matrix (e.g. 3x3 for Kagome)
# function Eliashberg.ε(k::SVector{2, Float64}, model::KagomeLattice)
#     kx, ky = k[1], k[2]
#     h12 = 2 * model.t * cos(kx / 2)
#     h13 = 2 * model.t * cos(ky / 2)
#     h23 = 2 * model.t * cos((kx - ky) / 2)
#     
#     H = [
#         0     h12   h13;
#         h12   0     h23;
#         h13   h23   0
#     ]
#     return Hermitian(H) - (model.EF * I) # 3x3 matrix
# end
# 
# # 3. That's it! 
# # Calling `visualize_dispersion(KagomeLattice(1.0, 0.0))` will automatically generate a 2D Heatmap
# # over the default 2D KGrid, calculate the 3x3 eigenvalues at each point, and pick band 1.
# # (If you modify the 2D visualizer to handle multiple bands, e.g. plotting sliders for band index, it works automatically).
# ============================================================================

end # module
