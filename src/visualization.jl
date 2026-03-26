module Visualization

using Makie
using StaticArrays
using LinearAlgebra
using ..Eliashberg: Dispersion, ElectronicDispersion, KGrid, AbstractKGrid, KPath, band_structure, ε, FreeElectron, TightBinding, RenormalizedDispersion

export visualize_dispersion, dimensionality, visualize_landscape, visualize_spectral_function

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
default_kgrid(::Val{1}) = KGrid([SVector{1}(k) for k in range(-π, π, length=200)], ones(200) / 200.0)
function default_kgrid(::Val{2})
    ks = [SVector{2}(kx, ky) for kx in range(-π, π, length=100) for ky in range(-π, π, length=100)]
    return KGrid(ks, ones(length(ks)) / length(ks))
end
function default_kgrid(::Val{3})
    # For 3D we typically want a high-symmetry path. 
    # Example placeholder: simple cubic Γ -> X -> M -> Γ
    path = [
        [SVector{3}(k, 0.0, 0.0) for k in range(0, π, length=50)]...,
        [SVector{3}(π, k, 0.0) for k in range(0, π, length=50)]...,
        [SVector{3}(π - k, π - k, 0.0) for k in range(0, π, length=50)]...
    ]
    return KGrid(path, ones(length(path)) / length(path))
end


# ============================================================================
# 2. Generic Plotting Pipeline
# ============================================================================

"""
    visualize_dispersion(disp::Dispersion, [kgrid::AbstractKGrid]; kwargs...)

Generates the appropriate Makie plot for the given dispersion model.
Automatically infers the dimension via the `dimensionality(disp)` trait.
"""
function visualize_dispersion(disp::Dispersion, kgrid::Union{AbstractKGrid,Nothing}=nothing; kwargs...)
    D = dimensionality(disp)
    grid = isnothing(kgrid) ? default_kgrid(Val(D)) : kgrid
    return visualize_dispersion(Val(D), disp, grid; kwargs...)
end

# ----------------- 1D Dispatch -----------------
function visualize_dispersion(::Val{1}, disp::Dispersion, kgrid::KGrid{1}; E_Fermi=0.0, axis=(;), kwargs...)
    k_vals = [k[1] for k in kgrid.points]

    # Sort for continuous line plotting
    perm = sortperm(k_vals)
    k_sorted = k_vals[perm]

    # Evaluate bands
    # band_structure(disp, k) returns `Eigen`, we take `.values`
    bands = [real(band_structure(disp, kgrid.points[i]).values) for i in perm]
    num_bands = length(bands[1])

    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel="k", ylabel="E(k)", title="1D Dispersion", axis...)

    for b in 1:num_bands
        E_b = [E[b] for E in bands]
        lines!(ax, k_sorted, E_b; label="Band $b", kwargs...)
    end

    # Add Fermi Level
    hlines!(ax, [E_Fermi], color=:black, linestyle=:dash, label="Fermi Level")

    return fig
end

# ----------------- 2D Dispatch -----------------
function visualize_dispersion(::Val{2}, disp::Dispersion, kgrid::KGrid{2}; E_Fermi=0.0, axis=(;), kwargs...)
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
    ax = Axis(fig[1, 1]; xlabel="kx", ylabel="ky", title="2D Dispersion", axis...)
    hm = heatmap!(ax, kxs, kys, E_matrix; colormap=:viridis, kwargs...)
    Colorbar(fig[1, 2], hm, label="Energy")

    # Fermi surface contour at E = E_Fermi
    contour!(ax, kxs, kys, E_matrix; levels=[E_Fermi], color=:red, linewidth=2, labels=true)

    # Brillouin Zone boundaries (placeholder box)
    lines!(ax, [-π, π, π, -π, -π], [-π, -π, π, π, -π], color=:white, linestyle=:dash)

    return fig
end

# ----------------- 3D Dispatch (Band Structure along KPath) -----------------

function visualize_dispersion(::Val{3}, disp::ElectronicDispersion, kpath::KPath; E_Fermi=0.0, axis=(;), kwargs...)
    # 1. 计算一维路径的累积距离 (用于 X 轴)
    dist = 0.0
    distances = zeros(length(kpath.points))
    for i in 2:length(kpath.points)
        dist += norm(kpath.points[i] - kpath.points[i-1])
        distances[i] = dist
    end

    # 2. 计算每个 k 点的能带矩阵本征值
    bands = [real(band_structure(disp, k).values) for k in kpath.points]
    num_bands = length(bands[1])

    # 3. 提取高对称点的距离坐标和标签
    tick_positions = distances[kpath.node_indices]
    tick_labels = kpath.node_labels

    # 4. 初始化图表 (将 X 轴刻度替换为高对称点标签)
    fig = Figure(size=(800, 600))
    ax = Axis(fig[1, 1];
        ylabel="Energy E(k)",
        title="3D Band Structure",
        xticks=(tick_positions, tick_labels), # 神来之笔：修改 X 轴刻度
        xgridvisible=false, # 关掉默认网格，因为我们要自己画高对称竖线
        axis...)

    # 5. 画高对称点的垂直虚线
    vlines!(ax, tick_positions, color=:gray, linestyle=:dot, linewidth=1.5)

    # 6. 画能带
    for b in 1:num_bands
        E_b = [E[b] for E in bands]
        lines!(ax, distances, E_b; linewidth=2.5, kwargs...)
    end

    # 7. 画费米能级
    hlines!(ax, [E_Fermi], color=:black, linestyle=:dash, linewidth=1.5)

    # 设置 X 轴的显示范围，让图看起来更紧凑
    xlims!(ax, distances[1], distances[end])

    return fig
end

# ----------------- 3D Isosurface (Fermi Surface) -----------------
"""
    visualize_dispersion(::Val{3}, disp::ElectronicDispersion, kgrid::KGrid{3}; E_Fermi=0.0, kwargs...)

Generates a 3D isosurface (Fermi Surface) plot for a 3D volume grid.
Extracts the contour where E(kx, ky, kz) = E_Fermi.
"""
function visualize_dispersion(::Val{3}, disp::ElectronicDispersion, kgrid::KGrid{3};
    E_Fermi=0.0, axis=(;), kwargs...)

    # 1. Extract unique kx, ky, kz from the KGrid
    kxs = unique(sort([k[1] for k in kgrid.points]))
    kys = unique(sort([k[2] for k in kgrid.points]))
    kzs = unique(sort([k[3] for k in kgrid.points]))

    # 2. Evaluate Dispersion over the 3D Volume
    E_volume = zeros(Float64, length(kxs), length(kys), length(kzs))
    for (i, kx) in enumerate(kxs)
        for (j, ky) in enumerate(kys)
            for (k, kz) in enumerate(kzs)
                # We assume the dispersion returns a Hermitian matrix; 
                # pick the first eigenvalue (lowest band) as the reference for the isosurface
                vals = real(band_structure(disp, SVector(kx, ky, kz)).values)
                E_volume[i, j, k] = vals[1]
            end
        end
    end

    # 3. Setup Axis3 and Isosurface Rendering
    fig = Figure(size=(800, 800))
    ax = Axis3(fig[1, 1];
        xlabel=L"k_x", ylabel=L"k_y", zlabel=L"k_z",
        title="3D Fermi Surface (E = $E_Fermi)",
        axis...)

    # Use Makie's contour! for 3D isosurface extraction via Marching Cubes
    # Note: For 3D volume-like data, Makie expects limits (start, end) for each axis
    contour!(ax,
        (kxs[1], kxs[end]),
        (kys[1], kys[end]),
        (kzs[1], kzs[end]),
        E_volume;
        levels=[E_Fermi],
        colormap=:viridis,
        alpha=0.6,
        transparency=true,
        kwargs...)

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

# ============================================================================
# 5. Susceptibility & Spectral Visualization
# ============================================================================

"""
    visualize_landscape(::Val{1}, qgrid::KGrid{1}, landscape_vector::Vector{Float64}; axis=(;), kwargs...)

Plots a 1D line plot of the static susceptibility landscape.
"""
function visualize_landscape(::Val{1}, qgrid::KGrid{1}, landscape_vector::Vector{Float64}; axis=(;), kwargs...)
    qs = [q[1] for q in qgrid.points]
    
    # Sort for continuous plotting
    perm = sortperm(qs)
    qs_sorted = qs[perm]
    vals_sorted = landscape_vector[perm]
    
    fig = Figure(size=(800, 500))
    ax = Axis(fig[1, 1]; 
        xlabel=L"q", ylabel=L"\chi_0(q, \omega=0)", 
        title="1D Instability Landscape",
        axis...)
    
    lines!(ax, qs_sorted, vals_sorted; linewidth=2, color=:crimson, kwargs...)
    
    # Draw Fermi level or zero line for reference
    hlines!(ax, [0.0], color=:black, alpha=0.3)
    
    return fig
end

"""
    visualize_landscape(::Val{2}, qgrid::KGrid{2}, landscape_matrix::Matrix{Float64}; axis=(;), kwargs...)

Plots a 2D heatmap of the static susceptibility landscape.
Assumes `landscape_matrix` matches the dimensions of unique (kx, ky) in `qgrid`.
"""
function visualize_landscape(::Val{2}, qgrid::KGrid{2}, landscape_matrix::Matrix{Float64}; axis=(;), kwargs...)
    # Extract unique kx and ky for the axes
    kxs = unique(sort([k[1] for k in qgrid.points]))
    kys = unique(sort([k[2] for k in qgrid.points]))
    
    fig = Figure(size=(800, 600))
    ax = Axis(fig[1, 1]; 
        xlabel=L"q_x", ylabel=L"q_y", 
        title="Instability Landscape (Static Susceptibility)",
        axis...)
    
    hm = heatmap!(ax, kxs, kys, landscape_matrix; colormap=:magma, kwargs...)
    Colorbar(fig[1, 2], hm, label=L"\chi_0(q, \omega=0)")
    
    # Optional: Draw BZ boundaries if provided or assumed standard
    lines!(ax, [-π, π, π, -π, -π], [-π, -π, π, π, -π], color=:white, linestyle=:dash, alpha=0.5)
    
    return fig
end

"""
    visualize_spectral_function(qpath::KPath{D}, omegas::AbstractVector{Float64}, spectral_matrix::Matrix{Float64}; axis=(;), kwargs...) where {D}

Plots the dynamic spectral function A(q, ω) along a high-symmetry path.
- X-axis: 1D path distance.
- Y-axis: Frequency ω.
- Color: Spectral weight A(q, ω).
"""
function visualize_spectral_function(qpath::KPath{D}, omegas::AbstractVector{Float64}, spectral_matrix::Matrix{Float64}; axis=(;), kwargs...) where {D}
    # 1. Calculate cumulative distance along qpath
    dist = 0.0
    distances = zeros(length(qpath.points))
    for i in 2:length(qpath.points)
        dist += norm(qpath.points[i] - qpath.points[i-1])
        distances[i] = dist
    end
    
    # 2. Extract symmetry node positions and labels
    tick_positions = distances[qpath.node_indices]
    tick_labels = qpath.node_labels
    
    # 3. Setup Figure and Axis
    fig = Figure(size=(900, 600))
    ax = Axis(fig[1, 1];
        ylabel=L"\omega",
        title="Dynamic Spectral Function " * L"A(q, \omega)",
        xticks=(tick_positions, tick_labels),
        xgridvisible=false,
        axis...)
    
    # 4. Plot Heatmap (X: distances, Y: omegas)
    hm = heatmap!(ax, distances, omegas, spectral_matrix; colormap=:inferno, kwargs...)
    Colorbar(fig[1, 2], hm, label=L"\text{Im}[\chi(q, \omega)]")
    
    # 5. Draw vertical symmetry lines
    vlines!(ax, tick_positions, color=:white, linestyle=:dash, linewidth=0.8, alpha=0.6)
    
    # Ensure tight x-limits
    xlims!(ax, distances[1], distances[end])
    
    return fig
end

end # module
