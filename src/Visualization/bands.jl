# src/Visualization/bands.jl

"""
    plot_dispersion_curves(k_coords, band_matrix; E_Fermi=0.0, axis=(;), kwargs...)

Plot one or more band curves over a one-dimensional coordinate axis.
"""
function plot_dispersion_curves(k_coords::AbstractVector{<:Real}, band_matrix::AbstractMatrix{<:Real}; E_Fermi=0.0, axis=(;), kwargs...)
    size(band_matrix, 1) == length(k_coords) || throw(DimensionMismatch("Band matrix row count must match the coordinate axis length."))

    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel="k", ylabel="E(k)", title="1D Dispersion", axis...)

    for band_idx in axes(band_matrix, 2)
        lines!(ax, k_coords, band_matrix[:, band_idx]; label="Band $band_idx", kwargs...)
    end

    hlines!(ax, [E_Fermi], color=:black, linestyle=:dash, label="Fermi Level")
    return fig
end

"""
    plot_dispersion_surface(kxs, kys, energy_matrix; E_Fermi=0.0, axis=(;), kwargs...)

Plot a two-dimensional scalar field together with an iso-energy contour.
"""
function plot_dispersion_surface(kxs::AbstractVector{<:Real}, kys::AbstractVector{<:Real}, energy_matrix::AbstractMatrix{<:Real}; E_Fermi=0.0, axis=(;), kwargs...)
    size(energy_matrix) == (length(kxs), length(kys)) || throw(DimensionMismatch("Energy matrix shape must match the provided axes."))

    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel=L"k_x", ylabel=L"k_y", title="2D Dispersion", axis...)
    hm = heatmap!(ax, kxs, kys, energy_matrix; colormap=:viridis, kwargs...)
    Colorbar(fig[1, 2], hm, label="Energy")
    contour!(ax, kxs, kys, energy_matrix; levels=[E_Fermi], color=:red, linewidth=2, labels=true)
    lines!(ax, [-π, π, π, -π, -π], [-π, -π, π, π, -π], color=:white, linestyle=:dash)
    return fig
end

"""
    plot_band_structure(kpath, band_matrix; E_Fermi=0.0, axis=(;), kwargs...)

Plot a band structure along a labelled path in parameter space.
"""
function plot_band_structure(kpath::KPath, band_matrix::AbstractMatrix{<:Real}; E_Fermi=0.0, axis=(;), kwargs...)
    size(band_matrix, 1) == length(kpath.points) || throw(DimensionMismatch("Band matrix row count must match the number of path samples."))
    distances = path_distances(kpath)
    tick_positions = distances[kpath.node_indices]
    tick_labels = kpath.node_labels

    fig = Figure(size=(800, 600))
    ax = Axis(fig[1, 1]; ylabel="Energy E(k)", title="$(length(first(kpath.points)))D Band Structure",
        xticks=(tick_positions, tick_labels), xgridvisible=false, axis...)

    vlines!(ax, tick_positions, color=:gray, linestyle=:dot, linewidth=1.5)

    for band_idx in axes(band_matrix, 2)
        lines!(ax, distances, band_matrix[:, band_idx]; linewidth=2.5, kwargs...)
    end

    hlines!(ax, [E_Fermi], color=:black, linestyle=:dash, linewidth=1.5)
    xlims!(ax, distances[1], distances[end])
    return fig
end

"""
    plot_fermi_surface(kxs, kys, kzs, energy_volume; E_Fermi=0.0, axis=(;), kwargs...)

Render an isosurface from a precomputed three-dimensional scalar field.
"""
function plot_fermi_surface(
    kxs::AbstractVector{<:Real},
    kys::AbstractVector{<:Real},
    kzs::AbstractVector{<:Real},
    energy_volume::AbstractArray{<:Real,3};
    E_Fermi=0.0,
    axis=(;),
    kwargs...
)
    size(energy_volume) == (length(kxs), length(kys), length(kzs)) || throw(DimensionMismatch("Volume shape must match the provided axes."))

    fig = Figure(size=(900, 800), fontsize=16)
    ax = Axis3(fig[1, 1]; xlabel=L"k_x", ylabel=L"k_y", zlabel=L"k_z",
        title="Interactive 3D Fermi Surface", elevation=π / 6, azimuth=π / 4, axis...)

    E_min, E_max = minimum(energy_volume), maximum(energy_volume)
    sg = SliderGrid(fig[2, 1], (label="μ (Fermi level)", range=range(E_min, E_max, length=300), startvalue=Float64(E_Fermi)))
    mu_slider = sg.sliders[1].value
    iso_level = lift(mu -> Float32[mu], mu_slider)

    contour!(ax, (first(kxs), last(kxs)), (first(kys), last(kys)), (first(kzs), last(kzs)), energy_volume;
        levels=iso_level, colormap=:viridis, alpha=0.5, transparency=true, kwargs...)
    return fig
end

"""
    plot_renormalized_bands(Ts, kpath, bare_bands, hole_bands, particle_bands, gaps; band_limits=(-2.5, 2.5))

Plot temperature-indexed renormalized band panels from precomputed spectral data.
"""
function plot_renormalized_bands(
    Ts::AbstractVector{<:Real},
    kpath::KPath,
    bare_bands::AbstractVector{<:Real},
    hole_bands::AbstractMatrix{<:Real},
    particle_bands::AbstractMatrix{<:Real},
    gaps::AbstractVector{<:Real};
    band_limits=(-2.5, 2.5)
)
    n_path = length(kpath.points)
    size(hole_bands) == (n_path, length(Ts)) || throw(DimensionMismatch("Hole-band matrix shape must be (length(kpath), length(Ts))."))
    size(particle_bands) == (n_path, length(Ts)) || throw(DimensionMismatch("Particle-band matrix shape must be (length(kpath), length(Ts))."))
    length(bare_bands) == n_path || throw(DimensionMismatch("Bare-band vector length must match the number of path samples."))
    length(gaps) == length(Ts) || throw(DimensionMismatch("Gap vector length must match the temperature axis."))

    distances = path_distances(kpath)
    tick_positions = distances[kpath.node_indices]
    labels = kpath.node_labels
    fig = Figure(size=(1400, 420), fontsize=16)

    for (idx, T) in enumerate(Ts)
        is_first = idx == 1
        ax = Axis(fig[1, idx],
            title="T = $(round(T, digits=2))   Δ = $(round(gaps[idx], digits=2))",
            titlesize=16,
            xticks=(tick_positions, labels),
            ylabel=is_first ? "Energy E(k)" : "",
            xgridvisible=false,
            yticksvisible=is_first,
            yticklabelsvisible=is_first
        )

        hlines!(ax, [0.0], color=:gray, linestyle=:dash, linewidth=1)
        vlines!(ax, tick_positions, color=:lightgray, linestyle=:dot, linewidth=1.5)
        lines!(ax, distances, bare_bands, color=:black, linestyle=:dot, alpha=0.4, linewidth=2, label="Bare Band")
        lines!(ax, distances, hole_bands[:, idx], color=:royalblue, linewidth=3, label="Hole Band")
        lines!(ax, distances, particle_bands[:, idx], color=:crimson, linewidth=3, label="Particle Band")

        ylims!(ax, band_limits...)
        xlims!(ax, distances[1], distances[end])

        if is_first
            axislegend(ax, position=:lt, framevisible=false)
        end
    end

    colgap!(fig.layout, 15)
    return fig
end

visualize_dispersion(k_coords::AbstractVector{<:Real}, band_matrix::AbstractMatrix{<:Real}; kwargs...) =
    plot_dispersion_curves(k_coords, band_matrix; kwargs...)

visualize_dispersion(kxs::AbstractVector{<:Real}, kys::AbstractVector{<:Real}, energy_matrix::AbstractMatrix{<:Real}; kwargs...) =
    plot_dispersion_surface(kxs, kys, energy_matrix; kwargs...)

visualize_dispersion(kpath::KPath, band_matrix::AbstractMatrix{<:Real}; kwargs...) =
    plot_band_structure(kpath, band_matrix; kwargs...)

visualize_dispersion(
    kxs::AbstractVector{<:Real},
    kys::AbstractVector{<:Real},
    kzs::AbstractVector{<:Real},
    energy_volume::AbstractArray{<:Real,3};
    kwargs...
) = plot_fermi_surface(kxs, kys, kzs, energy_volume; kwargs...)

visualize_renormalized_bands(args...; kwargs...) = plot_renormalized_bands(args...; kwargs...)
