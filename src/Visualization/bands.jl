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

plot_dispersion_surface(data::DispersionSurfaceData; kwargs...) =
    plot_dispersion_surface(data.kxs, data.kys, data.energy_matrix; kwargs...)

"""
    plot_band_structure(kpath, band_matrix; E_Fermi=0.0, axis=(;), kwargs...)

Plot a band structure along a labelled path in parameter space.
"""
function plot_band_structure(kpath::KPath, band_matrix::AbstractMatrix{<:Real}; E_Fermi=0.0, colors=Makie.wong_colors(), axis=(;), kwargs...)
    size(band_matrix, 1) == length(kpath.points) || throw(DimensionMismatch("Band matrix row count must match the number of path samples."))
    distances = path_distances(kpath)
    tick_positions = distances[kpath.node_indices]
    tick_labels = kpath.node_labels

    fig = Figure(size=(800, 600))
    ax = Axis(fig[1, 1]; ylabel="Energy E(k)", title="$(length(first(kpath.points)))D Band Structure",
        xticks=(tick_positions, tick_labels), xgridvisible=false, axis...)

    vlines!(ax, tick_positions, color=(:gray, 0.5), linestyle=:dot, linewidth=1.5)

    for band_idx in axes(band_matrix, 2)
        c = colors[mod1(band_idx, length(colors))]
        lines!(ax, distances, band_matrix[:, band_idx], color=c; linewidth=2.5, kwargs...)
    end

    hlines!(ax, [E_Fermi], color=(:black, 0.7), linestyle=:dash, linewidth=1.5)
    xlims!(ax, distances[1], distances[end])
    return fig
end

plot_band_structure(data::BandStructureData; kwargs...) =
    plot_band_structure(data.kpath, data.bands; kwargs...)

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

plot_fermi_surface(data::FermiSurfaceData; kwargs...) =
    plot_fermi_surface(data.kxs, data.kys, data.kzs, data.energy_volume; kwargs...)

function _plot_renormalized_bands(
    Ts::AbstractVector{<:Real},
    kpath::KPath,
    bare_bands::AbstractMatrix{<:Real},
    renormalized_bands::AbstractArray{<:Real,3},
    gaps::AbstractArray{<:Real};
    band_limits=(-2.5, 2.5),
    colors=Makie.wong_colors()
)
    n_path = length(kpath.points)
    size(bare_bands, 1) == n_path || throw(DimensionMismatch("Bare-band matrix row count must match the number of path samples."))
    size(renormalized_bands, 1) == n_path || throw(DimensionMismatch("Renormalized-band tensor first dimension must match the number of path samples."))
    size(renormalized_bands, 3) == length(Ts) || throw(DimensionMismatch("Renormalized-band tensor third dimension must match the temperature axis."))
    _validate_gap_storage(gaps, length(Ts))

    distances = path_distances(kpath)
    tick_positions = distances[kpath.node_indices]
    labels = kpath.node_labels
    fig = Figure(size=(1400, 420), fontsize=16)

    for (idx, T) in enumerate(Ts)
        is_first = idx == 1
        ax = Axis(fig[1, idx],
            title=_renormalized_band_title(T, gaps, idx),
            titlesize=16,
            xticks=(tick_positions, labels),
            ylabel=is_first ? "Energy E(k)" : "",
            xgridvisible=false,
            yticksvisible=is_first,
            yticklabelsvisible=is_first
        )

        hlines!(ax, [0.0], color=:gray, linestyle=:dash, linewidth=1)
        vlines!(ax, tick_positions, color=:lightgray, linestyle=:dot, linewidth=1.5)

        for band_idx in axes(bare_bands, 2)
            label = band_idx == 1 ? "Bare Band" : nothing
            lines!(ax, distances, bare_bands[:, band_idx], color=:black, linestyle=:dot, alpha=0.4, linewidth=2, label=label)
        end

        band_slice = @view renormalized_bands[:, :, idx]
        for band_idx in axes(band_slice, 2)
            color = colors[mod1(band_idx, length(colors))]
            label = band_idx == 1 ? "Renormalized Band" : nothing
            lines!(ax, distances, band_slice[:, band_idx], color=color, linewidth=3, label=label)
        end

        ylims!(ax, band_limits...)
        xlims!(ax, distances[1], distances[end])

        if is_first
            axislegend(ax, position=:lt, framevisible=false)
        end
    end

    colgap!(fig.layout, 15)
    return fig
end

_renormalized_band_title(T::Real, gaps::AbstractVector{<:Real}, idx::Int) =
    "T = $(round(T, digits=2))   Δ = $(round(gaps[idx], digits=2))"

function _renormalized_band_title(T::Real, gaps::AbstractMatrix{<:Real}, idx::Int)
    components = join((string(round(gap, digits=2)) for gap in gaps[:, idx]), ", ")
    return "T = $(round(T, digits=2))   ϕ = [$components]"
end

"""
    plot_renormalized_bands(data::RenormalizedBandData; band_limits=(-2.5, 2.5))

Plot temperature-indexed renormalized band panels from a typed response object.
"""
plot_renormalized_bands(data::RenormalizedBandData; kwargs...) =
    _plot_renormalized_bands(
        data.temperatures,
        data.kpath,
        data.bare_bands,
        data.renormalized_bands,
        data.gaps;
        kwargs...
    )

visualize_dispersion(k_coords::AbstractVector{<:Real}, band_matrix::AbstractMatrix{<:Real}; kwargs...) =
    plot_dispersion_curves(k_coords, band_matrix; kwargs...)

visualize_dispersion(kxs::AbstractVector{<:Real}, kys::AbstractVector{<:Real}, energy_matrix::AbstractMatrix{<:Real}; kwargs...) =
    plot_dispersion_surface(kxs, kys, energy_matrix; kwargs...)

visualize_dispersion(data::DispersionSurfaceData; kwargs...) =
    plot_dispersion_surface(data; kwargs...)

visualize_dispersion(kpath::KPath, band_matrix::AbstractMatrix{<:Real}; kwargs...) =
    plot_band_structure(kpath, band_matrix; kwargs...)

visualize_dispersion(data::BandStructureData; kwargs...) =
    plot_band_structure(data; kwargs...)

visualize_dispersion(
    kxs::AbstractVector{<:Real},
    kys::AbstractVector{<:Real},
    kzs::AbstractVector{<:Real},
    energy_volume::AbstractArray{<:Real,3};
    kwargs...
) = plot_fermi_surface(kxs, kys, kzs, energy_volume; kwargs...)

visualize_dispersion(data::FermiSurfaceData; kwargs...) =
    plot_fermi_surface(data; kwargs...)

visualize_renormalized_bands(data::RenormalizedBandData; kwargs...) =
    plot_renormalized_bands(data; kwargs...)

Makie.plot(data::DispersionSurfaceData; kwargs...) = plot_dispersion_surface(data; kwargs...)
Makie.plot(data::BandStructureData; kwargs...) = plot_band_structure(data; kwargs...)
Makie.plot(data::FermiSurfaceData; kwargs...) = plot_fermi_surface(data; kwargs...)
Makie.plot(data::RenormalizedBandData; kwargs...) = plot_renormalized_bands(data; kwargs...)
