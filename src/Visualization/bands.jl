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
            axis...,
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

function plot_band_structure(kpath::KPath, band_matrix::AbstractMatrix{<:Real}; E_Fermi=0.0, band_color=:royalblue, axis=(;), kwargs...)
    points = path_points(kpath)
    size(band_matrix, 1) == length(points) || throw(DimensionMismatch("Band matrix row count must match the number of path samples."))
    distances = path_distances(kpath)
    node_indices, tick_labels = path_node_metadata(kpath)
    branch_ranges = path_branch_ranges(kpath)

    fig = Figure(size=(800, 600))
    branch_grid = fig[1, 1] = GridLayout()
    branch_axes = _band_path_axes!(
        branch_grid,
        distances,
        node_indices,
        tick_labels,
        branch_ranges;
        ylabel="Energy E(k)",
        title="$(length(first(points)))D Band Structure",
        axis=axis,
    )

    for band_idx in axes(band_matrix, 2)
        for (ax, range) in zip(branch_axes, branch_ranges)
            lines!(ax, distances[range], band_matrix[range, band_idx], color=band_color; linewidth=2.5, kwargs...)
        end
    end

    for ax in branch_axes
        hlines!(ax, [E_Fermi], color=(:black, 0.7), linestyle=:dash, linewidth=1.5)
    end

    return fig
end

plot_band_structure(data::BandStructureData; kwargs...) =
    plot_band_structure(data.kpath, data.bands; kwargs...)

"""
    plot_wannier90_band_structure(bands_filename::String; labelinfo_filename=nothing, kwargs...)

Parse Wannier90 `*_band.dat` output and render it with the standard band
structure plotter. A sibling `*.labelinfo.dat` file is picked up
automatically when available.
"""
function plot_wannier90_band_structure(
    bands_filename::String;
    labelinfo_filename::Union{Nothing, AbstractString}=nothing,
    kwargs...
)
    data = band_data_from_wannier90_bands(bands_filename; labelinfo_filename)
    return plot_band_structure(data; kwargs...)
end

"""
    plot_wannier90_tb_band_comparison(
        comparison::Wannier90BandComparison;
        model_label="TB model (shifted)",
        reference_label="Wannier90 band.dat",
        model_color=:royalblue,
        reference_color=(:darkorange, 0.65),
        linewidth=2.5,
        markersize=4,
        legend_position=:rb,
        axis=(;),
    )

Overlay a reconstructed Wannier90 tight-binding model against the reference
Wannier90 `*_band.dat` energies on the exact sampled path.
"""
function plot_wannier90_tb_band_comparison(
    comparison::Wannier90BandComparison;
    model_label::AbstractString="TB model (shifted)",
    reference_label::AbstractString="Wannier90 band.dat",
    model_color=:royalblue,
    reference_color=(:darkorange, 0.65),
    linewidth::Real=2.5,
    markersize::Real=4,
    legend_position=:rb,
    axis=(;),
)
    reference = comparison.reference
    shifted_model = comparison.shifted_model
    distances = path_distances(reference.kpath)
    node_indices, labels = path_node_metadata(reference.kpath)
    branch_ranges = path_branch_ranges(reference.kpath)

    fig = Figure(size=(900, 650))
    branch_grid = fig[1, 1] = GridLayout()
    branch_axes = _band_path_axes!(
        branch_grid,
        distances,
        node_indices,
        labels,
        branch_ranges;
        ylabel="Energy (eV)",
        title="Wannier90 TB vs band.dat",
        axis=axis,
    )

    for band_idx in axes(reference.bands, 2)
        for (range_idx, (ax, range)) in enumerate(zip(branch_axes, branch_ranges))
            lines!(
                ax,
                distances[range],
                shifted_model.bands[range, band_idx];
                color=model_color,
                linewidth=linewidth,
                label=(band_idx == 1 && range_idx == 1 ? model_label : nothing),
            )
            scatter!(
                ax,
                distances[range],
                reference.bands[range, band_idx];
                color=reference_color,
                markersize=markersize,
                label=(band_idx == 1 && range_idx == 1 ? reference_label : nothing),
            )
        end
    end

    axislegend(first(branch_axes), position=legend_position, framevisible=false)
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

plot_fermi_surface(data::FermiSurfaceData; kwargs...) =
    plot_fermi_surface(data.kxs, data.kys, data.kzs, data.energy_volume; kwargs...)

function _plot_renormalized_bands(
    Ts::AbstractVector{<:Real},
    kpath::KPath,
    bare_bands::AbstractMatrix{<:Real},
    renormalized_bands::AbstractArray{<:Real,3},
    gaps::AbstractArray{<:Real};
    band_limits=(-2.5, 2.5),
    bare_band_color=:black,
    renormalized_band_color=:royalblue,
)
    n_path = length(kpath)
    size(bare_bands, 1) == n_path || throw(DimensionMismatch("Bare-band matrix row count must match the number of path samples."))
    size(renormalized_bands, 1) == n_path || throw(DimensionMismatch("Renormalized-band tensor first dimension must match the number of path samples."))
    size(renormalized_bands, 3) == length(Ts) || throw(DimensionMismatch("Renormalized-band tensor third dimension must match the temperature axis."))
    _validate_gap_storage(gaps, length(Ts))

    distances = path_distances(kpath)
    node_indices, labels = path_node_metadata(kpath)
    branch_ranges = path_branch_ranges(kpath)
    fig = Figure(size=(1400, 420), fontsize=16)

    for (idx, T) in enumerate(Ts)
        is_first = idx == 1
        panel_grid = fig[1, idx] = GridLayout()
        branch_axes = _band_path_axes!(
            panel_grid,
            distances,
            node_indices,
            labels,
            branch_ranges;
            ylabel=is_first ? "Energy E(k)" : "",
            title=_renormalized_band_title(T, gaps, idx),
            axis=(; titlesize=16),
        )

        for band_idx in axes(bare_bands, 2)
            label = band_idx == 1 ? "Bare Band" : nothing
            for (range_idx, (ax, range)) in enumerate(zip(branch_axes, branch_ranges))
                lines!(ax, distances[range], bare_bands[range, band_idx], color=bare_band_color, linestyle=:dot, alpha=0.4, linewidth=2, label=(range_idx == 1 ? label : nothing))
            end
        end

        band_slice = @view renormalized_bands[:, :, idx]
        for band_idx in axes(band_slice, 2)
            label = band_idx == 1 ? "Renormalized Band" : nothing
            for (range_idx, (ax, range)) in enumerate(zip(branch_axes, branch_ranges))
                lines!(ax, distances[range], band_slice[range, band_idx], color=renormalized_band_color, linewidth=3, label=(range_idx == 1 ? label : nothing))
            end
        end

        for ax in branch_axes
            hlines!(ax, [0.0], color=:gray, linestyle=:dash, linewidth=1)
            ylims!(ax, band_limits...)
        end

        if is_first
            axislegend(first(branch_axes), position=:lt, framevisible=false)
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
Makie.plot(data::Wannier90BandComparison; kwargs...) = plot_wannier90_tb_band_comparison(data; kwargs...)
Makie.plot(data::FermiSurfaceData; kwargs...) = plot_fermi_surface(data; kwargs...)
Makie.plot(data::RenormalizedBandData; kwargs...) = plot_renormalized_bands(data; kwargs...)
