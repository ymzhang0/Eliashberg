# src/Visualization/bands.jl

"""
    _plot_dispersion_curves(k_coords, band_matrix; E_Fermi=0.0, axis=(;), kwargs...)

Plot one or more band curves over a one-dimensional coordinate axis. Internal function.
"""
function _plot_dispersion_curves(k_coords::AbstractVector{<:Real}, band_matrix::AbstractMatrix{<:Real}; E_Fermi=0.0, axis=(;), kwargs...)
    return with_stage_log(
        "Plot dispersion curves";
        context=(n_coords=length(k_coords), n_bands=size(band_matrix, 2), E_Fermi=Float64(E_Fermi)),
        summarize_result=result -> (figure_type=string(typeof(result)),),
    ) do
        size(band_matrix, 1) == length(k_coords) || throw(DimensionMismatch("Band matrix row count must match the coordinate axis length."))

        fig = Figure()
        ax = Axis(fig[1, 1]; xlabel="k", ylabel="E(k)", title="1D Dispersion", axis...)

        for band_idx in axes(band_matrix, 2)
            lines!(ax, k_coords, band_matrix[:, band_idx]; label="Band $band_idx", kwargs...)
        end

        hlines!(ax, [E_Fermi], color=:black, linestyle=:dash, label="Fermi Level")
        return fig
    end
end

"""
    _plot_dispersion_surface(kxs, kys, energy_matrix; E_Fermi=0.0, axis=(;), kwargs...)

Plot a two-dimensional scalar field together with an iso-energy contour. Internal function.
"""
function _plot_dispersion_surface(kxs::AbstractVector{<:Real}, kys::AbstractVector{<:Real}, energy_matrix::AbstractMatrix{<:Real}; E_Fermi=0.0, axis=(;), kwargs...)
    return with_stage_log(
        "Plot dispersion surface";
        context=(nx=length(kxs), ny=length(kys), E_Fermi=Float64(E_Fermi)),
        summarize_result=result -> (figure_type=string(typeof(result)),),
    ) do
        size(energy_matrix) == (length(kxs), length(kys)) || throw(DimensionMismatch("Energy matrix shape must match the provided axes."))

        fig = Figure()
        ax = Axis(fig[1, 1]; xlabel=L"k_x", ylabel=L"k_y", title="2D Dispersion", axis...)
        hm = heatmap!(ax, kxs, kys, energy_matrix; colormap=:viridis, kwargs...)
        Colorbar(fig[1, 2], hm, label="Energy")
        contour!(ax, kxs, kys, energy_matrix; levels=[E_Fermi], color=:red, linewidth=2, labels=true)
        lines!(ax, [-π, π, π, -π, -π], [-π, -π, π, π, -π], color=:white, linestyle=:dash)
        return fig
    end
end

_plot_dispersion_surface(data::DispersionSurfaceData; kwargs...) =
    _plot_dispersion_surface(data.kxs, data.kys, data.energy_matrix; kwargs...)

"""
    plot_band_structure(kpath, band_matrix; E_Fermi=0.0, axis=(;), kwargs...)

Plot a band structure along a labelled path in parameter space.
"""

# Band structure plotting core logic follows

function _plot_band_structure(kpath::KPath, band_matrix::AbstractMatrix{<:Real}; E_Fermi=0.0, band_color=:royalblue, axis=(;), ylims=nothing, kwargs...)
    return with_stage_log(
        "Plot band structure";
        context=(kpath=kpath, n_bands=size(band_matrix, 2), E_Fermi=Float64(E_Fermi)),
        summarize_result=result -> (figure_type=string(typeof(result)),),
    ) do
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
            !isnothing(ylims) && ylims!(ax, ylims...)
        end

        return fig
    end
end

_plot_band_structure(data::BandStructureData; kwargs...) =
    _plot_band_structure(data.kpath, data.bands .- data.fermi_level; E_Fermi=0.0, kwargs...)

"""
    _plot_band_comparison(d1::BandStructureData, d2::BandStructureData; 
        label1="Bands 1", label2="Bands 2", 
        color1=:royalblue, color2=(:darkorange, 0.75), 
        linewidth=2.5, legend_position=:rb, axis=(;), ylims=nothing, kwargs...)

Overlay two band structures for comparison. Internally aligns the datasets by 
rescaling k-path branches based on high-symmetry node labels. 

Keywords:
- `sync_fermi::Bool`: If true, `d2` is plotted with the Fermi level of `d1`. 
  Defaults to false (each dataset aligned to its own E_F).
"""
function _plot_band_comparison(
    d1::BandStructureData, 
    d2::BandStructureData; 
    label1::AbstractString="Bands 1", 
    label2::AbstractString="Bands 2",
    color1=:royalblue, 
    color2=(:darkorange, 0.75),
    linewidth::Real=2.5,
    legend_position=:rb,
    sync_fermi::Bool=false,
    axis=(;),
    ylims=nothing,
    kwargs...
)
    return with_stage_log(
        "Plot band structure comparison";
        context=(n_bands1=d1.num_bands, n_bands2=d2.num_bands, sync_fermi=sync_fermi),
        summarize_result=result -> (figure_type=string(typeof(result)),),
    ) do
        # 1. Extract K-Path Metadata
        dist1 = path_distances(d1.kpath)
        dist2 = path_distances(d2.kpath)
        
        node_idx1, labels1 = path_node_metadata(d1.kpath)
        node_idx2, labels2 = path_node_metadata(d2.kpath)
        
        # 2. Check Compatibility and Prep Alignment
        if labels1 != labels2
            @warn "High-symmetry node labels do not match exactly: $labels1 vs $labels2. Alignment might be incorrect."
        end
        
        # We need the same number of nodes to perform branch-wise alignment
        if length(labels1) != length(labels2)
            error("Cannot align bands: different number of high-symmetry points found ($(length(labels1)) vs $(length(labels2))).")
        end

        branch_ranges1 = path_branch_ranges(d1.kpath)
        branch_ranges2 = path_branch_ranges(d2.kpath)

        # 3. Setup Figure
        fig = Figure(size=(800, 600))
        branch_grid = fig[1, 1] = GridLayout()
        branch_axes = _band_path_axes!(
            branch_grid,
            dist1,
            node_idx1,
            labels1,
            branch_ranges1;
            ylabel="Energy E - E_F (eV)",
            title="Band Structure Comparison",
            axis=axis,
        )

        # 4. Plot Dataset 1 (Solid, Reference)
        shifted1 = d1.bands .- d1.fermi_level
        for band_idx in axes(shifted1, 2)
            for (range_idx, (ax, range)) in enumerate(zip(branch_axes, branch_ranges1))
                lines!(ax, dist1[range], shifted1[range, band_idx], 
                    color=color1, linewidth=linewidth,
                    label=(band_idx == 1 && range_idx == 1 ? label1 : nothing), kwargs...)
            end
        end

        # 5. Plot Dataset 2 (Dashed, Aligned)
        # Determine effective Fermi level for d2 based on sync_fermi setting
        effective_fermi2 = sync_fermi ? d1.fermi_level : d2.fermi_level
        shifted2 = d2.bands .- effective_fermi2

        for band_idx in axes(shifted2, 2)
            # We must iterate through branches to perform alignment
            for (branch_idx, (ax, r1, r2)) in enumerate(zip(branch_axes, branch_ranges1, branch_ranges2))
                # Identify branch boundaries in both sets
                s1, e1 = dist1[first(r1)], dist1[last(r1)]
                s2, e2 = dist2[first(r2)], dist2[last(r2)]
                
                # Rescale x-axis of d2 to match d1's branch bounds
                branch_dist2 = dist2[r2]
                rescaled_dist2 = if e2 ≈ s2
                     fill(s1, length(branch_dist2)) # Avoid division by zero
                else
                    (branch_dist2 .- s2) ./ (e2 - s2) .* (e1 - s1) .+ s1
                end

                lines!(ax, rescaled_dist2, shifted2[r2, band_idx], 
                    color=color2, linewidth=linewidth, linestyle=:dash,
                    label=(band_idx == 1 && branch_idx == 1 ? label2 : nothing), kwargs...)
            end
        end

        # 6. Finalize
        for ax in branch_axes
            hlines!(ax, [0.0], color=(:black, 0.7), linestyle=:dash, linewidth=1.2)
            !isnothing(ylims) && ylims!(ax, ylims...)
        end
        
        axislegend(first(branch_axes), position=legend_position, framevisible=false)
        return fig
    end
end

"""
    _plot_wannier90_band_structure(dir::String, prefix::String, bands_file::String = "\$(prefix)_band.dat"; kwargs...)

Parse Wannier90 `*_band.dat` output and render it with the standard band
structure plotter using QE XML metadata for labels.
"""
function _plot_wannier90_band_structure(
    dir::String,
    prefix::String,
    bands_file::String="$(prefix)_band.dat";
    kwargs...
)
    return with_stage_log(
        "Plot Wannier90 band structure";
        context=(dir=dir, prefix=prefix, bands_file=bands_file),
        summarize_result=result -> (figure_type=string(typeof(result)),),
    ) do
        data = parse_wannier90_band_dat(dir, prefix, bands_file)
        return _plot_band_structure(data; kwargs...)
    end
end

"""
    _plot_wannier90_tb_band_comparison(
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
Wannier90 `*_band.dat` energies. Internal function.
"""
function _plot_wannier90_tb_band_comparison(
    comparison::Wannier90BandComparison;
    model_label::AbstractString="TB model (shifted)",
    reference_label::AbstractString="Wannier90 band.dat",
    model_color=:royalblue,
    reference_color=(:darkorange, 0.65),
    linewidth::Real=2.5,
    legend_position=:rb,
    axis=(;),
    ylims=nothing,
)
    return with_stage_log(
        "Plot Wannier90 TB comparison";
        context=(comparison=comparison, n_bands=size(comparison.reference.bands, 2)),
        summarize_result=result -> (figure_type=string(typeof(result)),),
    ) do
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

        !isnothing(ylims) && ylims!(first(branch_axes), ylims...)
        axislegend(first(branch_axes), position=legend_position, framevisible=false)
        return fig
    end
end

"""
    _plot_fermi_surface(kxs, kys, kzs, energy_volume; E_Fermi=0.0, axis=(;), kwargs...)

Render an isosurface from a precomputed three-dimensional scalar field. Internal function.
"""
function _plot_fermi_surface(
    kxs::AbstractVector{<:Real},
    kys::AbstractVector{<:Real},
    kzs::AbstractVector{<:Real},
    energy_volume::AbstractArray{<:Real,3};
    E_Fermi=0.0,
    axis=(;),
    kwargs...
)
    return with_stage_log(
        "Plot Fermi surface";
        context=(nx=length(kxs), ny=length(kys), nz=length(kzs), E_Fermi=Float64(E_Fermi)),
        summarize_result=result -> (figure_type=string(typeof(result)),),
    ) do
        size(energy_volume) == (length(kxs), length(kys), length(kzs)) || throw(DimensionMismatch("Volume shape must match the provided axes."))

        fig = Figure(size=(900, 800), fontsize=16)
        ax = Axis3(
            fig[1, 1];
            xlabel=L"k_x",
            ylabel=L"k_y",
            zlabel=L"k_z",
            title="Interactive 3D Fermi Surface",
            elevation=π / 6,
            azimuth=π / 4,
            axis...,
        )

        E_min, E_max = minimum(energy_volume), maximum(energy_volume)
        sg = SliderGrid(
            fig[2, 1],
            (label="μ (Fermi level)", range=range(E_min, E_max, length=300), startvalue=Float64(E_Fermi)),
        )
        mu_slider = sg.sliders[1].value
        iso_level = lift(mu -> Float32[mu], mu_slider)

        contour!(
            ax,
            (first(kxs), last(kxs)),
            (first(kys), last(kys)),
            (first(kzs), last(kzs)),
            energy_volume;
            levels=iso_level,
            colormap=:viridis,
            alpha=0.5,
            transparency=true,
            kwargs...,
        )
        return fig
    end
end

_plot_fermi_surface(data::FermiSurfaceData; kwargs...) =
    _plot_fermi_surface(data.kxs, data.kys, data.kzs, data.energy_volume; kwargs...)

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
    _plot_renormalized_bands(data::RenormalizedBandData; band_limits=(-2.5, 2.5))

Plot temperature-indexed renormalized band panels. Internal function.
"""
_plot_renormalized_bands(data::RenormalizedBandData; kwargs...) =
    _plot_renormalized_bands(
        data.Ts,
        data.kpath,
        data.bare_bands,
        data.renormalized_bands,
        data.gaps;
        kwargs...
    )

# Makie Multiple Dispatch Overloads

Makie.plot(data::DispersionSurfaceData; kwargs...) = _plot_dispersion_surface(data; kwargs...)
Makie.plot(data::BandStructureData; kwargs...) = _plot_band_structure(data; kwargs...)
Makie.plot(d1::BandStructureData, d2::BandStructureData; kwargs...) = _plot_band_comparison(d1, d2; kwargs...)
Makie.plot(data::Wannier90BandComparison; kwargs...) = _plot_wannier90_tb_band_comparison(data; kwargs...)
Makie.plot(data::FermiSurfaceData; kwargs...) = _plot_fermi_surface(data; kwargs...)
Makie.plot(data::RenormalizedBandData; kwargs...) = _plot_renormalized_bands(data; kwargs...)
