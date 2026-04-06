# src/Visualization/responses.jl

function _plot_phase_transition(
    phis::AbstractVector{<:Real},
    Ts::AbstractVector{<:Real},
    condensation_energy::AbstractMatrix{<:Real},
    order_parameters::AbstractVector{<:Real};
    axis_left=(;),
    axis_right=(;)
)
    size(condensation_energy) == (length(phis), length(Ts)) || throw(DimensionMismatch("Condensation-energy matrix shape must be (length(phis), length(Ts))."))
    length(order_parameters) == length(Ts) || throw(DimensionMismatch("Order-parameter vector length must match the temperature axis."))

    fig = Figure(size=(1000, 500), fontsize=16)
    ax1 = Axis(fig[1, 1],
        xlabel=L"Order Parameter $\phi$",
        ylabel=L"Condensation Energy $\mathcal{F}(\phi) - \mathcal{F}(0)$",
        title="Free Energy Landscape vs T",
        axis_left...
    )
    ax2 = Axis(fig[1, 3],
        xlabel=L"Temperature $T$",
        ylabel=L"Superconducting Gap $\Delta(T)$",
        title="Order Parameter vs T",
        axis_right...
    )

    colormap_choice = :plasma
    colors = cgrad(colormap_choice, length(Ts))

    for idx in eachindex(Ts)
        lines!(ax1, phis, condensation_energy[:, idx], color=colors[idx], linewidth=2.5)
    end

    Colorbar(fig[1, 2], limits=extrema(Ts), colormap=colormap_choice, label="Temperature T")
    scatterlines!(ax2, Ts, order_parameters, color=:crimson, markersize=10, linewidth=2.5)
    hlines!(ax1, [0.0], color=:black, linestyle=:dash, linewidth=1)
    hlines!(ax2, [0.0], color=:black, linestyle=:dash, linewidth=1)
    return fig
end

"""
    plot_phase_transition(data::PhaseDiagramData; axis_left=(;), axis_right=(;))

Plot a phase-diagram response object using its stored axes and observables.
"""
plot_phase_transition(data::PhaseDiagramData; kwargs...) =
    _plot_phase_transition(data.phis, data.Ts, data.condensation_energy, data.order_parameters; kwargs...)

function plot_landscape(::Val{1}, qs::AbstractVector{<:Real}, landscape_vector::AbstractVector{<:Real}; axis=(;), kwargs...)
    length(qs) == length(landscape_vector) || throw(DimensionMismatch("Coordinate and value vectors must have the same length."))

    fig = Figure(size=(800, 500))
    ax = Axis(fig[1, 1]; xlabel=L"q", ylabel=L"\chi_0(q, \omega=0)", title="1D Instability Landscape", axis...)
    lines!(ax, qs, landscape_vector; linewidth=2, color=:crimson, kwargs...)
    hlines!(ax, [0.0], color=:black, alpha=0.3)
    return fig
end

plot_landscape(data::LandscapeLineData; kwargs...) =
    plot_landscape(Val(1), data.qs, data.values; kwargs...)

function plot_landscape(::Val{2}, qxs::AbstractVector{<:Real}, qys::AbstractVector{<:Real}, landscape_matrix::AbstractMatrix{<:Real}; axis=(;), kwargs...)
    size(landscape_matrix) == (length(qxs), length(qys)) || throw(DimensionMismatch("Landscape matrix shape must match the provided axes."))

    fig = Figure(size=(800, 600))
    ax = Axis(fig[1, 1]; xlabel=L"q_x", ylabel=L"q_y", title="Instability Landscape (Static Susceptibility)", axis...)
    hm = heatmap!(ax, qxs, qys, landscape_matrix; colormap=:magma, kwargs...)
    Colorbar(fig[1, 2], hm, label=L"\chi_0(q, \omega=0)")
    lines!(ax, [-π, π, π, -π, -π], [-π, -π, π, π, -π], color=:white, linestyle=:dash, alpha=0.5)
    return fig
end

plot_landscape(data::LandscapeSurfaceData; kwargs...) =
    plot_landscape(Val(2), data.qxs, data.qys, data.landscape_matrix; kwargs...)

function plot_spectral_function(qpath::KPath, omegas::AbstractVector{<:Real}, spectral_matrix::AbstractMatrix{<:Real}; axis=(;), kwargs...)
    size(spectral_matrix) == (length(qpath.points), length(omegas)) || throw(DimensionMismatch("Spectral matrix shape must be (length(qpath), length(omegas))."))
    distances = path_distances(qpath)
    tick_positions = distances[qpath.node_indices]
    tick_labels = qpath.node_labels

    fig = Figure(size=(900, 600))
    ax = Axis(fig[1, 1]; ylabel=L"\omega", title="Dynamic Spectral Function " * L"A(q, \omega)",
        xticks=(tick_positions, tick_labels), xgridvisible=false, axis...)

    hm = heatmap!(ax, distances, omegas, spectral_matrix; colormap=:inferno, kwargs...)
    Colorbar(fig[1, 2], hm, label=L"\text{Im}[\chi(q, \omega)]")
    vlines!(ax, tick_positions, color=:white, linestyle=:dash, linewidth=0.8, alpha=0.6)
    xlims!(ax, distances[1], distances[end])
    return fig
end

plot_spectral_function(data::SpectralMapData; kwargs...) =
    plot_spectral_function(data.qpath, data.omegas, data.spectral_matrix; kwargs...)

function plot_zeeman_pairing_landscape(
    q_vals::AbstractVector{<:Real},
    condensation_energy::AbstractVector{<:Real},
    optimal_gaps::AbstractVector{<:Real};
    minimum_index::Union{Nothing,Integer}=nothing,
    axis_left=(;),
    axis_right=(;)
)
    length(q_vals) == length(condensation_energy) == length(optimal_gaps) || throw(DimensionMismatch("All FFLO plotting arrays must have the same length."))
    highlighted_index = isnothing(minimum_index) ? argmin(condensation_energy) : minimum_index

    fig = Figure(size=(1000, 450), fontsize=16)
    ax1 = Axis(fig[1, 1],
        xlabel=L"Center-of-Mass Momentum $q_x$",
        ylabel=L"Condensation Energy $\mathcal{F}(q) - \mathcal{F}_{\mathrm{normal}}$",
        title="Magnetic Pairing Landscape",
        axis_left...
    )

    lines!(ax1, q_vals, condensation_energy, color=:royalblue, linewidth=3)
    hlines!(ax1, [0.0], color=:gray, linestyle=:dash)
    scatter!(ax1, [q_vals[highlighted_index]], [condensation_energy[highlighted_index]], color=:crimson, markersize=12, label="Global Minimum")
    axislegend(ax1, position=:lt)

    ax2 = Axis(fig[1, 2],
        xlabel=L"Center-of-Mass Momentum $q_x$",
        ylabel=L"Optimal Gap $\Delta(q)$",
        title="Order Parameter vs Momentum",
        axis_right...
    )

    lines!(ax2, q_vals, optimal_gaps, color=:crimson, linewidth=3)
    scatter!(ax2, [q_vals[highlighted_index]], [optimal_gaps[highlighted_index]], color=:royalblue, markersize=12)
    return fig
end

plot_zeeman_pairing_landscape(data::ZeemanPairingData; kwargs...) =
    plot_zeeman_pairing_landscape(
        data.q_vals,
        data.condensation_energy,
        data.optimal_gaps;
        minimum_index=data.minimum_index,
        kwargs...
    )

function _plot_collective_modes(
    qpath::KPath,
    omegas::AbstractVector{<:Real},
    spectral_matrix::AbstractMatrix{<:Real};
    pair_breaking_edge::Union{Nothing,Real}=nothing,
    axis=(;),
    colormap=:magma
)
    size(spectral_matrix) == (length(qpath.points), length(omegas)) || throw(DimensionMismatch("Spectral matrix shape must be (length(qpath), length(omegas))."))
    distances = path_distances(qpath)
    tick_positions = distances[qpath.node_indices]

    fig = Figure(size=(800, 500), fontsize=16)
    ax = Axis(fig[1, 1],
        title=L"Superconducting Excitation Spectrum $\mathrm{Im}\chi(q, \omega)$",
        xlabel="Momentum Transfer q",
        ylabel=L"Frequency $\omega$",
        xticks=(tick_positions, qpath.node_labels),
        axis...
    )

    hm = heatmap!(ax, distances, omegas, spectral_matrix, colormap=colormap)
    Colorbar(fig[1, 2], hm, label=L"Spectral Weight $\mathrm{Im}\chi$")

    if !isnothing(pair_breaking_edge)
        hlines!(ax, [pair_breaking_edge], color=:cyan, linestyle=:dash, linewidth=2, label=L"Pair-breaking edge $2\Delta_0$")
        axislegend(ax, position=:lt)
    end

    return fig
end

"""
    plot_collective_modes(data::SpectralMapData)

Plot a collective-mode spectral map from a typed response object.
"""
plot_collective_modes(data::SpectralMapData; kwargs...) =
    _plot_collective_modes(
        data.qpath,
        data.omegas,
        data.spectral_matrix;
        pair_breaking_edge=data.pair_breaking_edge,
        kwargs...
    )

visualize_phase_transition(data::PhaseDiagramData; kwargs...) =
    plot_phase_transition(data; kwargs...)
visualize_landscape(args...; kwargs...) = plot_landscape(args...; kwargs...)
visualize_spectral_function(args...; kwargs...) = plot_spectral_function(args...; kwargs...)
visualize_zeeman_pairing_landscape(args...; kwargs...) = plot_zeeman_pairing_landscape(args...; kwargs...)
visualize_collective_modes(data::SpectralMapData; kwargs...) =
    plot_collective_modes(data; kwargs...)

Makie.plot(data::LandscapeLineData; kwargs...) = plot_landscape(data; kwargs...)
Makie.plot(data::LandscapeSurfaceData; kwargs...) = plot_landscape(data; kwargs...)
Makie.plot(data::PhaseDiagramData; kwargs...) = plot_phase_transition(data; kwargs...)
Makie.plot(data::SpectralMapData; kwargs...) = plot_collective_modes(data; kwargs...)
Makie.plot(data::ZeemanPairingData; kwargs...) = plot_zeeman_pairing_landscape(data; kwargs...)
