export BandStructureData, PhaseDiagramData, RenormalizedBandData, SpectralMapData

Base.@kwdef struct BandStructureData{D}
    kpath::KPath{D}
    bands::Matrix{Float64}
    num_bands::Int

    function BandStructureData{D}(kpath::KPath{D}, bands::Matrix{Float64}, num_bands::Int) where {D}
        size(bands, 1) == length(kpath.points) || throw(DimensionMismatch("Band matrix row count must match the number of k-path samples."))
        size(bands, 2) == num_bands || throw(DimensionMismatch("Band matrix column count must match `num_bands`."))
        return new{D}(kpath, bands, num_bands)
    end
end
BandStructureData(kpath::KPath{D}, bands::Matrix{Float64}, num_bands::Int) where {D} = BandStructureData{D}(kpath, bands, num_bands)

Base.@kwdef struct PhaseDiagramData
    phis::Vector{Float64}
    Ts::Vector{Float64}
    free_energy::Matrix{Float64}
    condensation_energy::Matrix{Float64}
    order_parameters::Vector{Float64}

    function PhaseDiagramData(
        phis::Vector{Float64},
        Ts::Vector{Float64},
        free_energy::Matrix{Float64},
        condensation_energy::Matrix{Float64},
        order_parameters::Vector{Float64}
    )
        size(free_energy) == (length(phis), length(Ts)) || throw(DimensionMismatch("Free-energy matrix shape must be (length(phis), length(Ts))."))
        size(condensation_energy) == (length(phis), length(Ts)) || throw(DimensionMismatch("Condensation-energy matrix shape must be (length(phis), length(Ts))."))
        length(order_parameters) == length(Ts) || throw(DimensionMismatch("Order-parameter vector length must match the temperature axis."))
        return new(phis, Ts, free_energy, condensation_energy, order_parameters)
    end
end

Base.@kwdef struct RenormalizedBandData{D}
    kpath::KPath{D}
    bare_bands::Matrix{Float64}
    renormalized_bands::Array{Float64,3}
    gaps::Vector{Float64}
    temperatures::Vector{Float64}

    function RenormalizedBandData{D}(
        kpath::KPath{D},
        bare_bands::Matrix{Float64},
        renormalized_bands::Array{Float64,3},
        gaps::Vector{Float64},
        temperatures::Vector{Float64}
    ) where {D}
        n_path = length(kpath.points)
        n_temperatures = length(temperatures)
        size(bare_bands, 1) == n_path || throw(DimensionMismatch("Bare-band matrix row count must match the number of k-path samples."))
        size(renormalized_bands, 1) == n_path || throw(DimensionMismatch("Renormalized-band tensor first dimension must match the number of k-path samples."))
        size(renormalized_bands, 3) == n_temperatures || throw(DimensionMismatch("Renormalized-band tensor third dimension must match the temperature axis length."))
        length(gaps) == n_temperatures || throw(DimensionMismatch("Gap vector length must match the temperature axis."))
        return new{D}(kpath, bare_bands, renormalized_bands, gaps, temperatures)
    end
end
RenormalizedBandData(
    kpath::KPath{D},
    bare_bands::Matrix{Float64},
    renormalized_bands::Array{Float64,3},
    gaps::Vector{Float64},
    temperatures::Vector{Float64}
) where {D} = RenormalizedBandData{D}(kpath, bare_bands, renormalized_bands, gaps, temperatures)

Base.@kwdef struct SpectralMapData{D}
    qpath::KPath{D}
    omegas::Vector{Float64}
    spectral_matrix::Matrix{Float64}
    gap::Float64
    pair_breaking_edge::Union{Nothing,Float64}
    temperature::Float64

    function SpectralMapData{D}(
        qpath::KPath{D},
        omegas::Vector{Float64},
        spectral_matrix::Matrix{Float64},
        gap::Float64,
        pair_breaking_edge::Union{Nothing,Float64},
        temperature::Float64
    ) where {D}
        size(spectral_matrix) == (length(qpath.points), length(omegas)) || throw(DimensionMismatch("Spectral matrix shape must be (length(qpath), length(omegas))."))
        return new{D}(qpath, omegas, spectral_matrix, gap, pair_breaking_edge, temperature)
    end
end
SpectralMapData(
    qpath::KPath{D},
    omegas::Vector{Float64},
    spectral_matrix::Matrix{Float64},
    gap::Float64,
    pair_breaking_edge::Union{Nothing,Float64},
    temperature::Float64
) where {D} = SpectralMapData{D}(qpath, omegas, spectral_matrix, gap, pair_breaking_edge, temperature)
