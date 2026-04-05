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
    bare_bands::Vector{Float64}
    hole_bands::Matrix{Float64}
    particle_bands::Matrix{Float64}
    gaps::Vector{Float64}
    temperatures::Vector{Float64}

    function RenormalizedBandData{D}(
        kpath::KPath{D},
        bare_bands::Vector{Float64},
        hole_bands::Matrix{Float64},
        particle_bands::Matrix{Float64},
        gaps::Vector{Float64},
        temperatures::Vector{Float64}
    ) where {D}
        n_path = length(kpath.points)
        n_temperatures = length(temperatures)
        length(bare_bands) == n_path || throw(DimensionMismatch("Bare-band vector length must match the number of k-path samples."))
        size(hole_bands) == (n_path, n_temperatures) || throw(DimensionMismatch("Hole-band matrix shape must be (length(kpath), length(temperatures))."))
        size(particle_bands) == (n_path, n_temperatures) || throw(DimensionMismatch("Particle-band matrix shape must be (length(kpath), length(temperatures))."))
        length(gaps) == n_temperatures || throw(DimensionMismatch("Gap vector length must match the temperature axis."))
        return new{D}(kpath, bare_bands, hole_bands, particle_bands, gaps, temperatures)
    end
end
RenormalizedBandData(
    kpath::KPath{D},
    bare_bands::Vector{Float64},
    hole_bands::Matrix{Float64},
    particle_bands::Matrix{Float64},
    gaps::Vector{Float64},
    temperatures::Vector{Float64}
) where {D} = RenormalizedBandData{D}(kpath, bare_bands, hole_bands, particle_bands, gaps, temperatures)

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
