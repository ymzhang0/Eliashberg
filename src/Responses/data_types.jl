export BandStructureData, DispersionSurfaceData, FermiSurfaceData, LandscapeLineData, LandscapeSurfaceData
export PhaseDiagramData, RenormalizedBandData, SpectralMapData, ZeemanPairingData

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

Base.@kwdef struct DispersionSurfaceData
    kxs::Vector{Float64}
    kys::Vector{Float64}
    energy_matrix::Matrix{Float64}

    function DispersionSurfaceData(
        kxs::AbstractVector{<:Real},
        kys::AbstractVector{<:Real},
        energy_matrix::AbstractMatrix{<:Real}
    )
        size(energy_matrix) == (length(kxs), length(kys)) || throw(DimensionMismatch("Energy matrix shape must match the provided axes."))
        return new(collect(Float64.(kxs)), collect(Float64.(kys)), Float64.(energy_matrix))
    end
end

Base.@kwdef struct FermiSurfaceData
    kxs::Vector{Float64}
    kys::Vector{Float64}
    kzs::Vector{Float64}
    energy_volume::Array{Float32,3}

    function FermiSurfaceData(
        kxs::AbstractVector{<:Real},
        kys::AbstractVector{<:Real},
        kzs::AbstractVector{<:Real},
        energy_volume::AbstractArray{<:Real,3}
    )
        size(energy_volume) == (length(kxs), length(kys), length(kzs)) || throw(DimensionMismatch("Volume shape must match the provided axes."))
        return new(
            collect(Float64.(kxs)),
            collect(Float64.(kys)),
            collect(Float64.(kzs)),
            Float32.(energy_volume)
        )
    end
end

Base.@kwdef struct LandscapeLineData
    qs::Vector{Float64}
    values::Vector{Float64}

    function LandscapeLineData(qs::AbstractVector{<:Real}, values::AbstractVector{<:Real})
        length(qs) == length(values) || throw(DimensionMismatch("Coordinate and value vectors must have the same length."))
        return new(collect(Float64.(qs)), collect(Float64.(values)))
    end
end

Base.@kwdef struct LandscapeSurfaceData
    qxs::Vector{Float64}
    qys::Vector{Float64}
    landscape_matrix::Matrix{Float64}

    function LandscapeSurfaceData(
        qxs::AbstractVector{<:Real},
        qys::AbstractVector{<:Real},
        landscape_matrix::AbstractMatrix{<:Real}
    )
        size(landscape_matrix) == (length(qxs), length(qys)) || throw(DimensionMismatch("Landscape matrix shape must match the provided axes."))
        return new(
            collect(Float64.(qxs)),
            collect(Float64.(qys)),
            Float64.(landscape_matrix)
        )
    end
end

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

Base.@kwdef struct ZeemanPairingData
    q_vals::Vector{Float64}
    condensation_energy::Vector{Float64}
    optimal_gaps::Vector{Float64}
    optimal_q::Float64
    minimum_index::Int

    function ZeemanPairingData(
        q_vals::AbstractVector{<:Real},
        condensation_energy::AbstractVector{<:Real},
        optimal_gaps::AbstractVector{<:Real},
        optimal_q::Real,
        minimum_index::Integer
    )
        length(q_vals) == length(condensation_energy) == length(optimal_gaps) || throw(DimensionMismatch("All FFLO data vectors must have the same length."))
        1 <= minimum_index <= length(q_vals) || throw(BoundsError(q_vals, minimum_index))
        return new(
            collect(Float64.(q_vals)),
            collect(Float64.(condensation_energy)),
            collect(Float64.(optimal_gaps)),
            Float64(optimal_q),
            Int(minimum_index)
        )
    end
end
