export BandStructureData, PhaseDiagramData, RenormalizedBandData, SpectralMapData

Base.@kwdef struct BandStructureData{D}
    kpath::KPath{D}
    bands::Matrix{Float64}
    num_bands::Int
end

Base.@kwdef struct PhaseDiagramData
    phis::Vector{Float64}
    Ts::Vector{Float64}
    free_energy::Matrix{Float64}
    condensation_energy::Matrix{Float64}
    order_parameters::Vector{Float64}
end

Base.@kwdef struct RenormalizedBandData{D}
    kpath::KPath{D}
    bare_bands::Vector{Float64}
    hole_bands::Matrix{Float64}
    particle_bands::Matrix{Float64}
    gaps::Vector{Float64}
    temperatures::Vector{Float64}
end

Base.@kwdef struct SpectralMapData{D}
    qpath::KPath{D}
    omegas::Vector{Float64}
    spectral_matrix::Matrix{Float64}
    gap::Float64
    pair_breaking_edge::Union{Nothing,Float64}
    temperature::Float64
end
