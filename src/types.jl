using StaticArrays

# Physics pipeline abstract types
abstract type PhysicalModel end
abstract type AuxiliaryField end
abstract type ApproximationLevel end

# Concrete approximation levels
struct ExactTrLn <: ApproximationLevel end
struct RPA <: ApproximationLevel end

# Concrete Auxiliary fields
struct ChargeDensityWave{D} <: AuxiliaryField
    Q::SVector{D, Float64}
end

abstract type Dispersion <: PhysicalModel end
abstract type ElectronicDispersion <: Dispersion end
abstract type PhononDispersion     <: Dispersion end

abstract type Smearing end

abstract type Interaction end
abstract type CoulombInteraction <: Interaction end
abstract type ElectronPhononInteraction <: Interaction end
abstract type ScreenedInteraction <: Interaction end
abstract type Polarization end 

abstract type Propagator end

abstract type PhononPropagator <: Propagator end
abstract type ElectronPropagator <: Propagator end
abstract type GorkovPropagator <: Propagator end

abstract type SelfEnergy end

abstract type SpectralFunction end
abstract type ElectronSpectralFunction <: SpectralFunction end
abstract type PhononSpectralFunction <: SpectralFunction end

abstract type GapFunction end

abstract type AbstractKGrid{D} end

# Effective Action struct
struct EffectiveAction{M<:PhysicalModel, F<:AuxiliaryField, G<:AbstractKGrid}
    model::M
    field::F
    grid::G
    V_bare::Float64
end

"""
    KGrid{D} <: AbstractKGrid{D}

A concrete generic implementation of a D-dimensional K-grid.
Contains the grid `points` as `SVector{D, Float64}` and corresponding 
integration `weights`.
"""
struct KGrid{D} <: AbstractKGrid{D}
    points::Vector{SVector{D, Float64}}
    weights::Vector{Float64}
end

Base.length(g::KGrid) = length(g.points)
Base.iterate(g::KGrid, state=1) = iterate(g.points, state)
Base.eltype(::Type{KGrid{D}}) where {D} = SVector{D, Float64}
Base.getindex(g::KGrid, i::Int) = g.points[i]
Base.firstindex(g::KGrid) = 1
Base.lastindex(g::KGrid) = length(g.points)
