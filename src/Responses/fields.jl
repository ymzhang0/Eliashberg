# Responses/fields.jl

# ----------------------------------------------------------------------------
# Auxiliary Field Concrete implementations
# ----------------------------------------------------------------------------

struct ChargeDensityWave{D} <: AuxiliaryField
    q::SVector{D,Float64}
end

"""
    BCSReducedPairing <: AuxiliaryField

Represents a standard uniform superconducting condensate (q=0).
Pairs electrons with momenta (k, -k).
"""
struct BCSReducedPairing <: AuxiliaryField
    symmetry::Symbol
end
BCSReducedPairing() = BCSReducedPairing(:s_wave)

"""
    FFLOPairing{D} <: AuxiliaryField

Represents the Fulde-Ferrell (FF) state with a single center-of-mass momentum `q`.
Pairs electrons with momenta (k, -k+q).
"""
struct FFLOPairing{D} <: AuxiliaryField
    q::SVector{D,Float64}
    symmetry::Symbol
    h::Float64 # Zeeman magnetic field strength
end
FFLOPairing(q::SVector{D,Float64}, h::Float64=0.0) where {D} = FFLOPairing{D}(q, :s_wave, h)

"""
    PairDensityWave{D} <: AuxiliaryField

Represents a commensurate Pair Density Wave (LO-like state) with standing wave modulation.
Couples the electron at `k` to holes at both `-k+q` and `-k-q`.
"""
struct PairDensityWave{D} <: AuxiliaryField
    q::SVector{D,Float64}
    symmetry::Symbol
end
PairDensityWave(q::SVector{D,Float64}) where {D} = PairDensityWave{D}(q, :s_wave)

"""
    StaticMeanField{D} <: AuxiliaryField

Represents a macroscopic, frozen condensate (e.g., T=0 CDW ground state).
"""
struct StaticMeanField{D} <: AuxiliaryField
    q::SVector{D,Float64}
end
StaticMeanField(q::SVector{S,<:Real}) where S = StaticMeanField{S}(SVector{S,Float64}(q))

"""
    DynamicalFluctuation{D} <: AuxiliaryField

Represents a propagating bosonic fluctuation with momentum q and frequency ω.
"""
struct DynamicalFluctuation{D} <: AuxiliaryField
    q::SVector{D,Float64}
    ω::Float64
end
DynamicalFluctuation(q::SVector{D,<:Real}, ω::Real) where D = DynamicalFluctuation{D}(SVector{D,Float64}(q), Float64(ω))
