# Responses/fields.jl

# ----------------------------------------------------------------------------
# Auxiliary Field Concrete implementations
# ----------------------------------------------------------------------------

"""
    DirectChannel{D} <: ParticleHoleChannel{D}

代表粒子-空穴直接通道 (Particle-Hole Direct Channel)。
物理上对应电荷密度涨落 (Charge Density Fluctuations)。
由长程库仑排斥力 V(q) 驱动，
在 q=0 处最强。
"""
struct DirectChannel <: ParticleHoleChannel{Any} end

"""
    ExchangeChannel{D} <: ParticleHoleChannel{D}

代表粒子-空穴交换通道 (Particle-Hole Exchange Channel)。
对应自旋密度涨落 (Spin Density Fluctuations)。
"""
struct ExchangeChannel <: ParticleHoleChannel{Any} end
"""
    ChargeDensityWave{D} <: ParticleHoleChannel{D}

代表宏观凝聚的电荷密度波序。
(物理上它是 DirectChannel 发生相变后的产物，基矢结构完全一致)
"""
struct ChargeDensityWave{D} <: ParticleHoleChannel{D}
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
