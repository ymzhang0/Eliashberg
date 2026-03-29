# Auxiliary field types
abstract type AuxiliaryField end


# Specific Auxiliary Field Concrete implementations
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
    h::Float64 # 新增：Zeeman 磁场强度
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

# ----------------------------------------------------------------------------
# Mean-Field Dispersion Structs & Constructors
# ----------------------------------------------------------------------------

struct MeanFieldDispersion{D,M<:ElectronicDispersion{D},F<:AuxiliaryField} <: ElectronicDispersion{D}
    bare_dispersion::M
    field::F
    phi::Float64
end

# Concrete Auxiliary fields
# Represents a macroscopic, frozen condensate (e.g., T=0 CDW ground state)
struct StaticMeanField{D} <: AuxiliaryField
    q::SVector{D,Float64}
end
StaticMeanField(q::SVector{S,<:Real}) where S = StaticMeanField{S}(SVector{S,Float64}(q))

# Represents a propagating bosonic fluctuation with momentum q and frequency ω
struct DynamicalFluctuation{D} <: AuxiliaryField
    q::SVector{D,Float64}
    ω::Float64
end
DynamicalFluctuation(q::SVector{D,<:Real}, ω::Real) where D = DynamicalFluctuation{D}(SVector{D,Float64}(q), Float64(ω))

# Normal Nambu Space Basis Promotion (For standard BCS)
struct NormalNambuDispersion{D,M<:ElectronicDispersion{D}} <: ElectronicDispersion{D}
    bare::M
end

normal_state_basis(model::ElectronicDispersion{D}, field::AuxiliaryField) where {D} = model
normal_state_basis(model::ElectronicDispersion{D}, field::BCSReducedPairing) where {D} = NormalNambuDispersion{D,typeof(model)}(model)
normal_state_basis(model::ElectronicDispersion{D}, field::FFLOPairing{D}) where {D} = FFLONormalDispersion{D,typeof(model)}(model, field.q)
normal_state_basis(model::ElectronicDispersion{D}, field::PairDensityWave{D}) where {D} = PDWNormalDispersion{D,typeof(model)}(model, field.q)

# Propagator types
abstract type Propagator end
abstract type PhononPropagator <: Propagator end
abstract type ElectronPropagator <: Propagator end
abstract type GorkovPropagator <: Propagator end

# Many-body types
abstract type SelfEnergy end

struct RenormalizedDispersion{D,M<:ElectronicDispersion{D},S<:SelfEnergy} <: ElectronicDispersion{D}
    bare_dispersion::M
    self_energy::S
end

abstract type Smearing end
abstract type Polarization end
abstract type SpectralFunction end
abstract type ElectronSpectralFunction <: SpectralFunction end
abstract type PhononSpectralFunction <: SpectralFunction end
abstract type GapFunction end
