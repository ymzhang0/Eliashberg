using LinearAlgebra
using StaticArrays

struct FreeElectron{D} <: ElectronicDispersion 
    EF::Float64
    mass::Float64
end
FreeElectron{D}(EF::Float64) where D = FreeElectron{D}(EF, 1.0) # default mass=1

struct TightBinding{D} <: ElectronicDispersion
    t::Float64
    tp::Float64
    EF::Float64
end
TightBinding{D}(t, EF) where D = TightBinding{D}(t, 0.0, EF)

struct EinsteinModel <: PhononDispersion
    ωE::Float64
end

struct DebyeModel <: PhononDispersion
    vs::Float64
    ωD::Float64
end

struct PolaritonModel <: PhononDispersion
    ωE::Float64  # Einstein frequency
    vs::Float64  # sound velocity
end

struct MonoatomicLatticeModel <: PhononDispersion
    K::Float64  # spring constant
    M::Float64  # mass
    a::Float64  # lattice constant
end

struct RenormalizedDispersion{D, M<:ElectronicDispersion, S<:SelfEnergy} <: ElectronicDispersion
    bare_dispersion::M
    self_energy::S
end

struct MeanFieldDispersion{D, M<:ElectronicDispersion, F<:AuxiliaryField} <: ElectronicDispersion
    bare_dispersion::M
    field::F
    phi::Float64
end

function MeanFieldDispersion(bare::M, field::ChargeDensityWave{D}, phi::Real) where {D, M<:ElectronicDispersion}
    return MeanFieldDispersion{D, M, typeof(field)}(bare, field, Float64(phi))
end

# Evaluate dispersion at momentum k as a Hermitian matrix
function ε(
    k::SVector{D, Float64},
    model::FreeElectron{D}
    ) where {D}
    val = sum(abs2, k) / (2 * model.mass) - model.EF
    return Hermitian(hcat(val)) # 1x1 Hermitian matrix
end

function ε(
    k::SVector{1, Float64}, 
    model::TightBinding{1}
    )
    val = -2 * model.t * cos(k[1]) - model.EF
    return Hermitian(hcat(val))
end

function ε(
    k::SVector{2, Float64}, 
    model::TightBinding{2}
    )
    val = -2 * model.t * (cos(k[1]) + cos(k[2])) - 4 * model.tp * cos(k[1]) * cos(k[2]) - model.EF
    return Hermitian(hcat(val))
end

function ε(
    k::SVector{3, Float64}, 
    model::TightBinding{3}
    )
    val = -2 * model.t * (cos(k[1]) + cos(k[2]) + cos(k[3])) - model.EF
    return Hermitian(hcat(val))
end

function ε(
    k::SVector{D, Float64}, 
    model::RenormalizedDispersion{D}
    ) where {D}
    # Simplified placeholder for self energy renormalization
    bare_H = ε(k, model.bare_dispersion)
    Σ_H = Σ(k, model.self_energy) 
    return Hermitian(bare_H + Σ_H)
end

function ε(
    k::SVector{D, Float64},
    model::MeanFieldDispersion{D, M, <:ChargeDensityWave{D}}
    ) where {D, M}
    
    bare_disp = model.bare_dispersion
    field = model.field
    phi = model.phi

    # Original dispersion at k
    H11 = real(ε(k, bare_disp)[1, 1])
    
    # Original dispersion at k + Q
    H22 = real(ε(k + field.q, bare_disp)[1, 1])
    
    # Off-diagonal coupling
    H12 = phi
    
    # Construct 2x2 Hermitian matrix
    H = @SMatrix [H11 H12;
                  H12 H22]
                  
    return Hermitian(H)
end

# Phonon dispersions
ω(q::SVector{D, Float64}, model::EinsteinModel) where {D} = model.ωE
ω(q::SVector{D, Float64}, model::DebyeModel) where {D} = model.vs * norm(q)
ω(q::SVector{D, Float64}, d::PolaritonModel) where {D} = sqrt(d.ωE^2 + (d.vs * norm(q))^2)

function ω(q::SVector{D, Float64}, d::MonoatomicLatticeModel) where {D}
    val = sum(1 - cos(d.a * qi) for qi in q)
    return sqrt(2 * d.K / d.M * val)
end

"""
    band_structure(disp, k)

Returns an `Eigen` object containing eigenvalues and eigenvectors for the 
Hamiltonian matrix at momentum `k`.
This abstracts away the diagonalization needed for Lindhard bubble calculations 
in generalized many-body systems.
"""
function band_structure(disp::ElectronicDispersion, k::SVector{D, Float64}) where {D}
    H = ε(k, disp)
    return eigen(H)
end