using LinearAlgebra
using StaticArrays
using QuadGK

struct ConstantInteraction{D} <: ElectronPhononInteraction
    V0::Float64
end

struct LocalInteraction{D} <: ElectronPhononInteraction
    V0::Float64
    threshold::Float64
    fsthick::Float64
end

struct YukawaInteraction{D} <: ElectronPhononInteraction
    V0::Float64
    λ::Float64
end

struct LimitedConstantInteraction{D, M<:ElectronicDispersion} <: ElectronPhononInteraction
    V0::Float64
    ωc::Float64
    dispersion::M  
end

struct FanMigdalInteraction{
    G<:ElectronPropagator, 
    D<:PhononPropagator
    } <: ElectronPhononInteraction
    electron::G
    phonon::D
end

struct BareCoulombInteraction <: CoulombInteraction
    cutoff::Float64
end

# Legacy polarization types removed. LindhardSusceptibility is the new source of truth.

struct ScreenedCoulombInteraction{
    I<:CoulombInteraction, 
    P<:Polarization
    } <: ScreenedInteraction
    bare::I
    polarization::P
end

# Legacy chi methods removed.

function V(
    interaction::ConstantInteraction{D}, 
    ) where {D}
    return interaction.V0
end

function V(
    k::SVector{D, Float64},
    kp::SVector{D, Float64},
    interaction::LocalInteraction{D}, 
    dispersion_model::ElectronicDispersion
    ) where {D}
    # Uses band structure energies nearest to Fermi surface
    Ek = band_structure(dispersion_model, k).values
    Ekp = band_structure(dispersion_model, kp).values
    
    # Check if any band is within fsthick
    if any(abs.(Ek) .≤ interaction.fsthick) && any(abs.(Ekp) .≤ interaction.fsthick)
        return norm(kp + k) < interaction.threshold ? interaction.V0 : 0.0
    else
        return 0.0
    end
end

function V(
    k::SVector{D, Float64}, 
    kp::SVector{D, Float64}, 
    interaction::YukawaInteraction{D}, 
    ) where {D}
    return interaction.V0 / (norm(k - kp)^2 + interaction.λ^2)
end

function V(
    k::SVector{D, Float64}, 
    kp::SVector{D, Float64},
    interaction::LimitedConstantInteraction{D},
    dispersion_model::ElectronicDispersion
    ) where {D}
    Ek = band_structure(dispersion_model, k).values
    Ekp = band_structure(dispersion_model, kp).values
    if any(abs.(Ek) .≤ interaction.ωc) && any(abs.(Ekp) .≤ interaction.ωc)
        return interaction.V0
    else
        return 0.0
    end
end

function V(
    q::SVector{D, Float64}, 
    interaction::BareCoulombInteraction
    ) where {D}
    q_norm = norm(q)
    return q_norm < interaction.cutoff ? 0.0 : 1.0 / (q_norm^2)
end

function V(
    q::SVector{D, Float64},
    kgrid::AbstractKGrid{D},
    screened_interaction::ScreenedCoulombInteraction
    ) where {D}

    bare_V = V(q, screened_interaction.bare)
    # The legacy χ call is removed. P should now be a LindhardSusceptibility-like functor.
    chi_val = screened_interaction.polarization(q)
    return bare_V / (1 + bare_V * chi_val)
end

# FanMigdal placeholder implementations 
function V(
    q::SVector{D, Float64},
    kgrid::AbstractKGrid{D},
    interaction::FanMigdalInteraction
    ) where {D}
    # Placeholder screened interaction for FanMigdal
    return 0.0 
end

function Σ(
    k::SVector{D, Float64},
    ω::Float64,
    qgrid::AbstractKGrid{D},
    interaction::FanMigdalInteraction{G, Ph},
    ) where {D, G, Ph}
    # Placeholder generic integral over qgrid
    # Needs g2 matrix element integration correctly adapted for D-dimensions
    return 0.0
end