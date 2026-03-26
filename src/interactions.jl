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

struct StaticRPAPolarization{
    M<:ElectronicDispersion, 
    S<:Smearing} <: Polarization
    dispersion::M
    smearing::S
end

struct DynamicalRPAPolarization{
    M<:ElectronicDispersion, 
    S<:Smearing} <: Polarization
    dispersion::M
    smearing::S
end

struct ScreenedCoulombInteraction{
    I<:CoulombInteraction, 
    P<:Polarization
    } <: ScreenedInteraction
    bare::I
    polarization::P
end

"""
    χ(q, kgrid, polarization::StaticRPAPolarization)

Compute the static Lindhard polarization bubble summing over band indices 
and implicitly integrating over the `AbstractKGrid{D}`.
"""
function χ(
    q::SVector{D, Float64},
    kgrid::AbstractKGrid{D}, 
    polarization::StaticRPAPolarization
    ) where {D}

    sum_val = 0.0
    for (i, k) in enumerate(kgrid)
        eig_k = band_structure(polarization.dispersion, k)
        eig_kq = band_structure(polarization.dispersion, k + q)

        Ek = eig_k.values
        Uk = eig_k.vectors
        Ekq = eig_kq.values
        Ukq = eig_kq.vectors

        # Coherence factor / Wavefunction overlap
        M_mat = Uk' * Ukq 

        for m in 1:length(Ek)
            for n in 1:length(Ekq)
                dE = Ek[m] - Ekq[n]
                overlap = abs2(M_mat[m, n])
                if abs(dE) > 1e-8
                    df = f(Ek[m], polarization.smearing) - f(Ekq[n], polarization.smearing)
                    term = -overlap * df / dE
                    sum_val += term * kgrid.weights[i]
                end
            end
        end
    end

    return sum_val / (2π)^D
end

"""
    χ(q, ω, kgrid, polarization::DynamicalRPAPolarization)

Compute the dynamical Lindhard polarization bubble.
"""
function χ(
    q::SVector{D, Float64},
    ω::Float64,
    kgrid::AbstractKGrid{D}, 
    polarization::DynamicalRPAPolarization
    ) where {D}

    sum_val = 0.0
    # small broadening or implicit limit can be added
    η = 1e-3
    for (i, k) in enumerate(kgrid)
        eig_k = band_structure(polarization.dispersion, k)
        eig_kq = band_structure(polarization.dispersion, k + q)

        Ek = eig_k.values
        Uk = eig_k.vectors
        Ekq = eig_kq.values
        Ukq = eig_kq.vectors

        M_mat = Uk' * Ukq 

        for m in 1:length(Ek)
            for n in 1:length(Ekq)
                overlap = abs2(M_mat[m, n])
                df = f(Ek[m], polarization.smearing) - f(Ekq[n], polarization.smearing)
                denom = Ekq[n] - Ek[m] - ω - im * η
                
                term = overlap * df / denom
                sum_val += real(term) * kgrid.weights[i]
            end
        end
    end

    return sum_val / (2π)^D
end

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
    V_screened = bare_V / (
        1 + bare_V * χ(q, kgrid, screened_interaction.polarization)
    )
    return V_screened
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