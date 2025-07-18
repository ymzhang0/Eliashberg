
struct ConstantInteraction_1d <: ElectronPhononInteraction
    V0::Float64
end

struct LocalInteraction_1d <: ElectronPhononInteraction
    V0::Float64
    threshold::Float64
    fsthick::Float64
end

struct YukawaInteraction_1d <: ElectronPhononInteraction
    V0::Float64
    λ::Float64
end

struct LimitedConstantInteraction{D<:ElectronicDispersion} <: ElectronPhononInteraction
    V0::Float64
    ωc::Float64
    dispersion::D  
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
    D<:ElectronicDispersion, 
    S<:Smearing} <: Polarization
    dispersion::D
    smearing::S
end

struct DynamicalRPAPolarization{
    D<:ElectronicDispersion, 
    S<:Smearing} <: Polarization
    dispersion::D
    smearing::S
end

struct ScreenedCoulombInteraction{
    I<:CoulombInteraction, 
    P<:Polarization
    } <: ScreenedInteraction
    bare::I
    polarization::P
end

function χ(
    q::Float64,
    kgrid::AbstractVector{Float64}, 
    polarization::StaticRPAPolarization
    ) ::Float64

    integrand(k) = begin
        εk = ε(k, polarization.dispersion)
        εkq = ε(k+q, polarization.dispersion)
        dε = εk - εkq
        -(
            f(εk, polarization.smearing) - f(εkq, polarization.smearing)
        ) / dε
    end

    integrand_vals = integrand.(kgrid)
    dk = mean(diff(kgrid))  # 均匀格点可简化

    return sum(integrand_vals) * dk / (2π)
end

function χ(
    q::Float64,
    ω::Float64,
    kgrid::AbstractVector{Float64}, 
    polarization::DynamicalRPAPolarization
    ) ::Float64

    integrand(k) = begin
        εk = ε(k, polarization.dispersion)
        εkq = ε(k+q, polarization.dispersion)
        (
            f(εk, polarization.smearing) - f(εkq, polarization.smearing)
        ) / (εkq - εk - ω)
    end

    integrand_vals = integrand.(kgrid)
    dk = mean(diff(kgrid))  # 均匀格点可简化

    return sum(integrand_vals) * dk / (2π)
end

function V(
    interaction::ConstantInteraction_1d, 
    ) ::Float64
    return interaction.V0
end

function V(
    interaction::LocalInteraction_1d, 
    ) ::Float64
    if abs(ε(k, dispersion_model)) ≤ interaction.fsthick && abs(ε(kp, dispersion_model)) ≤ interaction.fsthick
        return isapprox(kp, -k; atol=interaction.threshold) ? interaction.V0 : 0.0
    else
        return 0.0
    end
end

function V(
    k::Float64, 
    kp::Float64, 
    interaction::YukawaInteraction_1d, 
    ) ::Float64
    return interaction.V0 / ((k - kp)^2 + interaction.λ^2)
end

function V(
    k::Float64, 
    kp::Float64,
    interaction::LimitedConstantInteraction,
    dispersion_model::ElectronicDispersion
    ) ::Float64
    if abs(ε(k, dispersion_model)) ≤ interaction.ωc && abs(ε(kp, dispersion_model)) ≤ interaction.ωc
        return interaction.V0
    else
        return 0.0
    end
end



function V(
    q::Float64, 
    interaction::BareCoulombInteraction
    ) ::Float64
    return abs(q) < interaction.cutoff ? 0.0 : 1/ ( q)^2
end

function V(
    q::Float64,
    kgrid::AbstractVector{Float64},
    screened_interaction::ScreenedCoulombInteraction
    ) ::Float64

    V_screened = V(q, screened_interaction.bare) / (
        1 + V(q, screened_interaction.bare) * χ(q, kgrid, screened_interaction.polarization)
    )
    return V_screened
end

function V(
    q::Float64,
    kgrid::AbstractVector{Float64},
    interaction::FanMigdalInteraction
    ) ::Float64

    V_screened = V(q, screened_interaction.bare) / (
        1 + V(q, screened_interaction.bare) * χ(q, kgrid, screened_interaction.polarization)
    )
    return V_screened
end

function Σ(
    k::Float64,
    ω::Float64,
    qgrid::AbstractVector{Float64},
    interaction::FanMigdalInteraction,
    )

    integrand(q) = begin
        kq = k + q
        ε_kq = ε(kq, dispersion)
        ω_q = ω_phonon(q)

        f_val = f(ε_kq, μ, T)
        n_val = n(ω_q, T)
        g_val = g2(k, q)

        denom1 = ω - ε_kq - ω_q + im*η
        denom2 = ω - ε_kq + ω_q + im*η

        term1 = (n_val + f_val) / denom1
        term2 = (n_val + 1 - f_val) / denom2

        return g_val * (term1 + term2)
    end

    result, _ = quadgk(integrand, first(qgrid), last(qgrid); rtol=1e-4)
    return result / (2π)
end