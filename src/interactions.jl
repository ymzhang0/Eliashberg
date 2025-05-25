
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

struct BareCoulombInteraction <: CoulombInteraction
    cutoff::Float64
end

struct ScreenedCoulombInteraction <: CoulombInteraction
    λ::Float64
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
    return abs(q) < interaction.cutoff ? 0.0 : Constants.e^2 / ( q)^2 / (4π * Constants.ε0)
end

function V(
    q::Float64, 
    interaction::ScreenedCoulombInteraction
    ) ::Float64
    return 1 / (q^2 + interaction.λ^2)
end