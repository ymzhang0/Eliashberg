struct DebyePhononPropagator{D<:PhononDispersion} <: PhononPropagator
    ωD::Float64
    η::Float64
    dispersion::D
end

struct RetardedPhononPropagator{D<:PhononDispersion} <: PhononPropagator
    ω0::Float64
    damping::Float64
    dispersion::D
end

struct FreeElectronPropagator{D<:ElectronicDispersion} <: ElectronPropagator
    μ::Float64
    dispersion::D
end

function G(q::Float64, ω::Float64, model::FreeElectronPropagator) ::Float64
    return 1.0 / (ω^2 - q^2 + model.μ)
end

function D(q::Float64, ω::Float64, model::RetardedPhononPropagator) ::Float64
    ε = model.dispersion
    return 1.0 / (ω^2 - q^2 + model.ω0^2 + model.damping^2)
end

function D(q::Float64, ω::Float64, model::DebyePhononPropagator) ::Float64
    ε = model.dispersion
    return 1.0 / (ω^2 - q^2 + model.η + abs(ε(q, ε)))  
end