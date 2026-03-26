struct DebyePhononPropagator{Dim,T<:PhononDispersion{Dim}} <: PhononPropagator
    ωD::Float64
    η::Float64
    dispersion::T
end

struct RetardedPhononPropagator{Dim,T<:PhononDispersion{Dim}} <: PhononPropagator
    ω0::Float64
    damping::Float64
    dispersion::T
end

struct FreeElectronPropagator{Dim,T<:ElectronicDispersion{Dim}} <: ElectronPropagator
    μ::Float64
    dispersion::T
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