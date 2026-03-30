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

function G(k::SVector{Dim,Float64}, omega::Float64, model::FreeElectronPropagator{Dim}) where {Dim}
    eps_k = real(ε(k, model.dispersion)[1, 1])
    return 1.0 / (omega - (eps_k - model.μ))
end

function D(q::SVector{Dim,Float64}, omega::Float64, model::RetardedPhononPropagator{Dim}) where {Dim}
    dispersion_model = model.dispersion
    omega_q = ω(q, dispersion_model)
    return 1.0 / (omega^2 - omega_q^2 + model.damping^2)
end

function D(q::SVector{Dim,Float64}, omega::Float64, model::DebyePhononPropagator{Dim}) where {Dim}
    dispersion_model = model.dispersion
    omega_q = ω(q, dispersion_model)
    return 1.0 / (omega^2 - omega_q^2 + model.η + abs(omega_q))
end

G(k::Float64, omega::Float64, model::FreeElectronPropagator{1}) = G(SVector{1,Float64}(k), omega, model)
D(q::Float64, omega::Float64, model::RetardedPhononPropagator{1}) = D(SVector{1,Float64}(q), omega, model)
D(q::Float64, omega::Float64, model::DebyePhononPropagator{1}) = D(SVector{1,Float64}(q), omega, model)
