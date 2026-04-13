struct RenormalizedDispersion{D,M<:ElectronicDispersion{D},S<:SelfEnergy} <: ElectronicDispersion{D}
    bare_dispersion::M
    self_energy::S
end

struct HartreeFockSelfEnergy{Dim} <: SelfEnergy
    interaction::CoulombInteraction          # V(q)
    smearing::Smearing             # f(ε)
    dispersion::ElectronicDispersion{Dim}
end

struct RPASelfEnergy <: SelfEnergy
    U::Float64             # 
    n::Float64             # 
end


function Σ(
    k::SVector{1,Float64},
    model::HartreeFockSelfEnergy{1},
)
    dispersion_model = model.dispersion
    smearing_model = model.smearing
    interaction_model = model.interaction

    integrand(kp_scalar) = begin
        kp = SVector{1,Float64}(kp_scalar)
        eps_kp = real(ε(kp, dispersion_model)[1, 1])
        f(eps_kp, smearing_model) * V(k - kp, interaction_model)
    end

    result, _ = quadgk(integrand, -2, 2; rtol=1e-4)
    return Hermitian(hcat(-result / (2π)))
end

Σ(k::Float64, model::HartreeFockSelfEnergy{1}) = real(Σ(SVector{1,Float64}(k), model)[1, 1])

function Σ(::SVector{D,Float64}, ::HartreeFockSelfEnergy{D}) where {D}
    throw(ErrorException("HartreeFockSelfEnergy is currently implemented only for 1D momenta."))
end

Σ(::SVector{D,Float64}, model::RPASelfEnergy) where {D} = Hermitian(hcat(model.U * model.n))
Σ(::Float64, model::RPASelfEnergy) = model.U * model.n


