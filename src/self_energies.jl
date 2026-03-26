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
    p::Float64, 
    model::HartreeFockSelfEnergy,
    ) ::Float64

    dispersion = model.dispersion
    smearing = model.smearing
    interaction = model.interaction

    integrand(p′) = f(ε(p′, dispersion), smearing) * V(p - p′, interaction)

    result, _ = quadgk(integrand, -2, 2; rtol=1e-4)
    return -1 / (2π) * result
end




