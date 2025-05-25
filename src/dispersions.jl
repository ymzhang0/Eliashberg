struct FreeElectron_1d <: ElectronicDispersion 
    EF::Float64
end

struct TightBinding_1d <: ElectronicDispersion
    t::Float64
    EF::Float64
end

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

struct RenormalizedDispersion{D<:ElectronicDispersion, S<:SelfEnergy} <: ElectronicDispersion
    bare_dispersion::D
    self_energy::S
end

function ε(
    k::Float64,
    model::FreeElectron_1d
    ) ::Float64
    return k^2 - model.EF
end

function ε(
    k::Float64, 
    model::TightBinding_1d
    ) ::Float64
    return -2 * model.t * cos(k) - model.EF
end

function ε(
    k::Float64, 
    model::RenormalizedDispersion
    ) ::Float64
    return ε(k, model.bare_dispersion) + Σ(k, model.self_energy)
end
    
ω(
    q::Float64, 
    model::EinsteinModel
    ) = model.ωE

ω(
    q::Float64, 
    model::DebyeModel
    ) = model.vs * abs(q)

ω(
    q::Float64, 
    d::PolaritonModel
    ) = sqrt(d.ωE^2 + (d.vs * q)^2)

ω(
    q::Float64, 
    d::MonoatomicLatticeModel
    ) = sqrt(2 * d.K / d.M * (1 - cos(d.a * q)))