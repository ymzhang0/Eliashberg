# Models/dispersions.jl

# ---------------------------------------------------------
# Electronic Dispersion Structs
# ---------------------------------------------------------

struct FreeElectron{D} <: ElectronicDispersion{D}
    EF::Float64
    mass::Float64
end
Base.show(io::IO, m::FreeElectron{D}) where {D} = print(io, "FreeElectron (", D, "D, EF=", m.EF, ")")
FreeElectron{D}(EF::Float64) where D = FreeElectron{D}(EF, 1.0) # default mass=1

# ---------------------------------------------------------
# Phonon Dispersion Structs
# ---------------------------------------------------------

struct EinsteinModel{D} <: PhononDispersion{D}
    ωE::Float64
end

struct DebyeModel{D} <: PhononDispersion{D}
    vs::Float64
    ωD::Float64
end

struct PolaritonModel{D} <: PhononDispersion{D}
    ωE::Float64  # Einstein frequency
    vs::Float64  # sound velocity
end

struct MonoatomicLatticeModel{D} <: PhononDispersion{D}
    lattice::SMatrix{D,D,Float64}
    K::Float64  # spring constant
    M::Float64  # mass
end

# Predefined models (GrapheneModel, KagomeModel, SSHModel) moved to predefined_models.jl

# Band calculation functions (compute_band_data, etc.) moved to calculator.jl
