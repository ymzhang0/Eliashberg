# Models/types.jl

abstract type PhysicalModel end

# Dispersion types
abstract type Dispersion{D} <: PhysicalModel end

# ---------------------------------------------------------
# Electronic Dispersions
# ---------------------------------------------------------

abstract type ElectronicDispersion{D} <: Dispersion{D} end

struct FreeElectron{D} <: ElectronicDispersion{D}
    EF::Float64
    mass::Float64
end
FreeElectron{D}(EF::Float64) where D = FreeElectron{D}(EF, 1.0) # default mass=1

"""
    TightBinding{D} <: ElectronicDispersion{D}

A universal Tight-Binding model based on a real-space lattice.
It can be constructed either by providing explicit lattice and hopping parameters,
or by using legacy analytical parameters (e.g., t, tp) which will automatically
generate the corresponding standard lattices.
"""
struct TightBinding{D} <: ElectronicDispersion{D}
    lattice::AbstractLattice{D}
    hoppings::Vector{Tuple{SVector{D,Int},Float64}}
    EF::Float64
end

"""
    KagomeLattice <: ElectronicDispersion{2}

Represents a 2D Kagome lattice with 3 sublattice sites per unit cell.
Features a flat band, Dirac points, and van Hove singularities.
"""
struct KagomeLattice <: ElectronicDispersion{2}
    lattice::HexagonalLattice
    t::Float64   # Nearest-neighbor hopping amplitude
    EF::Float64  # Fermi energy
end

"""
    Graphene <: ElectronicDispersion{2}

Represents the 2D Honeycomb lattice of Graphene (2 sublattices).
Famous for its Dirac cones and linear dispersion.
"""
struct Graphene <: ElectronicDispersion{2}
    lattice::HexagonalLattice
    t::Float64   # Nearest-neighbor hopping
    EF::Float64
end

"""
    SSHModel <: ElectronicDispersion{1}

The 1D Su-Schrieffer-Heeger (SSH) model for polyacetylene.
A classic model for 1D topological insulators.
"""
struct SSHModel <: ElectronicDispersion{1}
    lattice::ChainLattice
    t1::Float64  # Intra-cell hopping
    t2::Float64  # Inter-cell hopping
    EF::Float64
end


# ---------------------------------------------------------
# Phonon Dispersions
# ---------------------------------------------------------
abstract type PhononDispersion{D} <: Dispersion{D} end

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
    lattice::Lattice{D}
    K::Float64  # spring constant
    M::Float64  # mass
end

# Interaction types
abstract type Interaction end
abstract type CoulombInteraction <: Interaction end
abstract type ElectronPhononInteraction <: Interaction end
abstract type ScreenedInteraction <: Interaction end

