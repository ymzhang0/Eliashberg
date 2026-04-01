# Models/dispersions.jl

# ---------------------------------------------------------
# Electronic Dispersion Structs
# ---------------------------------------------------------

struct FreeElectron{D} <: ElectronicDispersion{D}
    EF::Float64
    mass::Float64
end
FreeElectron{D}(EF::Float64) where D = FreeElectron{D}(EF, 1.0) # default mass=1

"""
    TightBinding{D} <: ElectronicDispersion{D}

A universal Tight-Binding model based on a real-space lattice.
"""
struct TightBinding{D} <: ElectronicDispersion{D}
    lattice::AbstractLattice{D}
    hoppings::Vector{Tuple{SVector{D,Int},Float64}}
    EF::Float64
end

"""
    MultiOrbitalTightBinding{D} <: ElectronicDispersion{D}

Multi-orbital tight-binding model defined on a `Crystal{D}` with an explicit
multi-atom basis. Each hopping is stored as
`(atom_i, atom_j, cell_offset_R, t)` and is interpreted as a unique real-space
term whose Hermitian partner is generated automatically in `ε(k)`.
"""
struct MultiOrbitalTightBinding{D} <: ElectronicDispersion{D}
    crystal::Crystal{D}
    hoppings::Vector{Tuple{Int,Int,SVector{D,Int},ComplexF64}}
    EF::Float64
end

"""
    MultiOrbitalTightBinding(crystal::Crystal{D}, hoppings, EF) where {D}

Build a `MultiOrbitalTightBinding` model directly from the internal `Crystal`
representation. All hoppings are normalized to the statically typed format
used by the evaluator. The cell offsets in `hoppings` may be given as plain
Julia vectors or tuples and are converted internally to `SVector{D,Int}`.
"""
function MultiOrbitalTightBinding(crystal::Crystal{D}, hoppings, EF) where {D}
    typed_hoppings = Tuple{Int,Int,SVector{D,Int},ComplexF64}[
        (Int(atom_i), Int(atom_j), SVector{D,Int}(cell_offset_R), ComplexF64(t))
        for (atom_i, atom_j, cell_offset_R, t) in hoppings
    ]
    return MultiOrbitalTightBinding{D}(crystal, typed_hoppings, Float64(EF))
end

"""
    MultiOrbitalTightBinding(system::AbstractSystem{D}, hoppings, EF) where {D}

Build a `MultiOrbitalTightBinding` model from an `AtomsBase.AbstractSystem` by
first converting the system into the internal `Crystal` representation.
"""
function MultiOrbitalTightBinding(system::AbstractSystem{D}, hoppings, EF) where {D}
    return MultiOrbitalTightBinding(Crystal(system), hoppings, EF)
end

"""
    KagomeLattice <: ElectronicDispersion{2}

Represents a 2D Kagome lattice with 3 sublattice sites per unit cell.
"""
struct KagomeLattice <: ElectronicDispersion{2}
    lattice::HexagonalLattice
    t::Float64   # Nearest-neighbor hopping amplitude
    EF::Float64  # Fermi energy
end

"""
    Graphene <: ElectronicDispersion{2}

Represents the 2D Honeycomb lattice of Graphene (2 sublattices).
"""
struct Graphene <: ElectronicDispersion{2}
    lattice::HexagonalLattice
    t::Float64   # Nearest-neighbor hopping
    EF::Float64
end

"""
    SSHModel <: ElectronicDispersion{1}

The 1D Su-Schrieffer-Heeger (SSH) model for polyacetylene.
"""
struct SSHModel <: ElectronicDispersion{1}
    lattice::ChainLattice
    t1::Float64  # Intra-cell hopping
    t2::Float64  # Inter-cell hopping
    EF::Float64
end

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
    lattice::AbstractLattice{D}
    K::Float64  # spring constant
    M::Float64  # mass
end

# ---------------------------------------------------------
# Unified Tight-Binding Constructors (Symbol Dispatch)
# ---------------------------------------------------------

# src/Models/dispersions.jl

# =========================================================
# Unified Tight-Binding Constructors (Concrete Type Dispatch)
# =========================================================

"""
    TightBinding(lat::AbstractLattice, args...)

Universal Tight-Binding model constructor. 
Dispatches perfectly on the concrete lattice type (e.g., `SquareLattice`, `FCCLattice`), 
automatically generating the correct hopping topology and extracting the correct dimension.
"""

# --- 1D Chain ---
function TightBinding(lat::ChainLattice, t::Float64, EF::Float64=0.0)
    hops = [(SVector{1,Int}(1), -t)]
    return TightBinding{1}(lat, hops, EF)
end

# --- 2D Square ---
function TightBinding(lat::SquareLattice, t::Float64, tp::Float64=0.0, EF::Float64=0.0)
    hops = [
        (SVector{2,Int}(1, 0), -t),
        (SVector{2,Int}(0, 1), -t)
    ]
    if abs(tp) > 1e-10
        push!(hops, (SVector{2,Int}(1, 1), -tp))
        push!(hops, (SVector{2,Int}(-1, 1), -tp))
    end
    return TightBinding{2}(lat, hops, EF)
end

# --- 2D Triangular (using HexagonalLattice basis) ---
function TightBinding(lat::HexagonalLattice, t::Float64, EF::Float64=0.0)
    hops = [
        (SVector{2,Int}(1, 0), -t),
        (SVector{2,Int}(0, 1), -t),
        (SVector{2,Int}(-1, 1), -t)
    ]
    return TightBinding{2}(lat, hops, EF)
end

# --- 3D Simple Cubic ---
function TightBinding(lat::CubicLattice, t::Float64, EF::Float64=0.0)
    hops = [
        (SVector{3,Int}(1, 0, 0), -t),
        (SVector{3,Int}(0, 1, 0), -t),
        (SVector{3,Int}(0, 0, 1), -t)
    ]
    return TightBinding{3}(lat, hops, EF)
end

# --- 3D Face-Centered Cubic (FCC) ---
function TightBinding(lat::FCCLattice, t::Float64, EF::Float64=0.0)
    hops = [
        (SVector{3,Int}(1, 0, 0), -t),
        (SVector{3,Int}(0, 1, 0), -t),
        (SVector{3,Int}(0, 0, 1), -t),
        (SVector{3,Int}(1, -1, 0), -t),
        (SVector{3,Int}(0, 1, -1), -t),
        (SVector{3,Int}(-1, 0, 1), -t)
    ]
    return TightBinding{3}(lat, hops, EF)
end

# --- 3D Body-Centered Cubic (BCC) ---
function TightBinding(lat::BCCLattice, t::Float64, EF::Float64=0.0)
    hops = [
        (SVector{3,Int}(1, 0, 0), -t),
        (SVector{3,Int}(0, 1, 0), -t),
        (SVector{3,Int}(0, 0, 1), -t),
        (SVector{3,Int}(1, 1, 1), -t)
    ]
    return TightBinding{3}(lat, hops, EF)
end

# ---------------------------------------------------------
# Multi-Orbital Legacy / Convenience Constructors
# ---------------------------------------------------------

KagomeLattice(t::Float64, EF::Float64=0.0) = KagomeLattice(HexagonalLattice(1.0), t, EF)
Graphene(t::Float64, EF::Float64=0.0) = Graphene(HexagonalLattice(1.0), t, EF)
SSHModel(t1::Float64, t2::Float64, EF::Float64=0.0) = SSHModel(ChainLattice(1.0), t1, t2, EF)

MonoatomicLatticeModel{1}(K::Float64, M::Float64, a::Float64=1.0) = MonoatomicLatticeModel{1}(ChainLattice(a), K, M)
MonoatomicLatticeModel{2}(K::Float64, M::Float64, a::Float64=1.0) = MonoatomicLatticeModel{2}(SquareLattice(a), K, M)
MonoatomicLatticeModel{3}(K::Float64, M::Float64, a::Float64=1.0) = MonoatomicLatticeModel{3}(CubicLattice(a), K, M)
