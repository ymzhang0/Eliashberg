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
    lattice::SMatrix{D,D,Float64}
    hoppings::Vector{Tuple{SVector{D,Int},Float64}}
    EF::Float64
end

"""
    SpinorDispersion{D,M} <: ElectronicDispersion{D}

Opt-in wrapper that promotes an existing bare electronic dispersion into a
spin-degenerate basis. The wrapped model keeps its original orbital structure,
while `ε(k)` is lifted to a block-diagonal Hamiltonian with explicit spin-up and
spin-down sectors.
"""
struct SpinorDispersion{D,M<:ElectronicDispersion{D}} <: ElectronicDispersion{D}
    bare::M
end

SpinorDispersion(model::SpinorDispersion) = model

"""
    MultiOrbitalTightBinding{D} <: ElectronicDispersion{D}

Multi-orbital tight-binding model defined on a Bravais `cell`, stored
internally as a primitive-vector matrix. Each hopping is
stored as `(orbital_i, orbital_j, cell_offset_R, t)` and is interpreted in the
Wannier-style convention where the Bloch phase depends only on the lattice
translation `R`.
"""
struct MultiOrbitalTightBinding{D} <: ElectronicDispersion{D}
    cell::SMatrix{D,D,Float64}
    num_orbitals::Int
    hoppings::Vector{Tuple{Int,Int,SVector{D,Int},ComplexF64}}
    EF::Float64
end

"""
    MultiOrbitalTightBinding(cell::AbstractLattice{D}, num_orbitals, hoppings, EF) where {D}

Build a `MultiOrbitalTightBinding` model from a standalone lattice cell. All
hoppings are normalized to the statically typed format used by the evaluator.
The cell offsets in `hoppings` may be given as plain Julia vectors or tuples
and are converted internally to `SVector{D,Int}`.
"""
function MultiOrbitalTightBinding(cell::AbstractLattice{D}, num_orbitals, hoppings, EF) where {D}
    typed_hoppings = Tuple{Int,Int,SVector{D,Int},ComplexF64}[
        (Int(atom_i), Int(atom_j), SVector{D,Int}(cell_offset_R), ComplexF64(t))
        for (atom_i, atom_j, cell_offset_R, t) in hoppings
    ]
    return MultiOrbitalTightBinding{D}(primitive_vectors(cell), Int(num_orbitals), typed_hoppings, Float64(EF))
end

function MultiOrbitalTightBinding(cell::AbstractMatrix{<:Number}, num_orbitals, hoppings, EF)
    primitive_cell = primitive_vectors(cell)
    D = size(primitive_cell, 1)
    typed_hoppings = Tuple{Int,Int,SVector{D,Int},ComplexF64}[
        (Int(atom_i), Int(atom_j), SVector{D,Int}(cell_offset_R), ComplexF64(t))
        for (atom_i, atom_j, cell_offset_R, t) in hoppings
    ]
    return MultiOrbitalTightBinding{D}(primitive_cell, Int(num_orbitals), typed_hoppings, Float64(EF))
end

"""
    MultiOrbitalTightBinding(crystal::Crystal{D}, hoppings, EF) where {D}

Backward-compatible constructor that extracts only the lattice cell and orbital
count from a `Crystal`.
"""
function MultiOrbitalTightBinding(crystal::Crystal{D}, hoppings, EF) where {D}
    return MultiOrbitalTightBinding(primitive_vectors(crystal), length(crystal.atomic_symbols), hoppings, EF)
end

"""
    MultiOrbitalTightBinding(system::AbstractSystem{D}, hoppings, EF) where {D}

Build a `MultiOrbitalTightBinding` model from an `AtomsBase.AbstractSystem`.
"""
function MultiOrbitalTightBinding(system::AbstractSystem{D}, hoppings, EF) where {D}
    return MultiOrbitalTightBinding(primitive_vectors(system), length(system), hoppings, EF)
end

"""
    KagomeLattice <: ElectronicDispersion{2}

Represents a 2D Kagome lattice with 3 sublattice sites per unit cell.
"""
struct KagomeLattice <: ElectronicDispersion{2}
    lattice::SMatrix{2,2,Float64,4}
    t::Float64   # Nearest-neighbor hopping amplitude
    EF::Float64  # Fermi energy
end

"""
    Graphene <: ElectronicDispersion{2}

Represents the 2D Honeycomb lattice of Graphene (2 sublattices).
"""
struct Graphene <: ElectronicDispersion{2}
    lattice::SMatrix{2,2,Float64,4}
    t::Float64   # Nearest-neighbor hopping
    EF::Float64
end

"""
    SSHModel <: ElectronicDispersion{1}

The 1D Su-Schrieffer-Heeger (SSH) model for polyacetylene.
"""
struct SSHModel <: ElectronicDispersion{1}
    lattice::SMatrix{1,1,Float64,1}
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
    lattice::SMatrix{D,D,Float64}
    K::Float64  # spring constant
    M::Float64  # mass
end

function TightBinding(lattice::AbstractLattice{D}, hoppings, EF) where {D}
    typed_hoppings = Tuple{SVector{D,Int},Float64}[
        (SVector{D,Int}(R_idx), Float64(t_hop))
        for (R_idx, t_hop) in hoppings
    ]
    return TightBinding{D}(primitive_vectors(lattice), typed_hoppings, Float64(EF))
end

function TightBinding(lattice::AbstractMatrix{<:Number}, hoppings, EF)
    primitive_lattice = primitive_vectors(lattice)
    D = size(primitive_lattice, 1)
    typed_hoppings = Tuple{SVector{D,Int},Float64}[
        (SVector{D,Int}(R_idx), Float64(t_hop))
        for (R_idx, t_hop) in hoppings
    ]
    return TightBinding{D}(primitive_lattice, typed_hoppings, Float64(EF))
end

KagomeLattice(lattice::AbstractLattice{2}, t::Float64, EF::Float64=0.0) = KagomeLattice(primitive_vectors(lattice), t, EF)
KagomeLattice(lattice::AbstractMatrix{<:Number}, t::Float64, EF::Float64=0.0) = KagomeLattice(primitive_vectors(lattice), t, EF)
Graphene(lattice::AbstractLattice{2}, t::Float64, EF::Float64=0.0) = Graphene(primitive_vectors(lattice), t, EF)
Graphene(lattice::AbstractMatrix{<:Number}, t::Float64, EF::Float64=0.0) = Graphene(primitive_vectors(lattice), t, EF)
SSHModel(lattice::AbstractLattice{1}, t1::Float64, t2::Float64, EF::Float64=0.0) = SSHModel(primitive_vectors(lattice), t1, t2, EF)
SSHModel(lattice::AbstractMatrix{<:Number}, t1::Float64, t2::Float64, EF::Float64=0.0) = SSHModel(primitive_vectors(lattice), t1, t2, EF)
MonoatomicLatticeModel{D}(lattice::AbstractLattice{D}, K::Float64, M::Float64) where {D} = MonoatomicLatticeModel{D}(primitive_vectors(lattice), K, M)
MonoatomicLatticeModel{D}(lattice::AbstractMatrix{<:Number}, K::Float64, M::Float64) where {D} = MonoatomicLatticeModel{D}(primitive_vectors(lattice), K, M)

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
    return TightBinding(lat, hops, EF)
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
    return TightBinding(lat, hops, EF)
end

# --- 2D Triangular (using HexagonalLattice basis) ---
function TightBinding(lat::HexagonalLattice, t::Float64, EF::Float64=0.0)
    hops = [
        (SVector{2,Int}(1, 0), -t),
        (SVector{2,Int}(0, 1), -t),
        (SVector{2,Int}(-1, 1), -t)
    ]
    return TightBinding(lat, hops, EF)
end

# --- 3D Simple Cubic ---
function TightBinding(lat::CubicLattice, t::Float64, EF::Float64=0.0)
    hops = [
        (SVector{3,Int}(1, 0, 0), -t),
        (SVector{3,Int}(0, 1, 0), -t),
        (SVector{3,Int}(0, 0, 1), -t)
    ]
    return TightBinding(lat, hops, EF)
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
    return TightBinding(lat, hops, EF)
end

# --- 3D Body-Centered Cubic (BCC) ---
function TightBinding(lat::BCCLattice, t::Float64, EF::Float64=0.0)
    hops = [
        (SVector{3,Int}(1, 0, 0), -t),
        (SVector{3,Int}(0, 1, 0), -t),
        (SVector{3,Int}(0, 0, 1), -t),
        (SVector{3,Int}(1, 1, 1), -t)
    ]
    return TightBinding(lat, hops, EF)
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
