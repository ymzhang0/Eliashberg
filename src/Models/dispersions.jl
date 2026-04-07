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
    lattice::SMatrix{D,D,Float64}
    periodicity::NTuple{D,Bool}
    num_orbitals::Int
    hoppings::Vector{Tuple{Int,Int,SVector{D,Int},ComplexF64}}
    EF::Float64
end

primitive_vectors(model::MultiOrbitalTightBinding) = getfield(model, :lattice)
periodicity(model::MultiOrbitalTightBinding) = getfield(model, :periodicity)

function periodic_cell(model::MultiOrbitalTightBinding{D}; length_unit=u"Å") where {D}
    return PeriodicCell(primitive_vectors(model); periodicity=periodicity(model), length_unit)
end

function Base.getproperty(model::MultiOrbitalTightBinding, name::Symbol)
    if name === :cell
        return periodic_cell(model)
    elseif name === :lattice
        return getfield(model, :lattice)
    end
    return getfield(model, name)
end

function MultiOrbitalTightBinding(cell::AbstractMatrix{<:Number}, num_orbitals, hoppings, EF)
    primitive_cell = primitive_vectors(cell)
    D = size(primitive_cell, 1)
    typed_hoppings = Tuple{Int,Int,SVector{D,Int},ComplexF64}[
        (Int(atom_i), Int(atom_j), SVector{D,Int}(cell_offset_R), ComplexF64(t))
        for (atom_i, atom_j, cell_offset_R, t) in hoppings
    ]
    return MultiOrbitalTightBinding{D}(primitive_cell, ntuple(_ -> true, D), Int(num_orbitals), typed_hoppings, Float64(EF))
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
    return MultiOrbitalTightBinding(system, length(system), hoppings, EF)
end

function MultiOrbitalTightBinding(cell::PeriodicCell{D}, num_orbitals, hoppings, EF) where {D}
    primitive_cell = primitive_vectors(cell)
    typed_hoppings = Tuple{Int,Int,SVector{D,Int},ComplexF64}[
        (Int(atom_i), Int(atom_j), SVector{D,Int}(cell_offset_R), ComplexF64(t))
        for (atom_i, atom_j, cell_offset_R, t) in hoppings
    ]
    return MultiOrbitalTightBinding{D}(primitive_cell, periodicity(cell), Int(num_orbitals), typed_hoppings, Float64(EF))
end

function MultiOrbitalTightBinding(system::AbstractSystem{D}, num_orbitals, hoppings, EF) where {D}
    primitive_cell = primitive_vectors(system)
    typed_hoppings = Tuple{Int,Int,SVector{D,Int},ComplexF64}[
        (Int(atom_i), Int(atom_j), SVector{D,Int}(cell_offset_R), ComplexF64(t))
        for (atom_i, atom_j, cell_offset_R, t) in hoppings
    ]
    return MultiOrbitalTightBinding{D}(primitive_cell, periodicity(system), Int(num_orbitals), typed_hoppings, Float64(EF))
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

function _is_square_cell(cell_like; atol=5e-3, rtol=1e-5)
    vectors = primitive_vectors(cell_like)
    size(vectors) == (2, 2) || return false
    a1 = vectors[:, 1]
    a2 = vectors[:, 2]
    return isapprox(norm(a1), norm(a2); rtol=rtol) && isapprox(dot(a1, a2), 0.0; atol=atol)
end

function _is_hexagonal_cell(cell_like; atol=5e-3, rtol=1e-5)
    vectors = primitive_vectors(cell_like)
    size(vectors) == (2, 2) || return false
    a1 = vectors[:, 1]
    a2 = vectors[:, 2]
    cosθ = dot(a1, a2) / (norm(a1) * norm(a2))
    return isapprox(norm(a1), norm(a2); rtol=rtol) && isapprox(cosθ, 0.5; atol=atol)
end

function _tight_binding_chain(cell_like, t::Float64, EF::Float64)
    hops = [(SVector{1,Int}(1), -t)]
    return TightBinding(primitive_vectors(cell_like), hops, EF)
end

function _tight_binding_square(cell_like, t::Float64, tp::Float64, EF::Float64)
    hops = [
        (SVector{2,Int}(1, 0), -t),
        (SVector{2,Int}(0, 1), -t),
    ]
    if abs(tp) > 1e-10
        push!(hops, (SVector{2,Int}(1, 1), -tp))
        push!(hops, (SVector{2,Int}(-1, 1), -tp))
    end
    return TightBinding(primitive_vectors(cell_like), hops, EF)
end

function _tight_binding_triangular(cell_like, t::Float64, EF::Float64)
    hops = [
        (SVector{2,Int}(1, 0), -t),
        (SVector{2,Int}(0, 1), -t),
        (SVector{2,Int}(-1, 1), -t),
    ]
    return TightBinding(primitive_vectors(cell_like), hops, EF)
end

function _tight_binding_cubic(cell_like, t::Float64, EF::Float64)
    bravais = bravais_lattice(cell_like)
    hops = if bravais == :cP
        [
            (SVector{3,Int}(1, 0, 0), -t),
            (SVector{3,Int}(0, 1, 0), -t),
            (SVector{3,Int}(0, 0, 1), -t),
        ]
    elseif bravais == :cF
        [
            (SVector{3,Int}(1, 0, 0), -t),
            (SVector{3,Int}(0, 1, 0), -t),
            (SVector{3,Int}(0, 0, 1), -t),
            (SVector{3,Int}(1, -1, 0), -t),
            (SVector{3,Int}(0, 1, -1), -t),
            (SVector{3,Int}(-1, 0, 1), -t),
        ]
    elseif bravais == :cI
        [
            (SVector{3,Int}(1, 0, 0), -t),
            (SVector{3,Int}(0, 1, 0), -t),
            (SVector{3,Int}(0, 0, 1), -t),
            (SVector{3,Int}(1, 1, 1), -t),
        ]
    else
        throw(ArgumentError("Only cubic primitive (`:cP`), face-centered (`:cF`), and body-centered (`:cI`) cells are supported by the scalar TightBinding convenience constructor."))
    end
    return TightBinding(primitive_vectors(cell_like), hops, EF)
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

function TightBinding(cell::PeriodicCell{D}, hoppings, EF) where {D}
    return TightBinding(primitive_vectors(cell), hoppings, EF)
end

function TightBinding(system::AbstractSystem{D}, hoppings, EF) where {D}
    return TightBinding(primitive_vectors(system), hoppings, EF)
end

TightBinding(cell::PeriodicCell{1}, t::Float64, EF::Float64=0.0) = _tight_binding_chain(cell, t, EF)
TightBinding(cell::PeriodicCell{2}, t::Float64, x::Float64=0.0) = _is_hexagonal_cell(cell) ? _tight_binding_triangular(cell, t, x) : _tight_binding_square(cell, t, x, 0.0)
function TightBinding(cell::PeriodicCell{2}, t::Float64, tp::Float64, EF::Float64)
    _is_hexagonal_cell(cell) && throw(ArgumentError("Hexagonal/triangular cells only support `TightBinding(cell, t, EF)` in the convenience constructor."))
    return _tight_binding_square(cell, t, tp, EF)
end
TightBinding(cell::PeriodicCell{3}, t::Float64, EF::Float64=0.0) = _tight_binding_cubic(cell, t, EF)
TightBinding(system::AbstractSystem{1}, t::Float64, EF::Float64=0.0) = _tight_binding_chain(system, t, EF)
TightBinding(system::AbstractSystem{2}, t::Float64, x::Float64=0.0) = _is_hexagonal_cell(system) ? _tight_binding_triangular(system, t, x) : _tight_binding_square(system, t, x, 0.0)
function TightBinding(system::AbstractSystem{2}, t::Float64, tp::Float64, EF::Float64)
    _is_hexagonal_cell(system) && throw(ArgumentError("Hexagonal/triangular cells only support `TightBinding(system, t, EF)` in the convenience constructor."))
    return _tight_binding_square(system, t, tp, EF)
end
TightBinding(system::AbstractSystem{3}, t::Float64, EF::Float64=0.0) = _tight_binding_cubic(system, t, EF)

function TightBinding(lattice::AbstractMatrix{<:Number}, t::Float64, x::Float64=0.0)
    primitive = primitive_vectors(lattice)
    D = size(primitive, 1)
    if D == 1
        return _tight_binding_chain(primitive, t, x)
    elseif D == 2
        return _is_hexagonal_cell(primitive) ? _tight_binding_triangular(primitive, t, x) : _tight_binding_square(primitive, t, x, 0.0)
    elseif D == 3
        return _tight_binding_cubic(primitive, t, x)
    end
    throw(ArgumentError("Only 1D, 2D, and 3D cells are supported by the scalar TightBinding convenience constructor."))
end

function TightBinding(lattice::AbstractMatrix{<:Number}, t::Float64, tp::Float64, EF::Float64)
    primitive = primitive_vectors(lattice)
    size(primitive) == (2, 2) || throw(ArgumentError("The four-argument scalar TightBinding convenience constructor is only defined for square-like 2D cells."))
    _is_hexagonal_cell(primitive) && throw(ArgumentError("Hexagonal/triangular cells only support `TightBinding(cell, t, EF)` in the convenience constructor."))
    return _tight_binding_square(primitive, t, tp, EF)
end

KagomeLattice(lattice::AbstractMatrix{<:Number}, t::Float64, EF::Float64=0.0) = KagomeLattice(primitive_vectors(lattice), t, EF)
KagomeLattice(cell::PeriodicCell{2}, t::Float64, EF::Float64=0.0) = KagomeLattice(primitive_vectors(cell), t, EF)
KagomeLattice(system::AbstractSystem{2}, t::Float64, EF::Float64=0.0) = KagomeLattice(primitive_vectors(system), t, EF)
Graphene(lattice::AbstractMatrix{<:Number}, t::Float64, EF::Float64=0.0) = Graphene(primitive_vectors(lattice), t, EF)
Graphene(cell::PeriodicCell{2}, t::Float64, EF::Float64=0.0) = Graphene(primitive_vectors(cell), t, EF)
Graphene(system::AbstractSystem{2}, t::Float64, EF::Float64=0.0) = Graphene(primitive_vectors(system), t, EF)
SSHModel(lattice::AbstractMatrix{<:Number}, t1::Float64, t2::Float64, EF::Float64=0.0) = SSHModel(primitive_vectors(lattice), t1, t2, EF)
SSHModel(cell::PeriodicCell{1}, t1::Float64, t2::Float64, EF::Float64=0.0) = SSHModel(primitive_vectors(cell), t1, t2, EF)
SSHModel(system::AbstractSystem{1}, t1::Float64, t2::Float64, EF::Float64=0.0) = SSHModel(primitive_vectors(system), t1, t2, EF)
MonoatomicLatticeModel{D}(lattice::AbstractMatrix{<:Number}, K::Float64, M::Float64) where {D} = MonoatomicLatticeModel{D}(primitive_vectors(lattice), K, M)
MonoatomicLatticeModel{D}(cell::PeriodicCell{D}, K::Float64, M::Float64) where {D} = MonoatomicLatticeModel{D}(primitive_vectors(cell), K, M)
MonoatomicLatticeModel{D}(system::AbstractSystem{D}, K::Float64, M::Float64) where {D} = MonoatomicLatticeModel{D}(primitive_vectors(system), K, M)

# ---------------------------------------------------------
# Unified Tight-Binding Constructors (Symbol Dispatch)
# ---------------------------------------------------------

# src/Models/dispersions.jl

# =========================================================
# Unified Tight-Binding Constructors (Concrete Type Dispatch)
# =========================================================

# ---------------------------------------------------------
# Multi-Orbital Legacy / Convenience Constructors
# ---------------------------------------------------------

KagomeLattice(t::Float64, EF::Float64=0.0) = KagomeLattice(HexagonalLattice(1.0), t, EF)
Graphene(t::Float64, EF::Float64=0.0) = Graphene(HexagonalLattice(1.0), t, EF)
SSHModel(t1::Float64, t2::Float64, EF::Float64=0.0) = SSHModel(ChainLattice(1.0), t1, t2, EF)

MonoatomicLatticeModel{1}(K::Float64, M::Float64, a::Float64=1.0) = MonoatomicLatticeModel{1}(ChainLattice(a), K, M)
MonoatomicLatticeModel{2}(K::Float64, M::Float64, a::Float64=1.0) = MonoatomicLatticeModel{2}(SquareLattice(a), K, M)
MonoatomicLatticeModel{3}(K::Float64, M::Float64, a::Float64=1.0) = MonoatomicLatticeModel{3}(CubicLattice(a), K, M)
