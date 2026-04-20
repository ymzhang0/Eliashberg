# src/Geometry/predefined_structures.jl

using AtomsBase
using Unitful
using StaticArrays

# ---------------------------------------------------------
# Lattice Factories (Returning PeriodicCell)
# ---------------------------------------------------------


# ---------------------------------------------------------
# Common Crystal Structures
# ---------------------------------------------------------

"""
    atomic_chain(a, element=:C)
Simple 1D atomic chain.
"""
function atomic_chain(a=1.0, element=:C)
    lattice = ChainLattice(a)
    atoms = [element => [0.0]]
    return periodic_system(atoms, lattice; fractional=true)
end

"""
    square_lattice(a, element=:C)
Simple 2D square lattice.
"""
function square_lattice(a=1.0, element=:C)
    lattice = SquareLattice(a)
    atoms = [element => [0.0, 0.0]]
    return periodic_system(atoms, lattice; fractional=true)
end

"""
    cF(element, a)
Generic Face-centered cubic (FCC) structure.
"""
function cF(element, a)
    lattice = FaceCenteredCubic(a)
    atoms = [
        element => [0.0, 0.0, 0.0],
    ]
    return periodic_system(atoms, lattice; fractional=true)
end

aluminium(a=4.0493) = cF(:Al, a)
copper(a=3.615) = cF(:Cu, a)


"""
    cI(element, a)
Generic Body-centered cubic (BCC) structure.
"""
function cI(element, a)
    lattice = BodyCenteredCubic(a)
    atoms = [
        element => [0.0, 0.0, 0.0],
    ]
    return periodic_system(atoms, lattice; fractional=true)
end

iron(a=2.8665) = cI(:Fe, a)
niobium(a=3.3005) = cI(:Nb, a)
vanadium(a=3.0272) = cI(:V, a)

"""
    cF8(element, a)
Generic diamond structure (FCC with two atoms at 0,0,0 and 0.25,0.25,0.25).
"""
function cF8(element, a)
    lattice = FaceCenteredCubic(a)
    atoms = [
        element => [0.0, 0.0, 0.0],
        element => [0.25, 0.25, 0.25]
    ]
    return periodic_system(atoms, lattice; fractional=true)
end


"""
    diamond(a=5.431)
Diamond in cF8 structure.
"""
diamond(a=3.567) = cF8(:C, a)


"""
    silicon(a=5.431)
Silicon in cF8 structure.
"""
silicon(a=5.431) = cF8(:Si, a)

"""
    germanium(a=5.658)
Germanium in cF8 structure.
"""
germanium(a=5.658) = cF8(:Ge, a)

"""
    zincblende(e1, e2, a)
Generic zincblende structure.
"""
function zincblende(e1, e2, a)
    lattice = FaceCenteredCubic(a)
    atoms = [
        e1 => [0.0, 0.0, 0.0],
        e2 => [0.25, 0.25, 0.25]
    ]
    return periodic_system(atoms, lattice; fractional=true)
end

"""
    sic(a=4.36)
3C-SiC in zincblende structure.
"""
sic(a=4.36) = zincblende(:Si, :C, a)

"""
    nacl(a=5.64)
NaCl structure (Rock-salt, FCC).
"""
function nacl(a=5.64)
    lattice = FaceCenteredCubic(a)
    atoms = [
        :Na => [0.0, 0.0, 0.0],
        :Cl => [0.5, 0.5, 0.5]
    ]
    return periodic_system(atoms, lattice; fractional=true)
end

"""
    graphene(a=2.46)
Graphene 2D honeycomb lattice.
"""
function graphene(a=2.46)
    lattice = HexagonalLattice2D(a)
    # Atoms at 1/3, 2/3 and 2/3, 1/3
    atoms = [
        :C => [1 / 3, 2 / 3],
        :C => [2 / 3, 1 / 3]
    ]
    return periodic_system(atoms, lattice; fractional=true)
end

"""
    graphite(a=2.46, c=6.71)
3D Graphite (AB stacking).
"""
function graphite(a=2.46, c=6.71)
    lattice = HexagonalLattice(a, c)
    atoms = [
        :C => [0.0, 0.0, 0.0],
        :C => [1 / 3, 2 / 3, 0.0],
        :C => [2 / 3, 1 / 3, 0.5],
        :C => [0.0, 0.0, 0.5]
    ]
    return periodic_system(atoms, lattice; fractional=true)
end

"""
    kagome(a=1.0)
2D Kagome lattice (3 atoms per unit cell).
"""
function kagome(a=1.0)
    lattice = HexagonalLattice2D(a)
    # Sites at mid-points of the primitive vectors
    atoms = [
        :C => [0.5, 0.0],
        :C => [0.0, 0.5],
        :C => [0.5, 0.5]
    ]
    return periodic_system(atoms, lattice; fractional=true)
end

"""
    ssh_lattice(a=1.0)
1D SSH lattice (2 atoms per unit cell).
"""
function ssh_lattice(a=1.0)
    lattice = ChainLattice(a)
    atoms = [
        :C => [0.0],
        :C => [0.5]
    ]
    return periodic_system(atoms, lattice; fractional=true)
end
# ---------------------------------------------------------
# Registry for Factory Dispatch
# ---------------------------------------------------------

"""
    PREDEFINED_STRUCTURE_REGISTRY

A dictionary mapping structure names to builder functions that accept a `PredefinedStructureOption`.
This allows the factory to be extended without modifying its core logic.
"""
const PREDEFINED_STRUCTURE_REGISTRY = Dict{String,Function}(
    "atomic_chain" => opt -> atomic_chain(opt.a, opt.element),
    "square_lattice" => opt -> square_lattice(opt.a, opt.element),
    "aluminium" => opt -> aluminium(opt.a),
    "copper" => opt -> copper(opt.a),
    "iron" => opt -> iron(opt.a),
    "niobium" => opt -> niobium(opt.a),
    "vanadium" => opt -> vanadium(opt.a),
    "diamond" => opt -> diamond(opt.a),
    "silicon" => opt -> silicon(opt.a),
    "germanium" => opt -> germanium(opt.a),
    "zincblende" => opt -> zincblende(opt.element, opt.element2, opt.a),
    "sic" => opt -> sic(opt.a),
    "nacl" => opt -> nacl(opt.a),
    "graphene" => opt -> graphene(opt.a),
    "graphite" => opt -> graphite(opt.a, opt.c),
    "kagome" => opt -> kagome(opt.a),
    "ssh_lattice" => opt -> ssh_lattice(opt.a),
)

"""
    register_structure!(name::String, f::Function)

Register a new predefined structure builder. `f` should take a `PredefinedStructureOption` 
and return an `AbstractSystem`.
"""
function register_structure!(name::String, f::Function)
    PREDEFINED_STRUCTURE_REGISTRY[name] = f
end
