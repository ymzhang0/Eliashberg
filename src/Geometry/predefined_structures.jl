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
    diamond(element, a)
Generic diamond structure (FCC with two atoms at 0,0,0 and 0.25,0.25,0.25).
"""
function diamond(element, a)
    lattice = FaceCenteredCubic(a)
    atoms = [
        element => [0.0, 0.0, 0.0],
        element => [0.25, 0.25, 0.25]
    ]
    return periodic_system(atoms, lattice; fractional=true)
end

"""
    silicon(a=5.431)
Silicon in diamond structure.
"""
silicon(a=5.431) = diamond(:Si, a)

"""
    germanium(a=5.658)
Germanium in diamond structure.
"""
germanium(a=5.658) = diamond(:Ge, a)

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
