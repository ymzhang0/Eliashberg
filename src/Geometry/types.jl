# src/Geometry/types.jl

"""
    AbstractLattice{D}

Represents a D-dimensional Bravais lattice in real space.
"""
abstract type AbstractLattice{D} end

"""
    AbstractKGrid{D}

Abstract type for all K-space grids and paths.
"""
abstract type AbstractKGrid{D} end