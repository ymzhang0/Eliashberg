# src/Geometry/types.jl

"""
    AbstractKGrid{D}

Abstract type for all K-space grids and paths.
"""
abstract type AbstractKGrid{D} end
Base.show(io::IO, ::AbstractKGrid{D}) where {D} = print(io, "AbstractKGrid (", D, "D)")

const KPath = Brillouin.KPathInterpolant
