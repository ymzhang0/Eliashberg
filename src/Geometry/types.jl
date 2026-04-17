# src/Geometry/types.jl

"""
    AbstractKGrid{D}

Abstract type for all K-space grids and paths.
"""
abstract type AbstractKGrid{D} end
Base.show(io::IO, ::AbstractKGrid{D}) where {D} = print(io, "AbstractKGrid (", D, "D)")

const KPath = Brillouin.KPathInterpolant

"""
    SymmetryPath{D}

Groups high-symmetry point coordinates and a default path sequence for a D-dimensional lattice.
"""
struct SymmetryPath{D}
    points::Dict{String, SVector{D, Float64}}
    path::Vector{String}
end
