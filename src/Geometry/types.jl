# src/Geometry/types.jl

"""
    AbstractLattice{D}

Represents a D-dimensional Bravais lattice in real space.
"""
abstract type AbstractLattice{D} end

abstract type AbstractKGrid{D} end

Base.length(g::AbstractKGrid) = length(g.points)
Base.iterate(g::AbstractKGrid, state=1) = iterate(g.points, state)
Base.getindex(g::AbstractKGrid, i::Int) = g.points[i]
Base.firstindex(g::AbstractKGrid) = 1
Base.lastindex(g::AbstractKGrid) = length(g.points)

"""
    KGrid{D} <: AbstractKGrid{D}

A concrete generic implementation of a D-dimensional K-grid.
Contains the grid `points` as `SVector{D, Float64}` and corresponding 
integration `weights`.
"""
struct KGrid{D} <: AbstractKGrid{D}
    points::Vector{SVector{D,Float64}}
    weights::Vector{Float64}
end

Base.eltype(::Type{<:AbstractKGrid{D}}) where {D} = SVector{D,Float64}

"""
    KPath{D} <: AbstractKGrid{D}

用于绘制能带图的高对称线路径。
不仅包含 k 点，还记录了高对称点（拐点）的索引和标签。
"""
struct KPath{D} <: AbstractKGrid{D}
    points::Vector{SVector{D,Float64}}
    weights::Vector{Float64}
    node_indices::Vector{Int}
    node_labels::Vector{String}
end