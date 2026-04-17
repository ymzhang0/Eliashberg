# src/Geometry/symmetry_points.jl

"""
    symmetry_path(lattice::AbstractBravaisLattice)

Returns a `SymmetryPath{D}` object containing point definitions and the default path sequence.
"""
function symmetry_path(::AbstractBravaisLattice{D}) where {D}
    return nothing
end

# 1D Lattices
function symmetry_path(::ChainLattice)
    points = Dict(
        "Γ" => SVector{1,Float64}(0.0),
        "X" => SVector{1,Float64}(0.5)
    )
    path = ["Γ", "X"]
    return SymmetryPath{1}(points, path)
end

# 2D Lattices
function symmetry_path(::SquareLattice)
    points = Dict(
        "Γ" => SVector{2,Float64}(0.0, 0.0),
        "X" => SVector{2,Float64}(0.5, 0.0),
        "M" => SVector{2,Float64}(0.5, 0.5)
    )
    path = ["Γ", "X", "M", "Γ"]
    return SymmetryPath{2}(points, path)
end

function symmetry_path(::RectangularLattice)
    points = Dict(
        "Γ" => SVector{2,Float64}(0.0, 0.0),
        "X" => SVector{2,Float64}(0.5, 0.0),
        "M" => SVector{2,Float64}(0.5, 0.5)
    )
    path = ["Γ", "X", "M", "Γ"]
    return SymmetryPath{2}(points, path)
end

function symmetry_path(::HexagonalLattice2D)
    points = Dict(
        "Γ" => SVector{2,Float64}(0.0, 0.0),
        "K" => SVector{2,Float64}(1 / 3, 1 / 3),
        "M" => SVector{2,Float64}(1 / 2, 0.0)
    )
    path = ["Γ", "K", "M", "Γ"]
    return SymmetryPath{2}(points, path)
end
