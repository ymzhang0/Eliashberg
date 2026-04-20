# src/Geometry/Lattices.jl

using StaticArrays
using LinearAlgebra

"""
Abstract base types for all Bravais lattices.
"""
abstract type AbstractBravaisLattice{D} end

abstract type AbstractLattice1D <: AbstractBravaisLattice{1} end
abstract type AbstractLattice2D <: AbstractBravaisLattice{2} end
abstract type AbstractLattice3D <: AbstractBravaisLattice{3} end

# =========================================================
# 1D Lattices (1 Type)
# =========================================================

"""
One-dimensional primitive chain lattice.
"""
struct ChainLattice <: AbstractLattice1D
    a::Float64
end

function primitive_cell(lattice::ChainLattice)
    return @SMatrix [lattice.a]
end

lattice_length(lattice::ChainLattice) = lattice.a

# =========================================================
# 2D Lattices (5 Types)
# =========================================================

"""
2D Oblique lattice.
Defined by lengths `a`, `b` and angle `γ` (in radians).
"""
struct ObliqueLattice <: AbstractLattice2D
    a::Float64
    b::Float64
    γ::Float64
end

"""
2D Rectangular primitive lattice.
"""
struct RectangularLattice <: AbstractLattice2D
    a::Float64
    b::Float64
end

"""
2D Centered Rectangular lattice.
"""
struct CenteredRectangularLattice <: AbstractLattice2D
    a::Float64
    b::Float64
end

"""
2D Square lattice.
"""
struct SquareLattice <: AbstractLattice2D
    a::Float64
end

"""
2D Hexagonal lattice.
"""
struct HexagonalLattice2D <: AbstractLattice2D
    a::Float64
end

# --- 2D Primitive Cells ---

function primitive_cell(lattice::ObliqueLattice)
    a, b, γ = lattice.a, lattice.b, lattice.γ
    return @SMatrix [a b*cos(γ);
        0.0 b*sin(γ)]
end

function primitive_cell(lattice::RectangularLattice)
    return @SMatrix [lattice.a 0.0;
        0.0 lattice.b]
end

function primitive_cell(lattice::CenteredRectangularLattice)
    a, b = lattice.a, lattice.b
    return @SMatrix [a/2 a/2;
        -b/2 b/2]
end

function primitive_cell(lattice::SquareLattice)
    a = lattice.a
    return @SMatrix [a 0.0;
        0.0 a]
end

function primitive_cell(lattice::HexagonalLattice2D)
    a = lattice.a
    return @SMatrix [a a/2;
        0.0 a*sqrt(3)/2]
end

# --- 2D Area ---
lattice_area(lattice::AbstractLattice2D) = abs(det(primitive_cell(lattice)))

# =========================================================
# 3D Lattices (14 Types)
# =========================================================

"""
Simple Cubic (cP) lattice.
"""
struct SimpleCubic <: AbstractLattice3D
    a::Float64
end

"""
Face-Centered Cubic (cF) lattice.
"""
struct FaceCenteredCubic <: AbstractLattice3D
    a::Float64
end

"""
Body-Centered Cubic (cI) lattice.
"""
struct BodyCenteredCubic <: AbstractLattice3D
    a::Float64
end

"""
Simple Tetragonal (tP) lattice.
"""
struct SimpleTetragonal <: AbstractLattice3D
    a::Float64
    c::Float64
end

"""
Body-Centered Tetragonal (tI) lattice.
"""
struct BodyCenteredTetragonal <: AbstractLattice3D
    a::Float64
    c::Float64
end

"""
Simple Orthorhombic (oP) lattice.
"""
struct SimpleOrthorhombic <: AbstractLattice3D
    a::Float64
    b::Float64
    c::Float64
end

"""
Base-Centered Orthorhombic (oC) lattice.
"""
struct BaseCenteredOrthorhombic <: AbstractLattice3D
    a::Float64
    b::Float64
    c::Float64
end

"""
Face-Centered Orthorhombic (oF) lattice.
"""
struct FaceCenteredOrthorhombic <: AbstractLattice3D
    a::Float64
    b::Float64
    c::Float64
end

"""
Body-Centered Orthorhombic (oI) lattice.
"""
struct BodyCenteredOrthorhombic <: AbstractLattice3D
    a::Float64
    b::Float64
    c::Float64
end

"""
Rhombohedral (hR) lattice.
Defined by length `a` and angle `α` (in radians).
"""
struct RhombohedralLattice <: AbstractLattice3D
    a::Float64
    α::Float64
end

"""
Simple Hexagonal (hP) lattice.
"""
struct SimpleHexagonal <: AbstractLattice3D
    a::Float64
    c::Float64
end

"""
Monoclinic Primitive (mP) lattice.
`angle` corresponds to γ if `unique_axis == :c`, or β if `unique_axis == :b`.
"""
struct MonoclinicPrimitive <: AbstractLattice3D
    a::Float64
    b::Float64
    c::Float64
    angle::Float64
    unique_axis::Symbol # :b or :c
end

"""
Monoclinic Base-Centered (mC) lattice.
"""
struct MonoclinicBaseCentered <: AbstractLattice3D
    a::Float64
    b::Float64
    c::Float64
    angle::Float64
    unique_axis::Symbol # :b or :c
end

"""
Triclinic (aP) lattice.
Angles α, β, γ in radians.
"""
struct TriclinicLattice <: AbstractLattice3D
    a::Float64
    b::Float64
    c::Float64
    α::Float64
    β::Float64
    γ::Float64
end

# --- 3D Primitive Cells (QE/Standard conventions) ---

function primitive_cell(lattice::SimpleCubic)
    a = lattice.a
    return @SMatrix [a 0.0 0.0;
        0.0 a 0.0;
        0.0 0.0 a]
end

function primitive_cell(lattice::FaceCenteredCubic)
    a = lattice.a
    return @SMatrix [0.0 a/2 a/2;
                     a/2 0.0 a/2;
                     a/2 a/2 0.0]
end

function primitive_cell(lattice::BodyCenteredCubic)
    a = lattice.a
    return @SMatrix [-a/2 a/2 a/2;
        a/2 -a/2 a/2;
        a/2 a/2 -a/2]
end

function primitive_cell(lattice::SimpleTetragonal)
    a, c = lattice.a, lattice.c
    return @SMatrix [a 0.0 0.0;
        0.0 a 0.0;
        0.0 0.0 c]
end

function primitive_cell(lattice::BodyCenteredTetragonal)
    a, c = lattice.a, lattice.c
    return @SMatrix [
        a/2 a/2 -a/2;
        -a/2 a/2 -a/2;
        c/2 c/2 c/2
    ]
end

function primitive_cell(lattice::SimpleOrthorhombic)
    a, b, c = lattice.a, lattice.b, lattice.c
    return @SMatrix [a 0.0 0.0;
        0.0 b 0.0;
        0.0 0.0 c]
end

function primitive_cell(lattice::BaseCenteredOrthorhombic)
    a, b, c = lattice.a, lattice.b, lattice.c
    return @SMatrix [
        a/2 -a/2 0.0;
        b/2 b/2 0.0;
        0.0 0.0 c
    ]
end

function primitive_cell(lattice::FaceCenteredOrthorhombic)
    a, b, c = lattice.a, lattice.b, lattice.c
    return @SMatrix [
        a/2 a/2 0.0;
        0.0 b/2 b/2;
        c/2 0.0 c/2]
end

function primitive_cell(lattice::BodyCenteredOrthorhombic)
    a, b, c = lattice.a, lattice.b, lattice.c
    return @SMatrix [
        a/2 -a/2 -a/2;
        b/2 b/2 -b/2;
        c/2 c/2 c/2
    ]
end

function primitive_cell(lattice::RhombohedralLattice)
    a, α = lattice.a, lattice.α
    cosα = cos(α)
    tx = sqrt((1 - cosα) / 2)
    ty = sqrt((1 - cosα) / 6)
    tz = sqrt((1 + 2cosα) / 3)
    return a .* @SMatrix [tx 0.0 -tx;
        -ty 2ty -ty;
        tz tz tz]
end

function primitive_cell(lattice::SimpleHexagonal)
    a, c = lattice.a, lattice.c
    return @SMatrix [a -a/2 0.0;
        0.0 a*sqrt(3)/2 0.0;
        0.0 0.0 c]
end

function primitive_cell(lattice::MonoclinicPrimitive)
    a, b, c, angle, axis = lattice.a, lattice.b, lattice.c, lattice.angle, lattice.unique_axis
    if axis == :c
        return @SMatrix [a b*cos(angle) 0.0;
            0.0 b*sin(angle) 0.0;
            0.0 0.0 c]
    elseif axis == :b
        return @SMatrix [a 0.0 c*cos(angle);
            0.0 b 0.0;
            0.0 0.0 c*sin(angle)]
    end
    error("Invalid unique axis: $axis. Must be :b or :c.")
end

function primitive_cell(lattice::MonoclinicBaseCentered)
    a, b, c, angle, axis = lattice.a, lattice.b, lattice.c, lattice.angle, lattice.unique_axis
    if axis == :c
        return @SMatrix [a/2 b*cos(angle) a/2;
            0.0 b*sin(angle) 0.0;
            -c/2 0.0 c/2]
    elseif axis == :b
        return @SMatrix [a/2 -a/2 c*cos(angle);
            b/2 b/2 0.0;
            0.0 0.0 c*sin(angle)]
    end
    error("Invalid unique axis: $axis. Must be :b or :c.")
end

function primitive_cell(lattice::TriclinicLattice)
    a, b, c = lattice.a, lattice.b, lattice.c
    α, β, γ = lattice.α, lattice.β, lattice.γ
    sinγ = sin(γ)
    cosα, cosβ, cosγ = cos(α), cos(β), cos(γ)
    vy = c * (cosα - cosβ * cosγ) / sinγ
    vz = c * sqrt(1 + 2cosα * cosβ * cosγ - cosα^2 - cosβ^2 - cosγ^2) / sinγ
    return @SMatrix [a b*cosγ c*cosβ;
        0.0 b*sinγ vy;
        0.0 0.0 vz]
end

# --- 3D Volume ---
lattice_volume(lattice::AbstractLattice3D) = abs(det(primitive_cell(lattice)))

# =========================================================
# Generic Interface
# =========================================================

primitive_vectors(lattice::AbstractBravaisLattice) = primitive_cell(lattice)

# =========================================================
# Printing and Display
# =========================================================

function Base.show(io::IO, lattice::AbstractBravaisLattice)
    print(io, "$(typeof(lattice))(")
    params = [string(n, "=", getfield(lattice, n)) for n in fieldnames(typeof(lattice))]
    print(io, join(params, ", "), ")")
end

function Base.show(io::IO, mime::MIME"text/plain", lattice::AbstractBravaisLattice{D}) where {D}
    println(io, "$(typeof(lattice)):")
    for name in fieldnames(typeof(lattice))
        println(io, "  $name = $(getfield(lattice, name))")
    end
    println(io, "Primitive Cell ($(D)x$(D)):")
    Base.print_matrix(io, primitive_cell(lattice))
end