# ---------------------------------------------------------
# 1D Lattices
# ---------------------------------------------------------
struct Lattice{D} <: AbstractLattice{D}
    vectors::SMatrix{D,D,Float64}
end

struct ChainLattice <: AbstractLattice{1}
    a::Float64
    vectors::SMatrix{1,1,Float64,1}
end
ChainLattice(a::Float64=1.0) = ChainLattice(a, SMatrix{1,1,Float64,1}(a))

# ---------------------------------------------------------
# 2D Lattices
# ---------------------------------------------------------
struct SquareLattice <: AbstractLattice{2}
    a::Float64
    vectors::SMatrix{2,2,Float64,4}
end
SquareLattice(a::Float64=1.0) = SquareLattice(a, @SMatrix [a 0.0; 0.0 a])

struct HexagonalLattice <: AbstractLattice{2}
    a::Float64
    vectors::SMatrix{2,2,Float64,4}
end
HexagonalLattice(a::Float64=1.0) = HexagonalLattice(a, @SMatrix [a a/2; 0.0 a*sqrt(3)/2])

# ---------------------------------------------------------
# 3D Lattices
# ---------------------------------------------------------
struct CubicLattice <: AbstractLattice{3}
    a::Float64
    vectors::SMatrix{3,3,Float64,9}
end
CubicLattice(a::Float64=1.0) = CubicLattice(a, @SMatrix [a 0.0 0.0; 0.0 a 0.0; 0.0 0.0 a])

struct FCCLattice <: AbstractLattice{3}
    a::Float64
    vectors::SMatrix{3,3,Float64,9}
end
FCCLattice(a::Float64=1.0) = FCCLattice(a, @SMatrix [0.0 a/2 a/2; a/2 0.0 a/2; a/2 a/2 0.0])

struct BCCLattice <: AbstractLattice{3}
    a::Float64
    vectors::SMatrix{3,3,Float64,9}
end
BCCLattice(a::Float64=1.0) = BCCLattice(a, @SMatrix [-a/2 a/2 a/2; a/2 -a/2 a/2; a/2 a/2 -a/2])

