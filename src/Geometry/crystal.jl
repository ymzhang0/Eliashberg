# src/Geometry/crystal.jl

using AtomsBase
using Unitful
using StaticArrays
using LinearAlgebra

# ---------------------------------------------------------
# Interface Definitions
# ---------------------------------------------------------

function periodic_rank(lattice::AbstractBravaisLattice{D}) where {D}
    return D
end

function periodic_rank(cell::PeriodicCell{D}) where {D}
    return D
end

function periodic_rank(system::AbstractSystem)
    return AtomsBase.n_dimensions(system)
end

"""
    primitive_vectors(system::AbstractSystem)
    primitive_vectors(cell::PeriodicCell)

Return a matrix where columns are the primitive lattice vectors (stripped of units, default Å).
"""
function primitive_vectors(system::AbstractSystem)
    box = AtomsBase.cell_vectors(system)
    # Unit check and convert to Ångström
    return hcat([ustrip.(uconvert.(u"Å", v)) for v in box]...)
end


function primitive_vectors(cell::PeriodicCell)
    box = AtomsBase.cell_vectors(cell)
    # Unit check and convert to Ångström
    return hcat([ustrip.(uconvert.(u"Å", v)) for v in box]...)
end

# ---------------------------------------------------------
# Periodic System Construction (from Lattices)
# ---------------------------------------------------------

# Internal helper to handle the [element => pos] Pair format
function _process_atoms(atoms::AbstractVector, D::Int, length_unit, fractional::Bool)
    if isempty(atoms)
        return []
    end
    
    # Check if first element is a Pair (our convenience format)
    if first(atoms) isa Pair
        # Use Pair instead of Tuple, as AtomsBase.parse_fractional expects Pair or Atom
        return [
            Symbol(a.first) => (fractional ? SVector{D}(Float64.(a.second)) : SVector{D}(Float64.(a.second)) .* length_unit)
            for a in atoms
        ]
    end
    
    # Fallback: assume it's already in a format AtomsBase likes
    return atoms
end

function AtomsBase.periodic_system(
    atoms::AbstractVector,
    lattice::AbstractBravaisLattice{D};
    length_unit=u"Å",
    kwargs...
) where {D}
    is_fractional = get(kwargs, :fractional, false)
    processed_atoms = _process_atoms(atoms, D, length_unit, is_fractional)
    
    # Extract box vectors
    matrix = primitive_cell(lattice)
    box = ntuple(i -> SVector{D}(matrix[:, i]) .* length_unit, D)

    return periodic_system(processed_atoms, box; kwargs...)
end

function AtomsBase.periodic_system(
    atoms::AbstractVector,
    cell::PeriodicCell{D};
    kwargs...
) where {D}
    # For PeriodicCell, use its own units for atoms unless specified
    is_fractional = get(kwargs, :fractional, false)
    processed_atoms = _process_atoms(atoms, D, u"Å", is_fractional) 
    box = cell_vectors(cell)
    
    return periodic_system(processed_atoms, box; kwargs...)
end

# QE ibrav Lattice Generation logic removed from crystal.jl

# ---------------------------------------------------------
# Spglib Integration
# ---------------------------------------------------------

"""
    Spglib.SpglibCell(system::AbstractSystem)

Convert an `AtomsBase.AbstractSystem` to an `Spglib.SpglibCell`.
Strictly restricted to 3D systems.
"""
function Spglib.SpglibCell(system::AbstractSystem)
    D = AtomsBase.n_dimensions(system)
    D == 3 || throw(ArgumentError("Spglib symmetry analysis is only supported for 3D systems (got $(D)D)."))

    # Lattice matrix (columns are primitive vectors)
    lattice = primitive_vectors(system)

    # Fractional positions
    # AtomsBase system.position returns Cartesian vectors with units
    pos_cartesian = hcat([ustrip.(p) for p in AtomsBase.position(system)]...) # 3xN matrix
    pos_fractional = inv(lattice) * pos_cartesian

    # Atomic types (Spglib needs integers)
    symbols = AtomsBase.atomic_symbol(system)
    unique_symbols = unique(symbols)
    sym_to_id = Dict(sym => i for (i, sym) in enumerate(unique_symbols))
    atom_types = [sym_to_id[s] for s in symbols]

    return SpglibCell(lattice, [SVector{3}(pos_fractional[:, i]) for i in 1:length(symbols)], atom_types)
end

# --- Unified Bravais Identification ---

"""
    bravais_lattice(lattice::AbstractBravaisLattice)
    bravais_lattice(system::Union{AbstractSystem, PeriodicCell})

Identify the Bravais lattice type. Returns a symbol (e.g., `:cP`, `:sqP`, `:hP`).
For `AbstractBravaisLattice` types, this is a zero-cost type lookup.
For `AbstractSystem`, it performs geometric analysis (Spglib for 3D, manual for 1D/2D).
"""
bravais_lattice(::ChainLattice) = :line
bravais_lattice(::SquareLattice) = :sqP
bravais_lattice(::HexagonalLattice2D) = :hP
bravais_lattice(::SimpleCubic) = :cP
bravais_lattice(::FaceCenteredCubic) = :cF
bravais_lattice(::BodyCenteredCubic) = :cI

function bravais_lattice(obj::Union{AbstractSystem,PeriodicCell})
    D = AtomsBase.n_dimensions(obj)
    if D == 1
        return :line
    elseif D == 2
        return _identify_2d_lattice(primitive_vectors(obj))
    elseif D == 3
        cell = SpglibCell(obj)
        dataset = get_dataset(cell)
        return _bravais_from_spacegroup(dataset.spacegroup_number, dataset.international_symbol)
    end
    return :unknown
end

function _identify_2d_lattice(lattice::AbstractMatrix; atol=1e-5)
    a1, a2 = lattice[:, 1], lattice[:, 2]
    len1, len2 = norm(a1), norm(a2)
    cos_gamma = dot(a1, a2) / (len1 * len2)

    is_square = isapprox(len1, len2; atol=atol) && abs(cos_gamma) < atol
    if is_square
        return :sqP
    end

    is_hexagonal = isapprox(len1, len2; atol=atol) && (isapprox(abs(cos_gamma), 0.5; atol=atol))
    if is_hexagonal
        return :hP
    end

    is_rectangular = abs(cos_gamma) < atol
    if is_rectangular
        return :rP
    end

    return :obP # Oblique
end

# Internal mapping for Bravais identification
function _bravais_from_spacegroup(sg_number::Int, symbol::AbstractString)
    # 1. Cubic (221-230)
    if sg_number >= 195 && sg_number <= 230
        if contains(symbol, 'F') return :cF end
        if contains(symbol, 'I') return :cI end
        return :cP
    # 2. Hexagonal (168-194)
    elseif sg_number >= 168 && sg_number <= 194
        return :hP
    # 3. Rhombohedral (143-167)
    elseif sg_number >= 143 && sg_number <= 167
        return :rR
    # 4. Tetragonal (75-142)
    elseif sg_number >= 75 && sg_number <= 142
        if contains(symbol, 'I') return :tI end
        return :tP
    # 5. Orthorhombic (16-74)
    elseif sg_number >= 16 && sg_number <= 74
        if contains(symbol, 'F') return :oF end
        if contains(symbol, 'I') return :oI end
        if contains(symbol, 'C') || contains(symbol, 'A') || contains(symbol, 'B') return :oC end
        return :oP
    # 6. Monoclinic (3-15)
    elseif sg_number >= 3 && sg_number <= 15
        if contains(symbol, 'C') || contains(symbol, 'A') || contains(symbol, 'I') return :mC end
        return :mP
    # 7. Triclinic (1-2)
    else
        return :aP
    end
end

# Legacy factories moved to predefined_structures.jl
