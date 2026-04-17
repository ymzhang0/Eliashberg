# Models/dispersions.jl

# ---------------------------------------------------------
# Electronic Dispersion Structs
# ---------------------------------------------------------

struct FreeElectron{D} <: ElectronicDispersion{D}
    EF::Float64
    mass::Float64
end
Base.show(io::IO, m::FreeElectron{D}) where {D} = print(io, "FreeElectron (", D, "D, EF=", m.EF, ")")
FreeElectron{D}(EF::Float64) where D = FreeElectron{D}(EF, 1.0) # default mass=1

"""
    TightBinding{D} <: ElectronicDispersion{D}

A universal Tight-Binding model based on a real-space lattice.
"""
struct TightBinding{D} <: ElectronicDispersion{D}
    lattice::SMatrix{D,D,Float64}
    hoppings::Vector{Tuple{SVector{D,Int},Float64}}
    EF::Float64
end
Base.show(io::IO, m::TightBinding{D}) where {D} = print(io, "TightBinding (", D, "D, ", length(m.hoppings), " hoppings, EF=", m.EF, ")")

"""
    SpinorDispersion{D,M} <: ElectronicDispersion{D}

Opt-in wrapper that promotes an existing bare electronic dispersion into a
spin-degenerate basis. The wrapped model keeps its original orbital structure,
while `ε(k)` is lifted to a block-diagonal Hamiltonian with explicit spin-up and
spin-down sectors.
"""
struct SpinorDispersion{D,M<:ElectronicDispersion{D}} <: ElectronicDispersion{D}
    bare::M
end
Base.show(io::IO, m::SpinorDispersion{D}) where {D} = print(io, "SpinorDispersion (", D, "D, bare=", m.bare, ")")

SpinorDispersion(model::SpinorDispersion) = model

"""
    MultiOrbitalTightBinding{D} <: ElectronicDispersion{D}

Multi-orbital tight-binding model defined on a Bravais `cell`, stored
internally as a primitive-vector matrix. Each hopping is
stored as `(orbital_i, orbital_j, cell_offset_R, t)` and is interpreted in the
Wannier-style convention where the Bloch phase depends only on the lattice
translation `R`.
"""
struct MultiOrbitalTightBinding{D} <: ElectronicDispersion{D}
    lattice::SMatrix{D,D,Float64}
    periodicity::NTuple{D,Bool}
    num_orbitals::Int
    hoppings::Vector{Tuple{Int,Int,SVector{D,Int},ComplexF64}}
    EF::Float64
end
Base.show(io::IO, m::MultiOrbitalTightBinding{D}) where {D} = print(io, "MultiOrbitalTightBinding (", D, "D, ", m.num_orbitals, " orbitals, EF=", m.EF, ")")

primitive_vectors(model::MultiOrbitalTightBinding) = getfield(model, :lattice)
periodicity(model::MultiOrbitalTightBinding) = getfield(model, :periodicity)

function periodic_cell(model::MultiOrbitalTightBinding{D}; length_unit=u"Å") where {D}
    return PeriodicCell(primitive_vectors(model); periodicity=periodicity(model), length_unit)
end

function Base.getproperty(model::MultiOrbitalTightBinding, name::Symbol)
    if name === :cell
        return periodic_cell(model)
    elseif name === :lattice
        return getfield(model, :lattice)
    end
    return getfield(model, name)
end

"""
    MultiOrbitalTightBinding(obj, num_orbitals, hoppings, EF)

Build a `MultiOrbitalTightBinding` model from an `AtomsBase.AbstractSystem` or `PeriodicCell`.
"""
function MultiOrbitalTightBinding(obj::Union{AbstractSystem,PeriodicCell}, num_orbitals, hoppings, EF)
    return with_stage_log(
        "Construct MultiOrbitalTightBinding";
        context=(cell=obj, num_orbitals=Int(num_orbitals), EF=Float64(EF)),
        summarize_result=identity,
    ) do
        primitive_cell = primitive_vectors(obj)
        D = size(primitive_cell, 1)
        typed_hoppings = Tuple{Int,Int,SVector{D,Int},ComplexF64}[
            (Int(atom_i), Int(atom_j), SVector{D,Int}(cell_offset_R), ComplexF64(t))
            for (atom_i, atom_j, cell_offset_R, t) in hoppings
        ]
        return MultiOrbitalTightBinding{D}(primitive_cell, periodicity(obj), Int(num_orbitals), typed_hoppings, Float64(EF))
    end
end

# Legacy Crystal and duplicate AbstractSystem/PeriodicCell overloads removed.

# ---------------------------------------------------------
# Phonon Dispersion Structs
# ---------------------------------------------------------

struct EinsteinModel{D} <: PhononDispersion{D}
    ωE::Float64
end

struct DebyeModel{D} <: PhononDispersion{D}
    vs::Float64
    ωD::Float64
end

struct PolaritonModel{D} <: PhononDispersion{D}
    ωE::Float64  # Einstein frequency
    vs::Float64  # sound velocity
end

struct MonoatomicLatticeModel{D} <: PhononDispersion{D}
    lattice::SMatrix{D,D,Float64}
    K::Float64  # spring constant
    M::Float64  # mass
end

# --- Tight-Binding Dispatch System ---

function TightBinding(cell::Union{AbstractSystem,PeriodicCell}, t::Real, EF::Real=0.0)
    return _TightBinding(Val(bravais_lattice(cell)), cell, Float64(t), Float64(EF))
end

function _TightBinding(::Val{:line}, cell, t, EF)
    hops = [(SVector{1,Int}(1), -t)]
    return TightBinding(primitive_vectors(cell), hops, EF)
end

function _TightBinding(::Val{:sqP}, cell, t, EF)
    hops = [(SVector{2,Int}(1, 0), -t), (SVector{2,Int}(0, 1), -t)]
    return TightBinding(primitive_vectors(cell), hops, EF)
end

function _TightBinding(::Val{:hP}, cell, t, EF)
    # Triangular lattice hoppings
    hops = [(SVector{2,Int}(1, 0), -t), (SVector{2,Int}(0, 1), -t), (SVector{2,Int}(-1, 1), -t)]
    return TightBinding(primitive_vectors(cell), hops, EF)
end

function _TightBinding(::Val{:cP}, cell, t, EF)
    hops = [(SVector{3,Int}(1, 0, 0), -t), (SVector{3,Int}(0, 1, 0), -t), (SVector{3,Int}(0, 0, 1), -t)]
    return TightBinding(primitive_vectors(cell), hops, EF)
end

function _TightBinding(::Val{:cF}, cell, t, EF)
    hops = [
        (SVector{3,Int}(1, 0, 0), -t), (SVector{3,Int}(0, 1, 0), -t), (SVector{3,Int}(0, 0, 1), -t),
        (SVector{3,Int}(1, -1, 0), -t), (SVector{3,Int}(0, 1, -1), -t), (SVector{3,Int}(-1, 0, 1), -t),
    ]
    return TightBinding(primitive_vectors(cell), hops, EF)
end

function _TightBinding(::Val{:cI}, cell, t, EF)
    hops = [
        (SVector{3,Int}(1, 0, 0), -t), (SVector{3,Int}(0, 1, 0), -t), (SVector{3,Int}(0, 0, 1), -t),
        (SVector{3,Int}(1, 1, 1), -t),
    ]
    return TightBinding(primitive_vectors(cell), hops, EF)
end

function _TightBinding(v::Val{S}, cell, t, EF) where {S}
    throw(ArgumentError("No default Tight-Binding pattern defined for Bravais lattice $S. Please provide explicit hoppings."))
end

# --- Primary Constructors ---

"""
    TightBinding(obj, hoppings, EF)

Build a `TightBinding` model from an `AtomsBase.AbstractSystem` or `PeriodicCell`.
"""
function TightBinding(obj::Union{AbstractSystem,PeriodicCell}, hoppings, EF)
    return with_stage_log(
        "Construct TightBinding";
        context=(cell=obj, EF=Float64(EF), n_hoppings=length(hoppings)),
        summarize_result=identity,
    ) do
        primitive_lattice = primitive_vectors(obj)
        D = size(primitive_lattice, 1)
        typed_hoppings = Tuple{SVector{D,Int},Float64}[
            (SVector{D,Int}(R_idx), Float64(t_hop))
            for (R_idx, t_hop) in hoppings
        ]
        return TightBinding{D}(primitive_lattice, typed_hoppings, Float64(EF))
    end
end

# Redundant matrix and system overloads removed.

TightBinding(cell::Union{AbstractSystem,PeriodicCell}, t::Float64, tp::Float64, EF::Float64) =
    _TightBinding(Val(:sqP_with_tp), cell, t, tp, EF)

function _TightBinding(::Val{:sqP_with_tp}, cell, t, tp, EF)
    hops = [(SVector{2,Int}(1, 0), -t), (SVector{2,Int}(0, 1), -t),
        (SVector{2,Int}(1, 1), -tp), (SVector{2,Int}(-1, 1), -tp)]
    return TightBinding(primitive_vectors(cell), hops, EF)
end

# Predefined models (GrapheneModel, KagomeModel, SSHModel) moved to predefined_models.jl



# Band calculation functions (compute_band_data, etc.) moved to calculator.jl
