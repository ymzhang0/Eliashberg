"""
    primitive_vectors(lattice_like)

Return the column-major matrix whose columns are the primitive real-space basis
vectors of the lattice-like object.
"""
primitive_vectors(lattice::AbstractLattice) = getfield(lattice, :vectors)

@inline _strip_length(value::Real, length_unit) = Float64(value)
@inline _strip_length(value, length_unit) = Float64(ustrip(uconvert(length_unit, value)))

function _strip_length_vector(vector_like, length_unit, ::Val{D}) where {D}
    return SVector{D,Float64}(ntuple(i -> _strip_length(vector_like[i], length_unit), D))
end

function _coerce_lattice_matrix(vectors::AbstractLattice, length_unit)
    return primitive_vectors(vectors)
end

function _coerce_lattice_matrix(vectors::AbstractMatrix, length_unit)
    size(vectors, 1) == size(vectors, 2) || throw(ArgumentError("`cell` must be a square matrix whose columns are primitive vectors."))
    D = size(vectors, 1)
    return SMatrix{D,D,Float64}([_strip_length(vectors[i, j], length_unit) for i in 1:D, j in 1:D])
end

function _coerce_lattice_matrix(vectors::Tuple, length_unit)
    D = length(vectors)
    all(length(vector) == D for vector in vectors) || throw(ArgumentError("`cell` must contain exactly $D primitive vectors of length $D."))
    return SMatrix{D,D,Float64}(hcat((_strip_length_vector(vector, length_unit, Val(D)) for vector in vectors)...))
end

function _coerce_lattice_matrix(vectors::AbstractVector, length_unit)
    D = length(vectors)
    all(length(vector) == D for vector in vectors) || throw(ArgumentError("`cell` must contain exactly $D primitive vectors of length $D."))
    return SMatrix{D,D,Float64}(hcat((_strip_length_vector(vector, length_unit, Val(D)) for vector in vectors)...))
end

function _normalize_fractional_position(position, lattice, ::Val{D}; fractional, length_unit, wrap::Bool=true) where {D}
    coordinates = _strip_length_vector(position, length_unit, Val(D))
    fractional_coordinates = fractional ? coordinates : inv(lattice) * coordinates
    return wrap ? mod.(fractional_coordinates, 1.0) : fractional_coordinates
end

_normalize_symbol(symbol) = Symbol(symbol)

"""
    IBravKeywordSpec

Describe one keyword-like parameter used by the QE-style `ibrav` constructor.
"""
struct IBravKeywordSpec
    name::Symbol
    meaning::String
end

"""
    IBravPositionalSpec

Describe one positional parameter used by the QE-style `ibrav` constructor.
"""
struct IBravPositionalSpec
    name::Symbol
    meaning::String
end

"""
    IBravInfo

Human-readable metadata for a Quantum ESPRESSO `ibrav` choice.
"""
struct IBravInfo
    id::Int
    name::String
    signature::String
    positional::Vector{IBravPositionalSpec}
    required_keywords::Vector{IBravKeywordSpec}
    optional_keywords::Vector{IBravKeywordSpec}
    example::String
end

function Base.show(io::IO, ::MIME"text/plain", info::IBravInfo)
    println(io, "IBrav $(info.id): $(info.name)")
    println(io, "  Signature: $(info.signature)")
    if !isempty(info.positional)
        println(io, "  Positional:")
        for spec in info.positional
            println(io, "    - $(spec.name): $(spec.meaning)")
        end
    end
    if !isempty(info.required_keywords)
        println(io, "  Required keywords:")
        for spec in info.required_keywords
            println(io, "    - $(spec.name): $(spec.meaning)")
        end
    else
        println(io, "  Required keywords: none")
    end
    if !isempty(info.optional_keywords)
        println(io, "  Optional keywords:")
        for spec in info.optional_keywords
            println(io, "    - $(spec.name): $(spec.meaning)")
        end
    end
    print(io, "  Example: $(info.example)")
end

Base.show(io::IO, info::IBravInfo) = print(io, "IBravInfo($(info.id), $(repr(info.name)))")

function _ibrav_keyword_docs(id::Integer)
    docs = Dict{Symbol,String}()
    if id in (8, 9, -9, 10, 11, 12, -12, 13, -13, 14, 91)
        docs[:celldm2] = "b / a"
    end
    if id in (4, 6, 7, 8, 9, -9, 10, 11, 12, -12, 13, -13, 14, 91)
        docs[:celldm3] = "c / a"
    end
    if id in (5, -5)
        docs[:celldm4] = "cosγ (rhombohedral angle)"
    elseif id in (12, 13)
        docs[:celldm4] = "cosγ = cos(ab)"
    elseif id == 14
        docs[:celldm4] = "cosbc = cos(alpha)"
    end
    if id in (-12, -13)
        docs[:celldm5] = "cosβ = cos(ac)"
    elseif id == 14
        docs[:celldm5] = "cosac = cos(beta)"
    end
    if id == 14
        docs[:celldm6] = "cosab = cos(gamma)"
    end
    return docs
end

function _ibrav_signature(id::Integer)
    return id == 0 ? "Lattice(vectors)" : "ibrav($id, a; ...)"
end

function _ibrav_positional_specs(id::Integer)
    return id == 0 ? IBravPositionalSpec[] : [IBravPositionalSpec(:a, "reference lattice length")]
end

function _check_positive(name::AbstractString, value::Real)
    value > 0 || throw(ArgumentError("`$name` must be strictly positive, got $value."))
    return Float64(value)
end

function _check_cosine(name::AbstractString, value::Real)
    -1.0 <= value <= 1.0 || throw(ArgumentError("`$name` must lie in [-1, 1], got $value."))
    return Float64(value)
end

function _require_parameter(name::AbstractString, value)
    isnothing(value) && throw(ArgumentError("Missing required keyword `$name` for this `ibrav` choice."))
    return value
end

function _ibrav_metadata(ibrav::Integer)
    metadata = Dict(
        -13 => (name="Monoclinic base-centered (unique axis b)", required=(:celldm2, :celldm3, :celldm5), example="ibrav(-13, a; celldm2=b_a, celldm3=c_a, celldm5=cosβ)"),
        -12 => (name="Monoclinic P (unique axis b)", required=(:celldm2, :celldm3, :celldm5), example="ibrav(-12, a; celldm2=b_a, celldm3=c_a, celldm5=cosβ)"),
        -9 => (name="Orthorhombic base-centered (alternate)", required=(:celldm2, :celldm3), example="ibrav(-9, a; celldm2=b_a, celldm3=c_a)"),
        -5 => (name="Trigonal R, 3-fold axis <111>", required=(:celldm4,), example="ibrav(-5, a; celldm4=cosγ)"),
        -3 => (name="Cubic I (bcc), symmetric axis", required=(), example="ibrav(-3, a)"),
        0 => (name="Free lattice", required=(), example="Lattice(vectors)"),
        1 => (name="Cubic P (sc)", required=(), example="ibrav(1, a)"),
        2 => (name="Cubic F (fcc)", required=(), example="ibrav(2, a)"),
        3 => (name="Cubic I (bcc)", required=(), example="ibrav(3, a)"),
        4 => (name="Hexagonal / trigonal P", required=(:celldm3,), example="ibrav(4, a; celldm3=c_a)"),
        5 => (name="Trigonal R, 3-fold axis c", required=(:celldm4,), example="ibrav(5, a; celldm4=cosγ)"),
        6 => (name="Tetragonal P", required=(:celldm3,), example="ibrav(6, a; celldm3=c_a)"),
        7 => (name="Tetragonal I", required=(:celldm3,), example="ibrav(7, a; celldm3=c_a)"),
        8 => (name="Orthorhombic P", required=(:celldm2, :celldm3), example="ibrav(8, a; celldm2=b_a, celldm3=c_a)"),
        9 => (name="Orthorhombic base-centered", required=(:celldm2, :celldm3), example="ibrav(9, a; celldm2=b_a, celldm3=c_a)"),
        10 => (name="Orthorhombic face-centered", required=(:celldm2, :celldm3), example="ibrav(10, a; celldm2=b_a, celldm3=c_a)"),
        11 => (name="Orthorhombic body-centered", required=(:celldm2, :celldm3), example="ibrav(11, a; celldm2=b_a, celldm3=c_a)"),
        12 => (name="Monoclinic P (unique axis c)", required=(:celldm2, :celldm3, :celldm4), example="ibrav(12, a; celldm2=b_a, celldm3=c_a, celldm4=cosγ)"),
        13 => (name="Monoclinic base-centered (unique axis c)", required=(:celldm2, :celldm3, :celldm4), example="ibrav(13, a; celldm2=b_a, celldm3=c_a, celldm4=cosγ)"),
        14 => (name="Triclinic", required=(:celldm2, :celldm3, :celldm4, :celldm5, :celldm6), example="ibrav(14, a; celldm2=b_a, celldm3=c_a, celldm4=cosbc, celldm5=cosac, celldm6=cosab)"),
        91 => (name="Orthorhombic base-centered A-type", required=(:celldm2, :celldm3), example="ibrav(91, a; celldm2=b_a, celldm3=c_a)"),
    )
    haskey(metadata, ibrav) || throw(ArgumentError("Unsupported QE `ibrav=$ibrav`."))
    return metadata[ibrav]
end

function _validate_ibrav_parameters(ibrav::Integer, a; celldm2=nothing, celldm3=nothing, celldm4=nothing, celldm5=nothing, celldm6=nothing)
    info = _ibrav_metadata(ibrav)
    a0 = _check_positive("a", a)

    params = Dict{Symbol,Float64}()
    if :celldm2 in info.required
        params[:celldm2] = _check_positive("celldm2", _require_parameter("celldm2", celldm2))
    elseif !isnothing(celldm2)
        params[:celldm2] = _check_positive("celldm2", celldm2)
    else
        params[:celldm2] = 1.0
    end

    if :celldm3 in info.required
        params[:celldm3] = _check_positive("celldm3", _require_parameter("celldm3", celldm3))
    elseif !isnothing(celldm3)
        params[:celldm3] = _check_positive("celldm3", celldm3)
    else
        params[:celldm3] = 1.0
    end

    for (name, value) in [(:celldm4, celldm4), (:celldm5, celldm5), (:celldm6, celldm6)]
        if name in info.required
            params[name] = _check_cosine(String(name), _require_parameter(String(name), value))
        elseif !isnothing(value)
            params[name] = _check_cosine(String(name), value)
        else
            params[name] = 0.0
        end
    end

    if ibrav in (5, -5)
        1 + 2params[:celldm4] >= 0 || throw(ArgumentError("`celldm4` is incompatible with a rhombohedral lattice because 1 + 2*cos(gamma) must be non-negative."))
    elseif ibrav in (12, 13)
        abs(params[:celldm4]) < 1 || throw(ArgumentError("`celldm4` must satisfy |cos(gamma)| < 1 for monoclinic cells."))
    elseif ibrav in (-12, -13)
        abs(params[:celldm5]) < 1 || throw(ArgumentError("`celldm5` must satisfy |cos(beta)| < 1 for monoclinic cells."))
    elseif ibrav == 14
        abs(params[:celldm4]) < 1 || throw(ArgumentError("`celldm4` must satisfy |cos(bc)| < 1 for triclinic cells."))
        abs(params[:celldm5]) < 1 || throw(ArgumentError("`celldm5` must satisfy |cos(ac)| < 1 for triclinic cells."))
        abs(params[:celldm6]) < 1 || throw(ArgumentError("`celldm6` must satisfy |cos(ab)| < 1 for triclinic cells."))
        radicand = 1 + 2params[:celldm4] * params[:celldm5] * params[:celldm6] - params[:celldm4]^2 - params[:celldm5]^2 - params[:celldm6]^2
        radicand >= 0 || throw(ArgumentError("The triclinic cosine triple is not geometrically valid."))
    end

    return (; a=a0, celldm2=params[:celldm2], celldm3=params[:celldm3], celldm4=params[:celldm4], celldm5=params[:celldm5], celldm6=params[:celldm6], name=info.name, required=info.required)
end

function _build_qe_lattice(ibrav::Integer, a::Real; celldm2=nothing, celldm3=nothing, celldm4=nothing, celldm5=nothing, celldm6=nothing)
    validated = _validate_ibrav_parameters(ibrav, a; celldm2, celldm3, celldm4, celldm5, celldm6)
    a0 = validated.a
    b = validated.celldm2 * a0
    c = validated.celldm3 * a0

    if ibrav == 0
        throw(ArgumentError("`ibrav=0` corresponds to a free cell. Use `Lattice(vectors)` or `Crystal(cell, atoms)` with explicit primitive vectors."))
    elseif ibrav == 1
        return [a0 0.0 0.0; 0.0 a0 0.0; 0.0 0.0 a0]
    elseif ibrav == 2
        return [-a0 / 2 0.0 -a0 / 2; 0.0 a0 / 2 a0 / 2; a0 / 2 a0 / 2 0.0]
    elseif ibrav == 3
        return [a0 / 2 -a0 / 2 -a0 / 2; a0 / 2 a0 / 2 -a0 / 2; a0 / 2 a0 / 2 a0 / 2]
    elseif ibrav == -3
        return [-a0 / 2 a0 / 2 a0 / 2; a0 / 2 -a0 / 2 a0 / 2; a0 / 2 a0 / 2 -a0 / 2]
    elseif ibrav == 4
        return [a0 -a0 / 2 0.0; 0.0 sqrt(3) * a0 / 2 0.0; 0.0 0.0 c]
    elseif ibrav == 5 || ibrav == -5
        cosγ = validated.celldm4
        tx = sqrt((1 - cosγ) / 2)
        ty = sqrt((1 - cosγ) / 6)
        tz = sqrt((1 + 2cosγ) / 3)

        if ibrav == 5
            return a0 .* [tx 0.0 -tx; -ty 2ty -ty; tz tz tz]
        end

        aprime = a0 / sqrt(3)
        u = tz - 2sqrt(2) * ty
        v = tz + sqrt(2) * ty
        return aprime .* [u v v; v u v; v v u]
    elseif ibrav == 6
        return [a0 0.0 0.0; 0.0 a0 0.0; 0.0 0.0 c]
    elseif ibrav == 7
        return [a0 / 2 a0 / 2 -a0 / 2; -a0 / 2 a0 / 2 -a0 / 2; c / 2 c / 2 c / 2]
    elseif ibrav == 8
        return [a0 0.0 0.0; 0.0 b 0.0; 0.0 0.0 c]
    elseif ibrav == 9
        return [a0 / 2 -a0 / 2 0.0; b / 2 b / 2 0.0; 0.0 0.0 c]
    elseif ibrav == -9
        return [a0 / 2 a0 / 2 0.0; -b / 2 b / 2 0.0; 0.0 0.0 c]
    elseif ibrav == 91
        return [a0 0.0 0.0; 0.0 b / 2 b / 2; 0.0 -c / 2 c / 2]
    elseif ibrav == 10
        return [a0 / 2 a0 / 2 0.0; 0.0 b / 2 b / 2; c / 2 0.0 c / 2]
    elseif ibrav == 11
        return [a0 / 2 -a0 / 2 -a0 / 2; b / 2 b / 2 -b / 2; c / 2 c / 2 c / 2]
    elseif ibrav == 12
        cosγ = validated.celldm4
        sinγ = sqrt(1 - cosγ^2)
        return [a0 b * cosγ 0.0; 0.0 b * sinγ 0.0; 0.0 0.0 c]
    elseif ibrav == -12
        cosβ = validated.celldm5
        sinβ = sqrt(1 - cosβ^2)
        return [a0 0.0 c * cosβ; 0.0 b 0.0; 0.0 0.0 c * sinβ]
    elseif ibrav == 13
        cosγ = validated.celldm4
        sinγ = sqrt(1 - cosγ^2)
        return [a0 / 2 b * cosγ a0 / 2; 0.0 b * sinγ 0.0; -c / 2 0.0 c / 2]
    elseif ibrav == -13
        cosβ = validated.celldm5
        sinβ = sqrt(1 - cosβ^2)
        return [a0 / 2 -a0 / 2 c * cosβ; b / 2 b / 2 0.0; 0.0 0.0 c * sinβ]
    elseif ibrav == 14
        cosα = validated.celldm4
        cosβ = validated.celldm5
        cosγ = validated.celldm6
        sinγ = sqrt(1 - cosγ^2)
        vy = c * (cosα - cosβ * cosγ) / sinγ
        vz = c * sqrt(1 + 2cosα * cosβ * cosγ - cosα^2 - cosβ^2 - cosγ^2) / sinγ
        return [a0 b * cosγ c * cosβ; 0.0 b * sinγ vy; 0.0 0.0 vz]
    else
        throw(ArgumentError("Unsupported QE `ibrav=$ibrav`."))
    end
end

"""
    Crystal{D} <: AbstractLattice{D}

Unitless, statically typed internal crystal representation used by the core
evaluation engine. The primitive real-space lattice vectors are stored as
columns of `lattice`, while the multi-atom basis is stored in fractional
coordinates.
"""
mutable struct Crystal{D} <: AbstractLattice{D}
    lattice::SMatrix{D,D,Float64}
    fractional_positions::Vector{SVector{D,Float64}}
    atomic_symbols::Vector{Symbol}
end

primitive_vectors(crystal::Crystal) = crystal.lattice

"""
    scaled_positions(crystal::Crystal)

Return a copy of the fractional basis positions stored in `crystal`.
"""
scaled_positions(crystal::Crystal) = copy(crystal.fractional_positions)

"""
    positions(crystal::Crystal)

Return the Cartesian basis positions of `crystal`.
"""
positions(crystal::Crystal) = cartesian_basis(crystal)

"""
    cartesian_basis(crystal::Crystal{D}) where {D}

Return the Cartesian basis positions associated with the fractional basis of
`crystal`.
"""
function cartesian_basis(crystal::Crystal{D}) where {D}
    lattice = primitive_vectors(crystal)
    return [lattice * position for position in crystal.fractional_positions]
end

"""
    Crystal(cell, positions, atomic_symbols; fractional=true, length_unit=u"Å")

Construct a `Crystal` directly from primitive vectors, basis positions, and
atomic symbols without going through `AtomsBase`. `cell` may be another
`AbstractLattice`, a square matrix with primitive vectors as columns, or a
tuple/vector of primitive vectors. Set `fractional=false` if the provided
positions are Cartesian coordinates in the same units as `cell`. Plain Julia
vectors and tuples are accepted and converted internally to `SVector`s.
"""
function Crystal(cell, positions::AbstractVector, atomic_symbols::AbstractVector; fractional::Bool=true, length_unit=u"Å")
    length(positions) == length(atomic_symbols) || throw(DimensionMismatch("`positions` and `atomic_symbols` must have the same length."))
    lattice = _coerce_lattice_matrix(cell, length_unit)
    D = size(lattice, 1)
    fractional_positions = SVector{D,Float64}[
        _normalize_fractional_position(position, lattice, Val(D); fractional, length_unit)
        for position in positions
    ]
    return Crystal{D}(lattice, fractional_positions, Symbol[_normalize_symbol(symbol) for symbol in atomic_symbols])
end

"""
    Crystal(cell, atoms; fractional=true, length_unit=u"Å")

Construct a `Crystal` directly from primitive vectors and a list of
`symbol => position` pairs. This provides an ASE-like creation path for the
internal unitless crystal representation. Positions may be given as plain Julia
vectors or tuples.
"""
function Crystal(cell, atoms::AbstractVector{<:Pair}; fractional::Bool=true, length_unit=u"Å")
    return Crystal(
        cell,
        [last(atom) for atom in atoms],
        [first(atom) for atom in atoms];
        fractional,
        length_unit,
    )
end

"""
    Crystal(system::AbstractSystem{D}; length_unit=u"Å") where {D}

Construct a unitless internal `Crystal` from an `AtomsBase.AbstractSystem`.
All `Unitful` lengths are converted to `length_unit` and stripped with
`ustrip` before storage so the downstream tight-binding engine remains
type-stable and `Float64`-based.
"""
function Crystal(system::AbstractSystem{D}; length_unit=u"Å") where {D}
    lattice_vectors = cell_vectors(system)
    lattice = _coerce_lattice_matrix(lattice_vectors, length_unit)
    lattice_inverse = inv(lattice)

    cartesian_positions = position(system, :)
    fractional_positions = Vector{SVector{D,Float64}}(undef, length(cartesian_positions))
    atomic_symbols = Vector{Symbol}(undef, length(cartesian_positions))

    for (index, cartesian_position) in pairs(cartesian_positions)
        unitless_position = _strip_length_vector(cartesian_position, length_unit, Val(D))
        fractional = lattice_inverse * unitless_position
        fractional_positions[index] = mod.(fractional, 1.0)
        atomic_symbols[index] = Symbol(atomic_symbol(system, index))
    end

    return Crystal{D}(lattice, fractional_positions, atomic_symbols)
end

"""
    append_atom!(crystal::Crystal{D}, atom::Pair; fractional=true, length_unit=u"Å") where {D}

Append one `symbol => position` entry to `crystal`. The new atom can be given
either in fractional coordinates (`fractional=true`) or in Cartesian
coordinates with the same units as the cell (`fractional=false`). Plain Julia
vectors and tuples are accepted.
"""
function append_atom!(crystal::Crystal{D}, atom::Pair; fractional::Bool=true, length_unit=u"Å") where {D}
    push!(crystal.atomic_symbols, _normalize_symbol(first(atom)))
    push!(crystal.fractional_positions, _normalize_fractional_position(last(atom), primitive_vectors(crystal), Val(D); fractional, length_unit))
    return crystal
end

"""
    append_atom!(crystal::Crystal, symbol, position; kwargs...)

Convenience overload of `append_atom!` that accepts the atomic symbol and
position as separate arguments.
"""
function append_atom!(crystal::Crystal, symbol, position; kwargs...)
    return append_atom!(crystal, symbol => position; kwargs...)
end

Base.push!(crystal::Crystal, atom::Pair) = append_atom!(crystal, atom)

"""
    set_scaled_positions!(crystal::Crystal{D}, positions) where {D}

Replace the basis of `crystal` with a new set of fractional coordinates.
The number of positions must match the number of stored atoms.
"""
function set_scaled_positions!(crystal::Crystal{D}, positions::AbstractVector) where {D}
    length(positions) == length(crystal.atomic_symbols) || throw(DimensionMismatch("The number of scaled positions must match the number of atoms in the crystal."))
    crystal.fractional_positions = SVector{D,Float64}[_strip_length_vector(position, u"Å", Val(D)) for position in positions]
    return crystal
end

"""
    set_positions!(crystal::Crystal{D}, positions; length_unit=u"Å") where {D}

Replace the basis of `crystal` with Cartesian positions. The positions are
converted to fractional coordinates with the current cell.
"""
function set_positions!(crystal::Crystal{D}, positions::AbstractVector; length_unit=u"Å") where {D}
    length(positions) == length(crystal.atomic_symbols) || throw(DimensionMismatch("The number of Cartesian positions must match the number of atoms in the crystal."))
    lattice = primitive_vectors(crystal)
    crystal.fractional_positions = SVector{D,Float64}[
        _normalize_fractional_position(position, lattice, Val(D); fractional=false, length_unit, wrap=false)
        for position in positions
    ]
    return crystal
end

"""
    set_cell!(crystal::Crystal, cell; scale_positions=false, length_unit=u"Å")

Update the primitive cell of `crystal`. By default this follows ASE's
`set_cell(...; scale_atoms=false)` behavior and keeps Cartesian atom positions
fixed while recomputing the fractional coordinates. Set
`scale_positions=true` to preserve the stored fractional basis instead.
"""
function set_cell!(crystal::Crystal{D}, cell; scale_positions::Bool=false, length_unit=u"Å") where {D}
    new_lattice = _coerce_lattice_matrix(cell, length_unit)
    size(new_lattice, 1) == D || throw(DimensionMismatch("The new cell must have the same dimensionality as the crystal."))
    old_cartesian = scale_positions ? nothing : cartesian_basis(crystal)
    crystal.lattice = new_lattice
    if !scale_positions
        set_positions!(crystal, old_cartesian; length_unit=u"Å")
    end
    return crystal
end

"""
    set_cell!(crystal::Crystal, ibrav_id::Integer, a; kwargs...)

Update the primitive cell of `crystal` using the QE `ibrav` convention.
Keyword arguments are forwarded to `ibrav(ibrav_id, a; kwargs...)`.
"""
function set_cell!(crystal::Crystal{3}, ibrav_id::Integer, a; kwargs...)
    return set_cell!(crystal, ibrav(ibrav_id, a; kwargs...))
end

"""
    ibrav(id::Integer)

Return metadata describing the QE `ibrav` choice, including its human-readable
name and construction requirements.
"""
function ibrav(id::Integer)
    info = _ibrav_metadata(id)
    keyword_docs = _ibrav_keyword_docs(id)
    required_keywords = IBravKeywordSpec[IBravKeywordSpec(name, keyword_docs[name]) for name in info.required]
    optional_keywords = IBravKeywordSpec[
        IBravKeywordSpec(name, keyword_docs[name])
        for name in setdiff(Tuple(keys(keyword_docs)), info.required)
    ]
    return IBravInfo(
        id,
        info.name,
        _ibrav_signature(id),
        _ibrav_positional_specs(id),
        required_keywords,
        optional_keywords,
        info.example,
    )
end

"""
    ibrav(id::Integer, a; kwargs...)

Construct a QE-style `Lattice{3}` using `id` as the `ibrav` selector. This is
the unified entry point corresponding to `qe_lattice(id, a; kwargs...)`.
"""
function ibrav(id::Integer, a; kwargs...)
    return qe_lattice(id, a; kwargs...)
end

"""
    qe_lattice(ibrav::Integer, a; celldm2=1.0, celldm3=1.0, celldm4=0.0, celldm5=0.0, celldm6=0.0)

Construct a three-dimensional `Lattice{3}` using Quantum ESPRESSO's `ibrav`
convention. The optional `celldm2:celldm6` keyword arguments follow QE's
definitions exactly:

- `celldm2 = b / a`
- `celldm3 = c / a`
- `celldm4`, `celldm5`, `celldm6` are the cosine parameters used by the chosen
  `ibrav`

The required parameters depend on the selected `ibrav` and are validated
before the primitive vectors are constructed.
"""
function qe_lattice(ibrav::Integer, a; celldm2=nothing, celldm3=nothing, celldm4=nothing, celldm5=nothing, celldm6=nothing)
    return Lattice(_build_qe_lattice(ibrav, a; celldm2, celldm3, celldm4, celldm5, celldm6))
end

"""
    cubic_p_lattice(a)

Construct the primitive simple-cubic lattice.
"""
cubic_p_lattice(a) = qe_lattice(1, a)

"""
    cubic_f_lattice(a)

Construct the face-centered cubic lattice using the QE `ibrav=2` convention.
"""
cubic_f_lattice(a) = qe_lattice(2, a)

"""
    cubic_i_lattice(a; symmetric_axis=false)

Construct the body-centered cubic lattice using the QE `ibrav=3` convention,
or `ibrav=-3` when `symmetric_axis=true`.
"""
cubic_i_lattice(a; symmetric_axis::Bool=false) = qe_lattice(symmetric_axis ? -3 : 3, a)

"""
    hexagonal_p_lattice(a; c_a)

Construct the hexagonal primitive lattice using the QE `ibrav=4` convention.
"""
hexagonal_p_lattice(a; c_a::Real) = qe_lattice(4, a; celldm3=c_a)

"""
    trigonal_r_lattice(a; cosγ, axis=:c)

Construct the rhombohedral primitive lattice using the QE `ibrav=5` or
`ibrav=-5` conventions.
"""
trigonal_r_lattice(a; cosγ::Real, axis::Symbol=:c) = qe_lattice(axis == :c ? 5 : -5, a; celldm4=cosγ)

"""
    tetragonal_p_lattice(a; c_a)

Construct the primitive tetragonal lattice using the QE `ibrav=6` convention.
"""
tetragonal_p_lattice(a; c_a::Real) = qe_lattice(6, a; celldm3=c_a)

"""
    tetragonal_i_lattice(a; c_a)

Construct the body-centered tetragonal lattice using the QE `ibrav=7`
convention.
"""
tetragonal_i_lattice(a; c_a::Real) = qe_lattice(7, a; celldm3=c_a)

"""
    orthorhombic_p_lattice(a; b_a, c_a)

Construct the primitive orthorhombic lattice using the QE `ibrav=8`
convention.
"""
orthorhombic_p_lattice(a; b_a::Real, c_a::Real) = qe_lattice(8, a; celldm2=b_a, celldm3=c_a)

"""
    orthorhombic_base_centered_lattice(a; b_a, c_a, setting=:default)

Construct a base-centered orthorhombic lattice using the QE `ibrav=9`,
`ibrav=-9`, or `ibrav=91` conventions.
"""
function orthorhombic_base_centered_lattice(a; b_a::Real, c_a::Real, setting::Symbol=:default)
    ibrav = setting == :default ? 9 : setting == :alternate ? -9 : setting == :a_type ? 91 : throw(ArgumentError("`setting` must be `:default`, `:alternate`, or `:a_type`."))
    return qe_lattice(ibrav, a; celldm2=b_a, celldm3=c_a)
end

"""
    orthorhombic_face_centered_lattice(a; b_a, c_a)

Construct the face-centered orthorhombic lattice using the QE `ibrav=10`
convention.
"""
orthorhombic_face_centered_lattice(a; b_a::Real, c_a::Real) = qe_lattice(10, a; celldm2=b_a, celldm3=c_a)

"""
    orthorhombic_body_centered_lattice(a; b_a, c_a)

Construct the body-centered orthorhombic lattice using the QE `ibrav=11`
convention.
"""
orthorhombic_body_centered_lattice(a; b_a::Real, c_a::Real) = qe_lattice(11, a; celldm2=b_a, celldm3=c_a)

"""
    monoclinic_p_lattice(a; b_a, c_a, unique_axis=:c, cosγ=0.0, cosβ=0.0)

Construct the primitive monoclinic lattice using the QE `ibrav=12` or
`ibrav=-12` conventions.
"""
function monoclinic_p_lattice(a; b_a::Real, c_a::Real, unique_axis::Symbol=:c, cosγ::Real=0.0, cosβ::Real=0.0)
    if unique_axis == :c
        return qe_lattice(12, a; celldm2=b_a, celldm3=c_a, celldm4=cosγ)
    elseif unique_axis == :b
        return qe_lattice(-12, a; celldm2=b_a, celldm3=c_a, celldm5=cosβ)
    end
    throw(ArgumentError("`unique_axis` must be `:b` or `:c`."))
end

"""
    monoclinic_base_centered_lattice(a; b_a, c_a, unique_axis=:c, cosγ=0.0, cosβ=0.0)

Construct the base-centered monoclinic lattice using the QE `ibrav=13` or
`ibrav=-13` conventions.
"""
function monoclinic_base_centered_lattice(a; b_a::Real, c_a::Real, unique_axis::Symbol=:c, cosγ::Real=0.0, cosβ::Real=0.0)
    if unique_axis == :c
        return qe_lattice(13, a; celldm2=b_a, celldm3=c_a, celldm4=cosγ)
    elseif unique_axis == :b
        return qe_lattice(-13, a; celldm2=b_a, celldm3=c_a, celldm5=cosβ)
    end
    throw(ArgumentError("`unique_axis` must be `:b` or `:c`."))
end

"""
    triclinic_lattice(a; b_a, c_a, cosbc, cosac, cosab)

Construct the triclinic lattice using the QE `ibrav=14` convention.
"""
triclinic_lattice(a; b_a::Real, c_a::Real, cosbc::Real, cosac::Real, cosab::Real) =
    qe_lattice(14, a; celldm2=b_a, celldm3=c_a, celldm4=cosbc, celldm5=cosac, celldm6=cosab)

# ---------------------------------------------------------
# 1D Lattices
# ---------------------------------------------------------
struct Lattice{D} <: AbstractLattice{D}
    vectors::SMatrix{D,D,Float64}
end

"""
    Lattice(vectors; length_unit=u"Å")

Construct a generic `Lattice{D}` from a square matrix or a tuple/vector of
primitive vectors stored as columns. Unitful lengths are converted to
`length_unit` and stripped on construction.
"""
function Lattice(vectors; length_unit=u"Å")
    matrix = _coerce_lattice_matrix(vectors, length_unit)
    D = size(matrix, 1)
    return Lattice{D}(matrix)
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
