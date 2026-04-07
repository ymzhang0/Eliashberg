# src/Interfaces/wannier90.jl

const WannierHopping = Tuple{Int, Int, SVector{3, Int}, ComplexF64}
const WannierPositionElement = Tuple{Int, Int, SVector{3, Int}, SVector{3, ComplexF64}}

function _read_next_nonempty_line(io::IO, filename::AbstractString)
    while !eof(io)
        line = readline(io)
        isempty(strip(line)) || return line
    end
    error("Unexpected end of file while reading $filename.")
end

function _parse_svector3_float(line::AbstractString, filename::AbstractString, context::AbstractString)
    fields = split(line)
    length(fields) == 3 || error("Expected 3 floating-point values for $context in $filename.")
    return SVector{3, Float64}(
        parse(Float64, fields[1]),
        parse(Float64, fields[2]),
        parse(Float64, fields[3]),
    )
end

function _parse_svector3_int(fields::AbstractVector{<:AbstractString}, filename::AbstractString, context::AbstractString)
    length(fields) == 3 || error("Expected 3 integer values for $context in $filename.")
    return SVector{3, Int}(
        parse(Int, fields[1]),
        parse(Int, fields[2]),
        parse(Int, fields[3]),
    )
end

function _read_wannier90_degeneracies(io::IO, filename::AbstractString, nrpts::Int)
    degeneracies = Vector{Int}(undef, nrpts)
    count = 0

    while count < nrpts
        for field in split(_read_next_nonempty_line(io, filename))
            count += 1
            count <= nrpts || error("Found more than nrpts degeneracy entries in $filename.")
            degeneracies[count] = parse(Int, field)
        end
    end

    return degeneracies
end

function _read_wannier90_hr_entries(io::IO, filename::AbstractString, num_wann::Int, degeneracies::Vector{Int}; tol::Float64=1e-6)
    hoppings = WannierHopping[]
    sizehint!(hoppings, length(degeneracies) * num_wann)

    for degen in degeneracies
        for _ in 1:(num_wann * num_wann)
            fields = split(_read_next_nonempty_line(io, filename))
            length(fields) == 7 || error("Expected 7 columns in the Hamiltonian block of $filename.")

            R = _parse_svector3_int(fields[1:3], filename, "R vector")
            m = parse(Int, fields[4])
            n = parse(Int, fields[5])
            t = ComplexF64(parse(Float64, fields[6]), parse(Float64, fields[7])) / degen

            abs(t) > tol && push!(hoppings, (m, n, R, t))
        end
    end

    return hoppings
end

function _read_wannier90_tb_hamiltonian(io::IO, filename::AbstractString, num_wann::Int, degeneracies::Vector{Int}; tol::Float64=1e-6)
    hoppings = WannierHopping[]
    sizehint!(hoppings, length(degeneracies) * num_wann)

    first_fields = split(_read_next_nonempty_line(io, filename))
    first_count = length(first_fields)
    entries_per_block = num_wann * num_wann

    if first_count == 3
        current_fields = first_fields
        for iR in eachindex(degeneracies)
            R = _parse_svector3_int(current_fields, filename, "R vector")
            inv_degen = 1.0 / degeneracies[iR]

            for _ in 1:entries_per_block
                fields = split(_read_next_nonempty_line(io, filename))
                length(fields) == 4 || error("Expected 4 columns in the grouped Hamiltonian block of $filename.")

                m = parse(Int, fields[1])
                n = parse(Int, fields[2])
                t = ComplexF64(parse(Float64, fields[3]), parse(Float64, fields[4])) * inv_degen

                abs(t) > tol && push!(hoppings, (m, n, R, t))
            end

            iR < length(degeneracies) && (current_fields = split(_read_next_nonempty_line(io, filename)))
        end
    elseif first_count == 7
        current_fields = first_fields
        for iR in eachindex(degeneracies)
            degen = degeneracies[iR]
            inv_degen = 1.0 / degen
            for entry_idx in 1:entries_per_block
                length(current_fields) == 7 || error("Expected 7 columns in the inline Hamiltonian block of $filename.")

                R = _parse_svector3_int(current_fields[1:3], filename, "R vector")
                m = parse(Int, current_fields[4])
                n = parse(Int, current_fields[5])
                t = ComplexF64(parse(Float64, current_fields[6]), parse(Float64, current_fields[7])) * inv_degen

                abs(t) > tol && push!(hoppings, (m, n, R, t))

                entry_idx < entries_per_block && (current_fields = split(_read_next_nonempty_line(io, filename)))
            end
            iR < length(degeneracies) && (current_fields = split(_read_next_nonempty_line(io, filename)))
        end
    else
        error("Unrecognized Hamiltonian block layout in $filename.")
    end

    return hoppings
end

function _read_wannier90_tb_positions(io::IO, filename::AbstractString, num_wann::Int, degeneracies::Vector{Int}; tol::Float64=1e-6)
    positions = WannierPositionElement[]
    sizehint!(positions, length(degeneracies) * num_wann)

    first_fields = split(_read_next_nonempty_line(io, filename))
    first_count = length(first_fields)
    entries_per_block = num_wann * num_wann

    if first_count == 3
        current_fields = first_fields
        for iR in eachindex(degeneracies)
            R = _parse_svector3_int(current_fields, filename, "R vector")
            inv_degen = 1.0 / degeneracies[iR]

            for _ in 1:entries_per_block
                fields = split(_read_next_nonempty_line(io, filename))
                length(fields) == 8 || error("Expected 8 columns in the grouped position block of $filename.")

                m = parse(Int, fields[1])
                n = parse(Int, fields[2])
                rvec = SVector{3, ComplexF64}(
                    ComplexF64(parse(Float64, fields[3]), parse(Float64, fields[4])) * inv_degen,
                    ComplexF64(parse(Float64, fields[5]), parse(Float64, fields[6])) * inv_degen,
                    ComplexF64(parse(Float64, fields[7]), parse(Float64, fields[8])) * inv_degen,
                )

                max(abs(rvec[1]), max(abs(rvec[2]), abs(rvec[3]))) > tol && push!(positions, (m, n, R, rvec))
            end

            iR < length(degeneracies) && (current_fields = split(_read_next_nonempty_line(io, filename)))
        end
    elseif first_count == 11
        current_fields = first_fields
        for iR in eachindex(degeneracies)
            degen = degeneracies[iR]
            inv_degen = 1.0 / degen
            for entry_idx in 1:entries_per_block
                length(current_fields) == 11 || error("Expected 11 columns in the inline position block of $filename.")

                R = _parse_svector3_int(current_fields[1:3], filename, "R vector")
                m = parse(Int, current_fields[4])
                n = parse(Int, current_fields[5])
                rvec = SVector{3, ComplexF64}(
                    ComplexF64(parse(Float64, current_fields[6]), parse(Float64, current_fields[7])) * inv_degen,
                    ComplexF64(parse(Float64, current_fields[8]), parse(Float64, current_fields[9])) * inv_degen,
                    ComplexF64(parse(Float64, current_fields[10]), parse(Float64, current_fields[11])) * inv_degen,
                )

                max(abs(rvec[1]), max(abs(rvec[2]), abs(rvec[3]))) > tol && push!(positions, (m, n, R, rvec))

                entry_idx < entries_per_block && (current_fields = split(_read_next_nonempty_line(io, filename)))
            end
            iR < length(degeneracies) && (current_fields = split(_read_next_nonempty_line(io, filename)))
        end
    else
        error("Unrecognized position block layout in $filename.")
    end

    return positions
end

"""
    parse_wannier90_hr(filename::String)

Read a `wannier90_hr.dat` file and return `(num_wann, hoppings)`, where each
hopping is stored as `(m, n, R, t)`.
"""
function parse_wannier90_hr(filename::String)
    open(filename, "r") do io
        readline(io) # Header/comment line.
        num_wann = parse(Int, strip(readline(io)))
        nrpts = parse(Int, strip(readline(io)))
        degeneracies = _read_wannier90_degeneracies(io, filename, nrpts)
        hoppings = _read_wannier90_hr_entries(io, filename, num_wann, degeneracies)
        return num_wann, hoppings
    end
end

"""
    parse_wannier90_tb(filename::String; periodicity=nothing)

Parse a `wannier90_tb.dat` file and return a `NamedTuple` with the lattice
vectors, an `AtomsBase.PeriodicCell` stored in `cell`, the number of Wannier orbitals, the sparse
Hamiltonian entries, and the sparse position-operator matrix elements.

The parser supports both the canonical Wannier90 grouped block layout and an
inline layout where each entry repeats its `R` vector.
"""
function parse_wannier90_tb(filename::String; periodicity=nothing)
    open(filename, "r") do io
        readline(io) # Header/comment line.

        lattice_vectors = (
            _parse_svector3_float(readline(io), filename, "lattice vector a1"),
            _parse_svector3_float(readline(io), filename, "lattice vector a2"),
            _parse_svector3_float(readline(io), filename, "lattice vector a3"),
        )
        atomsbase_periodicity = isnothing(periodicity) ? (true, true, true) :
            periodicity isa Bool ? ntuple(_ -> periodicity, 3) : Tuple(periodicity)
        cell = PeriodicCell(
            ;
            cell_vectors=ntuple(i -> lattice_vectors[i] .* u"Å", 3),
            periodicity=atomsbase_periodicity,
        )

        num_wann = parse(Int, strip(readline(io)))
        nrpts = parse(Int, strip(readline(io)))
        degeneracies = _read_wannier90_degeneracies(io, filename, nrpts)

        hoppings = _read_wannier90_tb_hamiltonian(io, filename, num_wann, degeneracies)
        position_matrices = _read_wannier90_tb_positions(io, filename, num_wann, degeneracies)

        return (
            cell = cell,
            periodicity = atomsbase_periodicity,
            lattice_vectors = lattice_vectors,
            num_wann = num_wann,
            hoppings = hoppings,
            position_matrices = position_matrices,
        )
    end
end

"""
    cell_from_wannier90_tb(filename::String)

Build an `AtomsBase.PeriodicCell` directly from the lattice vectors stored in a
`wannier90_tb.dat` file.
"""
function cell_from_wannier90_tb(filename::String; periodicity=nothing)
    return parse_wannier90_tb(filename; periodicity).cell
end

function cell_from_wannier90_tb(tb_data::NamedTuple; periodicity=nothing)
    hasproperty(tb_data, :cell) || error("The provided Wannier90 TB data does not contain a cell field.")
    if isnothing(periodicity)
        return tb_data.cell
    end
    return periodic_cell_from_wannier90_tb(tb_data; periodicity)
end

"""
    periodic_cell_from_wannier90_tb(filename::String; periodicity=nothing)

Build an `AtomsBase.PeriodicCell` directly from the lattice vectors stored in a
`wannier90_tb.dat` file. Use `periodicity=(true, true, false)` for slab
systems, for example.
"""
function periodic_cell_from_wannier90_tb(filename::String; periodicity=nothing)
    return parse_wannier90_tb(filename; periodicity).cell
end

function periodic_cell_from_wannier90_tb(tb_data::NamedTuple; periodicity=nothing)
    if hasproperty(tb_data, :cell) && isnothing(periodicity)
        return tb_data.cell
    end
    hasproperty(tb_data, :lattice_vectors) || error("The provided Wannier90 TB data does not contain lattice vectors.")
    atomsbase_periodicity = isnothing(periodicity) ? (true, true, true) :
        periodicity isa Bool ? ntuple(_ -> periodicity, 3) : Tuple(periodicity)
    return PeriodicCell(
        ;
        cell_vectors=ntuple(i -> tb_data.lattice_vectors[i] .* u"Å", 3),
        periodicity=atomsbase_periodicity,
    )
end

"""
    build_model_from_wannier90(filename::String, cell, EF::Float64)

Construct a `MultiOrbitalTightBinding` model from either a `wannier90_hr.dat`
or `wannier90_tb.dat` file and a given cell description. The preferred cell
inputs are `AtomsBase.PeriodicCell`, `AtomsBase.AbstractSystem`, or a primitive
vector matrix.
"""

function build_model_from_wannier90(filename::String, cell::AbstractMatrix{<:Number}, EF::Float64)
    if endswith(lowercase(basename(filename)), "_tb.dat")
        parsed = parse_wannier90_tb(filename)
        return MultiOrbitalTightBinding(parsed.cell, parsed.num_wann, parsed.hoppings, EF)
    elseif endswith(lowercase(basename(filename)), "_hr.dat")
        num_wann, hoppings = parse_wannier90_hr(filename)
        return MultiOrbitalTightBinding(cell, num_wann, hoppings, EF)
    else
        error("Unrecognized Wannier90 file type for $filename. Expected seedname_hr.dat or seedname_tb.dat suffix.")
    end
end

function build_model_from_wannier90(filename::String, crystal::Crystal, EF::Float64)
    return build_model_from_wannier90(filename, primitive_vectors(crystal), EF)
end

function build_model_from_wannier90(filename::String, cell::PeriodicCell, EF::Float64)
    if endswith(lowercase(basename(filename)), "_tb.dat")
        parsed = parse_wannier90_tb(filename; periodicity=periodicity(cell))
        return MultiOrbitalTightBinding(parsed.cell, parsed.num_wann, parsed.hoppings, EF)
    elseif endswith(lowercase(basename(filename)), "_hr.dat")
        num_wann, hoppings = parse_wannier90_hr(filename)
        return MultiOrbitalTightBinding(cell, num_wann, hoppings, EF)
    end
    error("Unrecognized Wannier90 file type for $filename. Expected seedname_hr.dat or seedname_tb.dat suffix.")
end

function build_model_from_wannier90(filename::String, system::AbstractSystem, EF::Float64)
    if endswith(lowercase(basename(filename)), "_tb.dat")
        parsed = parse_wannier90_tb(filename; periodicity=periodicity(system))
        return MultiOrbitalTightBinding(parsed.cell, parsed.num_wann, parsed.hoppings, EF)
    elseif endswith(lowercase(basename(filename)), "_hr.dat")
        num_wann, hoppings = parse_wannier90_hr(filename)
        return MultiOrbitalTightBinding(system, num_wann, hoppings, EF)
    end
    error("Unrecognized Wannier90 file type for $filename. Expected seedname_hr.dat or seedname_tb.dat suffix.")
end

"""
    build_model_from_wannier90(filename::String, EF::Float64)

Construct a `MultiOrbitalTightBinding` model directly from a `wannier90_tb.dat`
file using the standalone cell stored in the TB data.
"""
function build_model_from_wannier90(filename::String, EF::Float64, periodicity=nothing)
    endswith(lowercase(basename(filename)), "_tb.dat") || error("A standalone cell can only be reconstructed from a Wannier90 TB file.")
    parsed = parse_wannier90_tb(filename; periodicity)
    return MultiOrbitalTightBinding(parsed.cell, parsed.num_wann, parsed.hoppings, EF)
end
