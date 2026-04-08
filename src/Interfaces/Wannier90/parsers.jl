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

_parse_wannier90_number(token::AbstractString, filename::AbstractString, context::AbstractString) =
    try
        parse(Float64, replace(strip(token), r"[dD]" => "e"))
    catch err
        throw(ArgumentError("Could not parse $context in $filename from token `$(strip(token))`: $(sprint(showerror, err))"))
    end

function _normalize_wannier90_label(label::AbstractString)
    stripped = strip(label)
    uppercase(stripped) in ("G", "GAMMA", "Γ") && return "Γ"
    return stripped
end

function _wannier90_dimension_hint(node_coordinates::AbstractVector{<:SVector{3, Float64}}; tol::Float64=1e-10)
    isempty(node_coordinates) && return 1
    reference = first(node_coordinates)
    varying_axes = count(axis -> any(abs(coords[axis] - reference[axis]) > tol for coords in node_coordinates), 1:3)
    return max(varying_axes, 1)
end

"""
    parse_wannier90_hr(filename::String)

Read a `wannier90_hr.dat` file and return `(num_wann, hoppings)`, where each
hopping is stored as `(m, n, R, t)`.
"""
function parse_wannier90_hr(filename::String)
    open(filename, "r") do io
        readline(io)
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
        readline(io)

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
    parse_wannier90_band_dat(filename::String)

Parse a Wannier90 interpolated `*_band.dat` file and return a `NamedTuple` with
the cumulative path-distance coordinate and a dense `(num_kpoints, num_bands)`
energy matrix. The parser expects the standard block layout where each band is
written as a separate two-column section separated by blank lines.
"""
function parse_wannier90_band_dat(filename::String)
    blocks = Vector{Tuple{Vector{Float64}, Vector{Float64}}}()
    current_distances = Float64[]
    current_energies = Float64[]

    open(filename, "r") do io
        while !eof(io)
            line = readline(io)
            if isempty(strip(line))
                if !isempty(current_distances)
                    push!(blocks, (current_distances, current_energies))
                    current_distances = Float64[]
                    current_energies = Float64[]
                end
                continue
            end

            fields = split(line)
            length(fields) == 2 || error("Expected 2 columns in Wannier90 band file $filename.")
            push!(current_distances, _parse_wannier90_number(fields[1], filename, "path distance"))
            push!(current_energies, _parse_wannier90_number(fields[2], filename, "band energy"))
        end
    end

    isempty(current_distances) || push!(blocks, (current_distances, current_energies))
    isempty(blocks) && error("Wannier90 band file $filename is empty.")

    reference_distances, _ = first(blocks)
    num_kpoints = length(reference_distances)
    num_bands = length(blocks)
    num_kpoints > 0 || error("Wannier90 band file $filename does not contain any k-point samples.")

    bands = Matrix{Float64}(undef, num_kpoints, num_bands)
    bands[:, 1] = first(blocks)[2]

    for band_idx in 2:num_bands
        distances, energies = blocks[band_idx]
        length(distances) == num_kpoints || error("All Wannier90 band blocks in $filename must contain the same number of k-points.")
        for idx in eachindex(distances)
            isapprox(distances[idx], reference_distances[idx]; atol=1e-8, rtol=1e-8) ||
                error("Inconsistent path-distance grid across band blocks in $filename.")
        end
        bands[:, band_idx] = energies
    end

    return (
        distances = copy(reference_distances),
        bands = bands,
        num_kpoints = num_kpoints,
        num_bands = num_bands,
    )
end

"""
    parse_wannier90_kpoints(filename::String)

Parse a Wannier90 `*.kpt` file and return the fractional path samples together
with their weights. The standard Wannier90 format stores the number of k-points
on the first line followed by one four-column row per sample.
"""
function parse_wannier90_kpoints(filename::String)
    open(filename, "r") do io
        num_kpoints = parse(Int, strip(_read_next_nonempty_line(io, filename)))
        kpoints = Vector{SVector{3, Float64}}()
        weights = Float64[]
        sizehint!(kpoints, num_kpoints)
        sizehint!(weights, num_kpoints)

        while !eof(io)
            line = strip(readline(io))
            isempty(line) && continue

            fields = split(line)
            length(fields) >= 3 || error("Expected at least 3 columns in Wannier90 k-point file $filename.")

            push!(
                kpoints,
                SVector{3, Float64}(
                    _parse_wannier90_number(fields[1], filename, "k-point x"),
                    _parse_wannier90_number(fields[2], filename, "k-point y"),
                    _parse_wannier90_number(fields[3], filename, "k-point z"),
                ),
            )
            push!(weights, length(fields) >= 4 ? _parse_wannier90_number(fields[4], filename, "k-point weight") : 1.0)
        end

        length(kpoints) == num_kpoints ||
            error("Wannier90 k-point file $filename declares $num_kpoints samples but contains $(length(kpoints)).")

        return (
            kpoints = kpoints,
            weights = weights,
            num_kpoints = num_kpoints,
        )
    end
end

"""
    parse_wannier90_labelinfo(filename::String)

Parse a Wannier90 `*.labelinfo.dat` file and return the symmetry-point labels,
their 1-based indices along the path, cumulative path distances, and the raw
fractional coordinates recorded by Wannier90.
"""
function parse_wannier90_labelinfo(filename::String)
    node_labels = String[]
    node_indices = Int[]
    node_distances = Float64[]
    node_coordinates = SVector{3, Float64}[]

    open(filename, "r") do io
        while !eof(io)
            line = strip(readline(io))
            isempty(line) && continue

            fields = split(line)
            length(fields) >= 6 || error("Expected at least 6 columns in Wannier90 labelinfo file $filename.")

            push!(node_labels, _normalize_wannier90_label(fields[1]))
            push!(node_indices, parse(Int, fields[2]))
            push!(node_distances, _parse_wannier90_number(fields[3], filename, "label distance"))
            push!(
                node_coordinates,
                SVector{3, Float64}(
                    _parse_wannier90_number(fields[4], filename, "label kx"),
                    _parse_wannier90_number(fields[5], filename, "label ky"),
                    _parse_wannier90_number(fields[6], filename, "label kz"),
                ),
            )
        end
    end

    isempty(node_labels) && error("Wannier90 labelinfo file $filename is empty.")

    return (
        node_labels = node_labels,
        node_indices = node_indices,
        node_distances = node_distances,
        node_coordinates = node_coordinates,
        dimension = _wannier90_dimension_hint(node_coordinates),
    )
end
