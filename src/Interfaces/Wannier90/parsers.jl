const WannierHopping = Tuple{Int,Int,SVector{3,Int},ComplexF64}
const WannierPositionElement = Tuple{Int,Int,SVector{3,Int},SVector{3,ComplexF64}}


function _parse_svector3_float(line::AbstractString, filename::AbstractString, context::AbstractString)
    fields = split(line)
    length(fields) == 3 || error("Expected 3 floating-point values for $context in $filename.")
    return SVector{3,Float64}(
        parse(Float64, fields[1]),
        parse(Float64, fields[2]),
        parse(Float64, fields[3]),
    )
end

function _parse_svector3_int(fields::AbstractVector{<:AbstractString}, filename::AbstractString, context::AbstractString)
    length(fields) == 3 || error("Expected 3 integer values for $context in $filename.")
    return SVector{3,Int}(
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
        for _ in 1:(num_wann*num_wann)
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
                rvec = SVector{3,ComplexF64}(
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
                rvec = SVector{3,ComplexF64}(
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
        throw(ParsingError(filename, "Could not parse $context from token `$(strip(token))`: $(sprint(showerror, err))"))
    end

function _normalize_wannier90_label(label::AbstractString)
    stripped = strip(label)
    uppercase(stripped) in ("G", "GAMMA", "Γ") && return "Γ"
    return stripped
end

function _wannier90_dimension_hint(node_coordinates::AbstractVector{<:SVector{3,Float64}}; tol::Float64=1e-10)
    isempty(node_coordinates) && return 1
    reference = first(node_coordinates)
    varying_axes = count(axis -> any(abs(coords[axis] - reference[axis]) > tol for coords in node_coordinates), 1:3)
    return max(varying_axes, 1)
end

function _wannier90_band_stem(seedname::AbstractString)
    name = basename(String(seedname))
    if endswith(name, "_band.dat")
        return replace(name, r"_band\.dat$" => "")
    elseif endswith(name, ".dat")
        return replace(name, r"\.dat$" => "")
    end
    return name
end

_wannier90_band_data_filename(seedname::AbstractString) =
    endswith(String(seedname), ".dat") ? basename(String(seedname)) : "$(_wannier90_band_stem(seedname))_band.dat"

"""
    parse_wannier90_hr(filename::String)

Read a `wannier90_hr.dat` file and return `(num_wann, hoppings)`, where each
hopping is stored as `(m, n, R, t)`.
"""
function parse_wannier90_hr(filename::String)
    return with_stage_log(
        "Parse Wannier90 HR";
        context=(filename=filename,),
        summarize_result=result -> (num_wann=result[1], n_hoppings=length(result[2])),
    ) do
        open(filename, "r") do io
            readline(io)
            num_wann = parse(Int, strip(readline(io)))
            nrpts = parse(Int, strip(readline(io)))
            degeneracies = _read_wannier90_degeneracies(io, filename, nrpts)
            hoppings = _read_wannier90_hr_entries(io, filename, num_wann, degeneracies)
            return num_wann, hoppings
        end
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
    return with_stage_log(
        "Parse Wannier90 TB";
        context=(filename=filename, periodicity=periodicity),
        summarize_result=result -> (
            cell=result.cell,
            num_wann=result.num_wann,
            n_hoppings=length(result.hoppings),
            n_position_matrices=length(result.position_matrices),
        ),
    ) do
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
                cell=cell,
                periodicity=atomsbase_periodicity,
                num_wann=num_wann,
                hoppings=hoppings,
                position_matrices=position_matrices,
            )
        end
    end
end

"""
    parse_wannier90_band_dat(dir::String, prefix::String, bands_file::String = "\$(prefix)_band.dat")

Parse a Wannier90 interpolated `*_band.dat` file and its companion metadata to return 
a `BandStructureData` object. It requires a companion Quantum ESPRESSO XML file 
(`<prefix>.xml`) in the same directory for crystal structure information.

Arguments:
- `dir`: The directory containing the output files.
- `prefix`: The prefix used for the calculation (used to find `<prefix>.xml`).
- `seedname`: The seedname used for the calculation (used to find `<seedname>_band.dat`).
- `branch_gap_factor`: Split consecutive k-path branches when the adjacent k-point
  spacing in `<seedname>_band.kpt` exceeds this multiple of the median spacing.
- `branch_gap_threshold`: Optional absolute k-point spacing threshold for branch splitting.
"""
function parse_wannier90_band_dat(
    dir::String,
    prefix::String,
    seedname::String;
    branch_gap_factor::Real=5.0,
    branch_gap_threshold::Union{Nothing,Real}=nothing,
)
    xml_path = joinpath(dir, "$prefix.xml")
    if !isfile(xml_path)
        error("Quantum ESPRESSO XML file not found at: $xml_path. Structure information is required for Wannier90 bands.")
    end

    return with_stage_log(
        "Parse Wannier90 bands with XML metadata";
        context=(dir=dir, prefix=prefix, seedname=seedname),
    ) do
        # 1. Load Structure and Metadata in one pass
        xml_dict = parse_quantum_espresso_xml(xml_path)
        system = build_atoms_from_xml(xml_dict)
        recip_basis_cart = reciprocal_vectors(primitive_vectors(system))
        basis = [SVector{3,Float64}(recip_basis_cart[:, i]) for i in 1:3]

        # 2. Extract Fermi Energy from XML metadata
        root_key = first(filter(k -> endswith(string(k), "espresso"), keys(xml_dict)))
        root = xml_dict[root_key]

        fermi_level = 0.0
        if haskey(root, "output") && haskey(root["output"], "band_structure")
            bs_node = root["output"]["band_structure"]
            if haskey(bs_node, "fermi_energy")
                fermi_val = bs_node["fermi_energy"]
                fermi_str = fermi_val isa AbstractDict ? get(fermi_val, "", get(fermi_val, "_content", "0.0")) : fermi_val
                fermi_level = parse(Float64, strip(fermi_str)) * ustrip(Ha2eV)
            end
        end

        # 3. Parse raw band data (distances and energies)
        band_stem = _wannier90_band_stem(seedname)
        dat_path = joinpath(dir, _wannier90_band_data_filename(seedname))
        raw = _parse_wannier90_raw_bands(dat_path)

        # 4. Handle Labels and Coordinates via labelinfo.dat
        # Standard Wannier90 generates seedname.labelinfo.dat when bands are plotted
        labelinfo_path = joinpath(dir, "$(band_stem)_band.labelinfo.dat")
        labels = Dict{Int,Symbol}()
        if isfile(labelinfo_path)
            info = parse_wannier90_labelinfo(labelinfo_path)

            for (idx, label, distance) in zip(info.node_indices, info.node_labels, info.node_distances)
                1 <= idx <= length(raw.distances) || throw(BoundsError(raw.distances, idx))
                isapprox(raw.distances[idx], distance; atol=1e-6, rtol=1e-6) ||
                    error("Wannier90 label distance mismatch at index $idx: file gives $distance but the band path contains $(raw.distances[idx]).")
                labels[idx] = Symbol(label)
            end
        end

        points = [SVector{3,Float64}(d, 0.0, 0.0) for d in raw.distances]
        kpoints_path = joinpath(dir, "$(band_stem)_band.kpt")
        gap_points = if isfile(kpoints_path)
            parsed_kpoints = parse_wannier90_kpoints(kpoints_path)
            length(parsed_kpoints.kpoints) == length(points) ||
                error("Wannier90 k-point file $kpoints_path contains $(length(parsed_kpoints.kpoints)) samples, but $dat_path contains $(length(points)) path samples.")
            _qe_cartesianize_kpoints(parsed_kpoints.kpoints, basis)
        else
            points
        end
        branched = _branch_kpath_at_large_point_gaps(
            points,
            labels;
            gap_points=gap_points,
            gap_factor=branch_gap_factor,
            gap_threshold=branch_gap_threshold,
        )
        kpath = KPath{3}(branched.branches, branched.labels, basis, Ref(Brillouin.CARTESIAN))

        return BandStructureData(
            kpath=kpath,
            bands=raw.bands,
            num_bands=raw.num_bands,
            fermi_level=fermi_level
        )
    end
end

# Internal helper to preserve original parsing logic
function _parse_wannier90_raw_bands(filename::String)
    blocks = Vector{Tuple{Vector{Float64},Vector{Float64}}}()
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

    ref_dists, _ = first(blocks)
    num_kpoints = length(ref_dists)
    num_bands = length(blocks)
    bands = Matrix{Float64}(undef, num_kpoints, num_bands)
    for b in 1:num_bands
        bands[:, b] = blocks[b][2]
    end

    return (distances=ref_dists, bands=bands, num_bands=num_bands, num_kpoints=num_kpoints)
end

"""
    parse_wannier90_kpoints(filename::String)

Parse a Wannier90 `*.kpt` file and return the fractional path samples together
with their weights. The standard Wannier90 format stores the number of k-points
on the first line followed by one four-column row per sample.
"""
function parse_wannier90_kpoints(filename::String)
    return with_stage_log(
        "Parse Wannier90 k-points";
        context=(filename=filename,),
        summarize_result=result -> (num_kpoints=result.num_kpoints, weight_sum=_safe_weight_sum(result.weights)),
    ) do
        open(filename, "r") do io
            num_kpoints = parse(Int, strip(_read_next_nonempty_line(io, filename)))
            kpoints = Vector{SVector{3,Float64}}()
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
                    SVector{3,Float64}(
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
                kpoints=kpoints,
                weights=weights,
                num_kpoints=num_kpoints,
            )
        end
    end
end

"""
    parse_wannier90_labelinfo(filename::String)

Parse a Wannier90 `*.labelinfo.dat` file and return the symmetry-point labels,
their 1-based indices along the path, cumulative path distances, and the raw
fractional coordinates recorded by Wannier90.
"""
function parse_wannier90_labelinfo(filename::String)
    return with_stage_log(
        "Parse Wannier90 labelinfo";
        context=(filename=filename,),
        summarize_result=result -> (n_nodes=length(result.node_labels), dimension=result.dimension),
    ) do
        node_labels = String[]
        node_indices = Int[]
        node_distances = Float64[]
        node_coordinates = SVector{3,Float64}[]

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
                    SVector{3,Float64}(
                        _parse_wannier90_number(fields[4], filename, "label kx"),
                        _parse_wannier90_number(fields[5], filename, "label ky"),
                        _parse_wannier90_number(fields[6], filename, "label kz"),
                    ),
                )
            end
        end

        isempty(node_labels) && error("Wannier90 labelinfo file $filename is empty.")

        return (
            node_labels=node_labels,
            node_indices=node_indices,
            node_distances=node_distances,
            node_coordinates=node_coordinates,
            dimension=_wannier90_dimension_hint(node_coordinates),
        )
    end
end
