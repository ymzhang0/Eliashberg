const QE_BOHR_TO_ANGSTROM = 0.529177210903

_qe_bohr_to_angstrom(value::Real) = Float64(value) * QE_BOHR_TO_ANGSTROM

function _parse_qe_number(token::AbstractString, filename::AbstractString, context::AbstractString)
    try
        return parse(Float64, replace(strip(token), r"[dD]" => "e"))
    catch err
        throw(ParsingError(filename, "Could not parse $context from token `$(strip(token))`: $(sprint(showerror, err))"))
    end
end

function _parse_qe_int(token::AbstractString, filename::AbstractString, context::AbstractString)
    try
        return parse(Int, strip(token))
    catch err
        throw(ParsingError(filename, "Could not parse $context from token `$(strip(token))`: $(sprint(showerror, err))"))
    end
end

"""
    parse_quantum_espresso_cell(filename::String; periodicity=nothing)

Parse a primitive cell from a Quantum ESPRESSO input or output file using AtomsIO.
"""
function parse_quantum_espresso_cell(filename::String; periodicity=nothing)
    system = load_system(filename)
    if periodicity !== nothing
        # Update periodicity if explicitly requested
        pbc = periodicity isa Bool ? ntuple(_ -> periodicity, 3) : Tuple(periodicity)
        system = AbstractSystem(system; periodicity=pbc)
    end
    return system
end

# ---------------------------------------------------------
# Quantum ESPRESSO Bands Parsing
# ---------------------------------------------------------


function _parse_qe_bands_header(line::AbstractString, filename::AbstractString)
    nbnd_match = match(r"nbnd=\s*(\d+)", line)
    nks_match = match(r"nks=\s*(\d+)", line)

    nbnd_match === nothing && error("Could not find `nbnd` in Quantum ESPRESSO bands header of $filename.")
    nks_match === nothing && error("Could not find `nks` in Quantum ESPRESSO bands header of $filename.")

    return parse(Int, nbnd_match.captures[1]), parse(Int, nks_match.captures[1])
end

function _parse_qe_kpoint(line::AbstractString, filename::AbstractString)
    fields = split(line)
    length(fields) == 3 || error("Expected 3 coordinates for a k-point in $filename.")

    return SVector{3,Float64}(
        parse(Float64, fields[1]),
        parse(Float64, fields[2]),
        parse(Float64, fields[3]),
    )
end

function _qe_root_node(xml_dict::AbstractDict)
    root_key = first(filter(k -> endswith(string(k), "espresso"), keys(xml_dict)))
    return xml_dict[root_key]
end

function _qe_atomic_structure_node(root::AbstractDict)
    if haskey(root, "output") && haskey(root["output"], "atomic_structure")
        return root["output"]["atomic_structure"]
    elseif haskey(root, "input") && haskey(root["input"], "atomic_structure")
        return root["input"]["atomic_structure"]
    end

    error("Could not find <atomic_structure> in Quantum ESPRESSO XML dictionary.")
end

function _qe_alat_angstrom(xml_dict::AbstractDict)
    root = _qe_root_node(xml_dict)
    structure = _qe_atomic_structure_node(root)
    haskey(structure, :alat) || error("Could not find Quantum ESPRESSO `alat` in XML atomic_structure metadata.")
    return _qe_bohr_to_angstrom(_parse_qe_number(string(structure[:alat]), "Quantum ESPRESSO XML", "alat"))
end

function _qe_path_node_indices(kpoints::AbstractVector{<:SVector{3,<:Real}}; atol::Real=1e-8, direction_atol::Real=1e-5)
    length(kpoints) <= 2 && return collect(eachindex(kpoints))

    nodes = [1]
    for idx in 2:(length(kpoints)-1)
        incoming = kpoints[idx] - kpoints[idx-1]
        outgoing = kpoints[idx+1] - kpoints[idx]
        if norm(incoming) <= atol || norm(outgoing) <= atol
            push!(nodes, idx)
            continue
        end

        direction_change = norm(incoming / norm(incoming) - outgoing / norm(outgoing))
        direction_change > direction_atol && push!(nodes, idx)
    end
    push!(nodes, length(kpoints))
    return unique(nodes)
end

function _qe_standard_cell_dataset(system)
    dataset = Spglib.get_dataset(Spglib.SpglibCell(system))
    std_lattice = Matrix{Float64}(dataset.std_lattice)
    std_positions = [SVector{3,Float64}(pos) for pos in dataset.std_positions]
    std_types = Vector{Int}(dataset.std_types)
    std_dataset = Spglib.get_dataset(Spglib.SpglibCell(std_lattice, std_positions, std_types))
    return dataset, std_lattice, std_dataset
end

function _qe_high_symmetry_candidates_tpiba(system, tpiba::Real)
    dataset, std_lattice, std_dataset = _qe_standard_cell_dataset(system)
    direct_basis = [SVector{3,Float64}(std_lattice[:, axis]) for axis in 1:3]
    kpath = Brillouin.irrfbz_path(Int(dataset.spacegroup_number), direct_basis)
    cartesian_path = Brillouin.cartesianize(kpath)

    reciprocal_basis = reciprocal_vectors(std_lattice)
    translation_basis_tpiba = reciprocal_vectors(Matrix{Float64}(dataset.primitive_lattice)) ./ tpiba
    reciprocal_basis_inv = inv(reciprocal_basis)
    label_directions = Dict{String,Vector{SVector{3,Float64}}}()
    for path_labels in kpath.paths
        for pair in zip(path_labels[1:end-1], path_labels[2:end])
            first_label = _normalize_wannier90_label(String(pair[1]))
            second_label = _normalize_wannier90_label(String(pair[2]))
            first_point = SVector{3,Float64}(cartesian_path.points[pair[1]] ./ tpiba)
            second_point = SVector{3,Float64}(cartesian_path.points[pair[2]] ./ tpiba)
            direction = second_point - first_point
            norm(direction) > 1e-12 || continue
            push!(get!(label_directions, first_label, SVector{3,Float64}[]), direction)
            push!(get!(label_directions, second_label, SVector{3,Float64}[]), -direction)
        end
    end

    candidates = Tuple{String,SVector{3,Float64},Vector{SVector{3,Float64}}}[]
    translations = [SVector{3,Float64}(i, j, k) for i in -2:2 for j in -2:2 for k in -2:2]

    for (label, point_cartesian) in cartesian_path.points
        label_string = _normalize_wannier90_label(String(label))
        point_tpiba = SVector{3,Float64}(point_cartesian ./ tpiba)

        for rotation in std_dataset.rotations
            rotation_matrix = Matrix{Float64}(rotation)
            reciprocal_rotation = reciprocal_basis * transpose(inv(rotation_matrix)) * reciprocal_basis_inv
            candidate = SVector{3,Float64}(reciprocal_rotation * point_tpiba)
            candidate_directions = [
                SVector{3,Float64}(reciprocal_rotation * direction)
                for direction in get(label_directions, label_string, SVector{3,Float64}[])
            ]

            for translation in translations
                translated_candidate = SVector{3,Float64}(candidate + translation_basis_tpiba * translation)
                if !any(existing_label == label_string && norm(existing_point - translated_candidate) < 1e-8 for (existing_label, existing_point, _) in candidates)
                    push!(candidates, (label_string, translated_candidate, candidate_directions))
                end
            end
        end
    end

    return candidates
end

function _qe_match_high_symmetry_label(
    kpoint_tpiba::SVector{3,<:Real},
    candidates::AbstractVector{<:Tuple{String,SVector{3,Float64},Vector{SVector{3,Float64}}}};
    tangent::Union{Nothing,SVector{3,Float64}}=nothing,
    atol::Real=1e-5,
)
    matches = Tuple{String,Float64,Float64}[]
    for (label, candidate, directions) in candidates
        distance = norm(kpoint_tpiba - candidate)
        distance <= atol || continue

        direction_score = 0.0
        if tangent !== nothing && norm(tangent) > 1e-12
            tangent_unit = tangent / norm(tangent)
            for direction in directions
                norm(direction) > 1e-12 || continue
                direction_score = max(direction_score, abs(dot(tangent_unit, direction / norm(direction))))
            end
        end

        push!(matches, (label, distance, direction_score))
    end
    isempty(matches) && return nothing

    labels = unique(first.(matches))
    length(labels) == 1 && return only(labels)
    label_rank(label::AbstractString) = get(Dict("Γ" => 0, "G" => 0, "L" => 1, "X" => 2, "K" => 3, "W" => 4, "U" => 5), label, 100)

    if tangent !== nothing
        best_by_label = Dict{String,Tuple{Float64,Float64}}()
        for (label, distance, direction_score) in matches
            current = get(best_by_label, label, (Inf, -Inf))
            if direction_score > current[2] + 1e-8 || (abs(direction_score - current[2]) <= 1e-8 && distance < current[1])
                best_by_label[label] = (distance, direction_score)
            end
        end

        ranked = sort!(collect(best_by_label); by=entry -> (-entry[2][2], label_rank(entry[1]), entry[2][1]))
        best_label, (best_distance, best_score) = first(ranked)
        if best_score > 1 - 1e-6
            competing = [
                (label, distance, score)
                for (label, (distance, score)) in ranked[2:end]
                if label_rank(label) == label_rank(best_label) && abs(score - best_score) <= 1e-8 && abs(distance - best_distance) <= atol
            ]
            isempty(competing) && return best_label
        end
    end

    sort!(matches; by=match -> (match[2], -match[3], label_rank(match[1])))
    best_label, best_distance, best_score = first(matches)
    competing = [(label, distance, score) for (label, distance, score) in matches if label != best_label && label_rank(label) == label_rank(best_label) && abs(distance - best_distance) <= atol && abs(score - best_score) <= 1e-8]
    isempty(competing) || error("Ambiguous high-symmetry label match for QE k-point $kpoint_tpiba: $matches")
    return best_label
end

function _qe_path_tangent(kpoints::AbstractVector{<:SVector{3,<:Real}}, idx::Integer)
    if length(kpoints) == 1
        return nothing
    elseif idx == firstindex(kpoints)
        tangent = kpoints[idx + 1] - kpoints[idx]
    elseif idx == lastindex(kpoints)
        tangent = kpoints[idx] - kpoints[idx - 1]
    else
        tangent = kpoints[idx + 1] - kpoints[idx - 1]
    end

    norm(tangent) > 1e-12 || return nothing
    return SVector{3,Float64}(tangent)
end

function _qe_labels_from_high_symmetry(
    kpoints_tpiba::AbstractVector{<:SVector{3,<:Real}},
    system,
    tpiba::Real;
    atol::Real=1e-5,
)
    candidates = _qe_high_symmetry_candidates_tpiba(system, tpiba)
    labels = Dict{Int,Symbol}()
    node_indices = _qe_path_node_indices(kpoints_tpiba)

    for idx in eachindex(kpoints_tpiba)
        kpoint = kpoints_tpiba[idx]
        label = _qe_match_high_symmetry_label(kpoint, candidates; tangent=_qe_path_tangent(kpoints_tpiba, idx), atol)
        label === nothing || (labels[idx] = Symbol(label))
    end

    unmatched_nodes = [idx for idx in node_indices if !haskey(labels, idx)]
    if !isempty(unmatched_nodes)
        details = join(["$idx => $(kpoints_tpiba[idx])" for idx in unmatched_nodes], ", ")
        error("Could not match all QE band-path nodes to generated high-symmetry points. Unmatched nodes: $details")
    end

    return labels
end

function _kpath_median(values::AbstractVector{<:Real})
    isempty(values) && return 0.0

    sorted_values = sort!(Float64.(values))
    mid = length(sorted_values) ÷ 2
    return isodd(length(sorted_values)) ? sorted_values[mid + 1] : (sorted_values[mid] + sorted_values[mid + 1]) / 2
end

function _branch_kpath_at_large_point_gaps(
    points::AbstractVector{<:SVector{D,<:Real}},
    labels::Dict{Int,Symbol},
    ;
    gap_points::AbstractVector=points,
    gap_factor::Real=5.0,
    gap_threshold::Union{Nothing,Real}=nothing,
) where {D}
    path_points = [SVector{D,Float64}(point) for point in points]
    gap_reference_points = [Float64.(point) for point in gap_points]
    length(path_points) == length(gap_reference_points) ||
        throw(DimensionMismatch("Path points and gap-reference points must have the same length."))

    isempty(path_points) && return (branches=[path_points], labels=[Dict{Int,Symbol}()])
    length(path_points) == 1 && return (branches=[path_points], labels=[copy(labels)])

    adjacent_distances = [norm(gap_reference_points[idx + 1] - gap_reference_points[idx]) for idx in 1:(length(gap_reference_points)-1)]
    positive_distances = [distance for distance in adjacent_distances if distance > 1e-12]
    isempty(positive_distances) && return (branches=[path_points], labels=[copy(labels)])

    typical_step = _kpath_median(positive_distances)
    threshold = gap_threshold === nothing ? gap_factor * typical_step : Float64(gap_threshold)

    split_after = Int[]
    for (idx, distance) in enumerate(adjacent_distances)
        distance > threshold && push!(split_after, idx)
    end

    isempty(split_after) && return (branches=[path_points], labels=[copy(labels)])

    starts = [1; split_after .+ 1]
    stops = [split_after; length(path_points)]
    branches = Vector{SVector{D,Float64}}[]
    branch_labels = Dict{Int,Symbol}[]

    for (start_idx, stop_idx) in zip(starts, stops)
        push!(branches, path_points[start_idx:stop_idx])
        local_labels = Dict{Int,Symbol}()
        for (idx, label) in labels
            if start_idx <= idx <= stop_idx
                local_labels[idx - start_idx + 1] = label
            end
        end
        push!(branch_labels, local_labels)
    end

    return (branches=branches, labels=branch_labels)
end

function _parse_qe_band_block(io::IO, filename::AbstractString, num_bands::Int)
    energies = Vector{Float64}(undef, num_bands)
    count = 0

    while count < num_bands
        eof(io) && error("Unexpected end of file while reading band energies from $filename.")

        line = strip(readline(io))
        isempty(line) && continue

        for field in split(line)
            count += 1
            count <= num_bands || error("Found more than $num_bands energies for a single k-point in $filename.")
            energies[count] = parse(Float64, field)
        end
    end

    return energies
end

"""
    parse_quantum_espresso_bands(dir::String, prefix::String, bands_file::String = "\$prefix.band.dat")

Parse a Quantum ESPRESSO `bands.x` output file and its companion XML file to return
a `BandStructureData` object. The function automatically identifies high-symmetry 
points by analyzing the crystal structure provided in the XML.

Arguments:
- `dir`: The directory containing the output files.
- `prefix`: The Quantum ESPRESSO prefix (used to find `<prefix>.xml`).
- `bands_file`: The name of the band data file (default: `<prefix>.band.dat`).
- `branch_gap_factor`: Split consecutive k-path branches when the adjacent k-point
  spacing exceeds this multiple of the median adjacent spacing.
- `branch_gap_threshold`: Optional absolute spacing threshold for branch splitting.
"""
function parse_quantum_espresso_bands(
    dir::String,
    prefix::String,
    bands_file::String="$(prefix).band.dat",
    ;
    branch_gap_factor::Real=5.0,
    branch_gap_threshold::Union{Nothing,Real}=nothing,
)
    xml_path = joinpath(dir, "$prefix.xml")
    if !isfile(xml_path)
        error("Quantum ESPRESSO XML file not found at: $xml_path. High-symmetry path information is required.")
    end

    return with_stage_log(
        "Parse Quantum ESPRESSO bands with XML metadata";
        context=(dir=dir, prefix=prefix, bands_file=bands_file),
    ) do
        # 1. Parse XML and Structure
        xml_dict = parse_quantum_espresso_xml(xml_path)
        system = build_atoms_from_xml(xml_dict)
        alat_angstrom = _qe_alat_angstrom(xml_dict)

        # 2. Extract Fermi Energy from XML metadata
        # Resolve root regardless of namespace
        root = _qe_root_node(xml_dict)

        fermi_level = 0.0
        if haskey(root, "output") && haskey(root["output"], "band_structure")
            bs_node = root["output"]["band_structure"]
            if haskey(bs_node, "fermi_energy")
                fermi_val = bs_node["fermi_energy"]
                fermi_str = fermi_val isa AbstractDict ? get(fermi_val, "", get(fermi_val, "_content", "0.0")) : fermi_val
                fermi_level = parse(Float64, strip(fermi_str)) * ustrip(Ha2eV)
            end
        end

        # 3. Parse Band Data
        dat_path = joinpath(dir, bands_file)
        result = open(dat_path, "r") do io
            eof(io) && error("Quantum ESPRESSO bands file $dat_path is empty.")

            num_bands, num_kpoints = _parse_qe_bands_header(readline(io), dat_path)
            kpoints = Vector{SVector{3,Float64}}(undef, num_kpoints)
            bands = Matrix{Float64}(undef, num_kpoints, num_bands)

            for ik in 1:num_kpoints
                kpoints[ik] = _parse_qe_kpoint(_read_next_nonempty_line(io, dat_path), dat_path)
                bands[ik, :] = _parse_qe_band_block(io, dat_path, num_bands)
            end
            return (kpoints=kpoints, bands=bands, num_bands=num_bands)
        end

        # 4. Construct KPath. Quantum ESPRESSO bands.x writes k-points in
        # Cartesian coordinates in units of 2π/alat (`tpiba`), regardless of
        # whether the input path was specified with `crystal_b` or `tpiba_b`.
        recip_basis_cart = reciprocal_vectors(primitive_vectors(system))
        basis = [SVector{3,Float64}(recip_basis_cart[:, i]) for i in 1:3]
        tpiba = 2π / alat_angstrom

        kpoints_cart = [SVector{3,Float64}(tpiba .* k) for k in result.kpoints]
        labels = _qe_labels_from_high_symmetry(result.kpoints, system, tpiba)
        branched = _branch_kpath_at_large_point_gaps(
            kpoints_cart,
            labels;
            gap_factor=branch_gap_factor,
            gap_threshold=branch_gap_threshold,
        )

        kpath = KPath{3}(
            branched.branches,
            branched.labels,
            basis,
            Ref(Brillouin.CARTESIAN)
        )

        return BandStructureData(
            kpath=kpath,
            bands=result.bands,
            num_bands=result.num_bands,
            fermi_level=fermi_level
        )
    end
end

# Custom QE cell parser removed in favor of AtomsIO.load_system.
