# src/Interfaces/quantum_espresso.jl

const QE_BOHR_TO_ANGSTROM = 0.529177210903

_qe_bohr_to_angstrom(value::Real) = Float64(value) * QE_BOHR_TO_ANGSTROM

function _parse_qe_number(token::AbstractString, filename::AbstractString, context::AbstractString)
    try
        return parse(Float64, replace(strip(token), r"[dD]" => "e"))
    catch err
        throw(ArgumentError("Could not parse $context in $filename from token `$(strip(token))`: $(sprint(showerror, err))"))
    end
end

function _parse_qe_int(token::AbstractString, filename::AbstractString, context::AbstractString)
    try
        return parse(Int, strip(token))
    catch err
        throw(ArgumentError("Could not parse $context in $filename from token `$(strip(token))`: $(sprint(showerror, err))"))
    end
end

function _qe_assignment_capture(text::AbstractString, pattern::Regex)
    match_result = match(pattern, text)
    return match_result === nothing ? nothing : match_result.captures[1]
end

function _qe_assignment_float(text::AbstractString, filename::AbstractString, context::AbstractString, patterns::AbstractVector{Regex})
    for pattern in patterns
        capture = _qe_assignment_capture(text, pattern)
        capture === nothing || return _parse_qe_number(capture, filename, context)
    end
    return nothing
end

function _qe_assignment_int(text::AbstractString, filename::AbstractString, context::AbstractString, patterns::AbstractVector{Regex})
    for pattern in patterns
        capture = _qe_assignment_capture(text, pattern)
        capture === nothing || return _parse_qe_int(capture, filename, context)
    end
    return nothing
end

function _qe_parse_vector3(line::AbstractString, filename::AbstractString, context::AbstractString)
    fields = split(strip(line))
    length(fields) == 3 || error("Expected 3 floating-point values for $context in $filename.")
    return SVector{3, Float64}(
        _parse_qe_number(fields[1], filename, "$context x"),
        _parse_qe_number(fields[2], filename, "$context y"),
        _parse_qe_number(fields[3], filename, "$context z"),
    )
end

function _qe_alat_angstrom(text::AbstractString, filename::AbstractString)
    celldm1 = _qe_assignment_float(
        text,
        filename,
        "`celldm(1)`",
        [r"(?im)celldm\s*\(\s*1\s*\)\s*=\s*([+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[de][+\-]?\d+)?)"],
    )
    celldm1 === nothing || return _qe_bohr_to_angstrom(celldm1)

    a_value = _qe_assignment_float(
        text,
        filename,
        "`A`",
        [r"(?im)(?:^|[, \t])A\s*=\s*([+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[de][+\-]?\d+)?)"],
    )
    a_value === nothing || return a_value

    alat_value = _qe_assignment_float(
        text,
        filename,
        "`alat`",
        [r"(?im)lattice\s+parameter\s*\(alat\)\s*=\s*([+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[de][+\-]?\d+)?)"],
    )
    alat_value === nothing || return _qe_bohr_to_angstrom(alat_value)

    return nothing
end

function _qe_scale_cell_vectors(vectors::Vector{SVector{3, Float64}}, unit_token::AbstractString, text::AbstractString, filename::AbstractString)
    unit = lowercase(strip(unit_token))

    if isempty(unit) || unit == "angstrom"
        scale = 1.0
    elseif unit == "bohr" || unit == "a.u."
        scale = _qe_bohr_to_angstrom(1.0)
    elseif unit == "alat"
        alat = _qe_alat_angstrom(text, filename)
        alat === nothing && error("Could not determine `alat` while parsing CELL_PARAMETERS in $filename.")
        scale = alat
    else
        error("Unsupported CELL_PARAMETERS unit `$unit_token` in $filename.")
    end

    return [scale * vector for vector in vectors]
end

function _qe_parse_cell_parameters(lines::Vector{String}, text::AbstractString, filename::AbstractString)
    for idx in eachindex(lines)
        stripped = strip(lines[idx])
        startswith(uppercase(stripped), "CELL_PARAMETERS") || continue

        parts = split(stripped)
        unit_token = length(parts) >= 2 ? parts[2] : ""
        idx + 3 <= length(lines) || error("Incomplete CELL_PARAMETERS block in $filename.")

        vectors = [
            _qe_parse_vector3(lines[idx + offset], filename, "CELL_PARAMETERS vector $offset")
            for offset in 1:3
        ]
        return PeriodicCell(_qe_scale_cell_vectors(vectors, unit_token, text, filename))
    end

    return nothing
end

function _qe_parse_output_crystal_axes(lines::Vector{String}, text::AbstractString, filename::AbstractString)
    alat = _qe_alat_angstrom(text, filename)

    for idx in eachindex(lines)
        occursin(r"(?i)crystal axes", lines[idx]) || continue
        alat === nothing && error("Found `crystal axes` block in $filename but could not determine `alat`.")
        idx + 3 <= length(lines) || error("Incomplete `crystal axes` block in $filename.")

        vectors = Vector{SVector{3, Float64}}(undef, 3)
        for offset in 1:3
            line = lines[idx + offset]
            match_result = match(r"=\s*\(\s*([^\)]+)\)", line)
            match_result === nothing && error("Could not parse crystal axis line in $filename: $line")
            vectors[offset] = alat * _qe_parse_vector3(match_result.captures[1], filename, "crystal axis $offset")
        end

        return PeriodicCell(vectors)
    end

    return nothing
end

function _qe_parse_ibrav_cell(text::AbstractString, filename::AbstractString)
    ibrav = _qe_assignment_int(
        text,
        filename,
        "`ibrav`",
        [
            r"(?im)(?:^|[, \t])ibrav\s*=\s*([+\-]?\d+)",
            r"(?im)bravais-lattice index\s*=\s*([+\-]?\d+)",
        ],
    )
    ibrav === nothing && return nothing
    ibrav == 0 && return nothing

    from_A = _qe_assignment_capture(text, r"(?im)(?:^|[, \t])A\s*=\s*([+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[de][+\-]?\d+)?)") !== nothing
    a_value = _qe_assignment_float(
        text,
        filename,
        "`A`/`celldm(1)`",
        [
            r"(?im)(?:^|[, \t])A\s*=\s*([+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[de][+\-]?\d+)?)",
            r"(?im)celldm\s*\(\s*1\s*\)\s*=\s*([+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[de][+\-]?\d+)?)",
        ],
    )
    a_value === nothing && error("Could not determine lattice parameter `a` for `ibrav=$ibrav` in $filename.")
    from_A || (a_value = _qe_bohr_to_angstrom(a_value))

    from_B = _qe_assignment_capture(text, r"(?im)(?:^|[, \t])B\s*=\s*([+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[de][+\-]?\d+)?)") !== nothing
    from_C = _qe_assignment_capture(text, r"(?im)(?:^|[, \t])C\s*=\s*([+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[de][+\-]?\d+)?)") !== nothing

    celldm2 = _qe_assignment_float(
        text,
        filename,
        "`celldm(2)`/`B`",
        [
            r"(?im)celldm\s*\(\s*2\s*\)\s*=\s*([+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[de][+\-]?\d+)?)",
            r"(?im)(?:^|[, \t])B\s*=\s*([+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[de][+\-]?\d+)?)",
        ],
    )
    celldm3 = _qe_assignment_float(
        text,
        filename,
        "`celldm(3)`/`C`",
        [
            r"(?im)celldm\s*\(\s*3\s*\)\s*=\s*([+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[de][+\-]?\d+)?)",
            r"(?im)(?:^|[, \t])C\s*=\s*([+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[de][+\-]?\d+)?)",
        ],
    )
    cosbc = _qe_assignment_float(
        text,
        filename,
        "`celldm(4)`/`cosBC`",
        [
            r"(?im)celldm\s*\(\s*4\s*\)\s*=\s*([+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[de][+\-]?\d+)?)",
            r"(?im)(?:^|[, \t])cosBC\s*=\s*([+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[de][+\-]?\d+)?)",
        ],
    )
    cosac = _qe_assignment_float(
        text,
        filename,
        "`celldm(5)`/`cosAC`",
        [
            r"(?im)celldm\s*\(\s*5\s*\)\s*=\s*([+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[de][+\-]?\d+)?)",
            r"(?im)(?:^|[, \t])cosAC\s*=\s*([+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[de][+\-]?\d+)?)",
        ],
    )
    cosab = _qe_assignment_float(
        text,
        filename,
        "`celldm(6)`/`cosAB`",
        [
            r"(?im)celldm\s*\(\s*6\s*\)\s*=\s*([+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[de][+\-]?\d+)?)",
            r"(?im)(?:^|[, \t])cosAB\s*=\s*([+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[de][+\-]?\d+)?)",
        ],
    )

    celldm2 === nothing || (from_B && (celldm2 /= a_value))
    celldm3 === nothing || (from_C && (celldm3 /= a_value))

    return qe_lattice(
        ibrav,
        a_value;
        celldm2=celldm2,
        celldm3=celldm3,
        celldm4=cosbc,
        celldm5=cosac,
        celldm6=cosab,
    )
end

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

    return SVector{3, Float64}(
        parse(Float64, fields[1]),
        parse(Float64, fields[2]),
        parse(Float64, fields[3]),
    )
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
    parse_quantum_espresso_bands(filename::String)

Parse a Quantum ESPRESSO `bands.x` output file such as `*.bands.dat` and return
a `NamedTuple` with the k-point coordinates and band energies. The returned
`bands` matrix has shape `(num_kpoints, num_bands)`.
"""
function parse_quantum_espresso_bands(filename::String)
    open(filename, "r") do io
        eof(io) && error("Quantum ESPRESSO bands file $filename is empty.")

        num_bands, num_kpoints = _parse_qe_bands_header(readline(io), filename)
        kpoints = Vector{SVector{3, Float64}}(undef, num_kpoints)
        bands = Matrix{Float64}(undef, num_kpoints, num_bands)

        for ik in 1:num_kpoints
            kpoints[ik] = _parse_qe_kpoint(_read_next_nonempty_line(io, filename), filename)
            bands[ik, :] = _parse_qe_band_block(io, filename, num_bands)
        end

        while !eof(io)
            isempty(strip(readline(io))) || error("Found unexpected trailing content in $filename after reading $num_kpoints k-points.")
        end

        return (
            kpoints = kpoints,
            bands = bands,
            num_kpoints = num_kpoints,
            num_bands = num_bands,
        )
    end
end

"""
    parse_quantum_espresso_cell(filename::String; periodicity=nothing)

Parse a primitive cell from a Quantum ESPRESSO input or output file. The parser
supports explicit `CELL_PARAMETERS` blocks, output `crystal axes` blocks, and
QE-style `ibrav` definitions. By default the returned cell is an
`AtomsBase.PeriodicCell` with full three-dimensional periodicity; pass
`periodicity=(true, true, false)` for slab systems, for example.
"""
function parse_quantum_espresso_cell(filename::String; periodicity=nothing)
    lines = readlines(filename)
    isempty(lines) && error("Quantum ESPRESSO cell file $filename is empty.")
    text = join(lines, '\n')
    atomsbase_periodicity = isnothing(periodicity) ? (true, true, true) :
        periodicity isa Bool ? ntuple(_ -> periodicity, 3) : Tuple(periodicity)

    cell = _qe_parse_cell_parameters(lines, text, filename)
    cell === nothing || return PeriodicCell(cell; periodicity=atomsbase_periodicity)

    cell = _qe_parse_output_crystal_axes(lines, text, filename)
    cell === nothing || return PeriodicCell(cell; periodicity=atomsbase_periodicity)

    cell = _qe_parse_ibrav_cell(text, filename)
    cell === nothing || error("Could not determine a Quantum ESPRESSO cell from $filename.")
    return PeriodicCell(cell; periodicity=atomsbase_periodicity)
end

function _qe_reciprocal_basis(cell_like)
    reciprocal = reciprocal_vectors(primitive_vectors(cell_like))
    D = size(reciprocal, 1)
    return [SVector{D, Float64}(reciprocal[:, idx]) for idx in 1:D]
end

function _qe_cartesianize_kpoints(kpoints::AbstractVector{<:SVector{D, <:Real}}, basis::AbstractVector{<:SVector{D, Float64}}) where {D}
    basis_matrix = reduce(hcat, basis)
    return [SVector{D, Float64}(basis_matrix * k) for k in kpoints]
end

"""
    kpath_from_quantum_espresso_bands(
        kpoints;
        cell=nothing,
        coordinates=:fractional,
        node_labels=nothing,
    )

Build a single-branch `KPath` from Quantum ESPRESSO band-path samples. When a
`cell` is provided and `coordinates == :fractional`, the k-points are
interpreted as fractional coordinates in the reciprocal basis and converted to
Cartesian reciprocal-space vectors.
"""
function kpath_from_quantum_espresso_bands(
    kpoints::AbstractVector{<:SVector{D, <:Real}};
    cell=nothing,
    coordinates::Symbol=:fractional,
    node_labels::Union{Nothing, AbstractVector{<:AbstractString}}=nothing,
) where {D}
    coordinates in (:fractional, :cartesian) || throw(ArgumentError("`coordinates` must be either `:fractional` or `:cartesian`."))

    basis = cell === nothing ?
        [SVector{D, Float64}(ntuple(i -> i == j ? 1.0 : 0.0, D)) for j in 1:D] :
        _qe_reciprocal_basis(cell)

    path_points = if coordinates == :fractional && cell !== nothing
        _qe_cartesianize_kpoints(kpoints, basis)
    else
        [SVector{D, Float64}(point) for point in kpoints]
    end

    labels = Dict{Int, Symbol}()
    if node_labels !== nothing
        length(node_labels) == length(path_points) || throw(DimensionMismatch("`node_labels` must match the number of k-points."))
        for (idx, label) in pairs(node_labels)
            isempty(strip(label)) || (labels[idx] = Symbol(label))
        end
    end

    return KPath{D}([path_points], [labels], basis, Ref(Brillouin.CARTESIAN))
end

"""
    band_data_from_quantum_espresso_bands(
        bands_filename::String;
        cell=nothing,
        cell_filename=nothing,
        coordinates=:fractional,
        node_labels=nothing,
    )

Parse a Quantum ESPRESSO `bands.x` output file and wrap it as
`BandStructureData` for direct plotting with `plot_band_structure`.
"""
function band_data_from_quantum_espresso_bands(
    bands_filename::String;
    cell=nothing,
    cell_filename::Union{Nothing, AbstractString}=nothing,
    coordinates::Symbol=:fractional,
    node_labels::Union{Nothing, AbstractVector{<:AbstractString}}=nothing,
)
    parsed = parse_quantum_espresso_bands(bands_filename)
    resolved_cell = cell_filename === nothing ? cell : parse_quantum_espresso_cell(String(cell_filename))
    kpath = kpath_from_quantum_espresso_bands(
        parsed.kpoints;
        cell=resolved_cell,
        coordinates=coordinates,
        node_labels=node_labels,
    )

    return BandStructureData(kpath, parsed.bands, parsed.num_bands)
end
