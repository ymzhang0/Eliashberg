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
    parse_quantum_espresso_bands(dir::String, prefix::String, bands_file::String = "$prefix.bands.dat")

Parse a Quantum ESPRESSO `bands.x` output file and its companion XML file to return
a `BandStructureData` object. The function automatically identifies high-symmetry 
points by analyzing the crystal structure provided in the XML.

Arguments:
- `dir`: The directory containing the output files.
- `prefix`: The Quantum ESPRESSO prefix (used to find `<prefix>.xml`).
- `bands_file`: The name of the band data file (default: `<prefix>.bands.dat`).
"""
function parse_quantum_espresso_bands(dir::String, prefix::String, bands_file::String = "$prefix.bands.dat")
    xml_path = joinpath(dir, "$prefix.xml")
    if !isfile(xml_path)
        error("Quantum ESPRESSO XML file not found at: $xml_path. High-symmetry path information is required.")
    end

    return with_stage_log(
        "Parse Quantum ESPRESSO bands with XML metadata";
        context=(dir=dir, prefix=prefix, bands_file=bands_file),
    ) do
        # 1. Parse Structure and Symmetry
        system = parse_quantum_espresso_xml(xml_path)
        path_info = symmetry_path(system)
        
        # 2. Parse Band Data
        dat_path = joinpath(dir, bands_file)
        result = open(dat_path, "r") do io
            eof(io) && error("Quantum ESPRESSO bands file $dat_path is empty.")

            num_bands, num_kpoints = _parse_qe_bands_header(readline(io), dat_path)
            kpoints = Vector{SVector{3, Float64}}(undef, num_kpoints)
            bands = Matrix{Float64}(undef, num_kpoints, num_bands)

            for ik in 1:num_kpoints
                kpoints[ik] = _parse_qe_kpoint(_read_next_nonempty_line(io, dat_path), dat_path)
                bands[ik, :] = _parse_qe_band_block(io, dat_path, num_bands)
            end
            return (kpoints=kpoints, bands=bands, num_bands=num_bands)
        end

        # 3. Match Symmetry Points
        labels = Dict{Int, Symbol}()
        if !isnothing(path_info)
            for (label, coord) in path_info.points
                for (ik, k) in enumerate(result.kpoints)
                    if norm(k - coord) < 1e-4
                        labels[ik] = Symbol(label)
                    end
                end
            end
        end

        # 4. Construct KPath
        recip_basis_cart = reciprocal_vectors(primitive_vectors(system))
        basis = [SVector{3, Float64}(recip_basis_cart[:, i]) for i in 1:3]
        
        # Convert parsed k-points (fractional) to Cartesian
        k_matrix = reduce(hcat, basis)
        kpoints_cart = [SVector{3, Float64}(k_matrix * k) for k in result.kpoints]

        kpath = KPath{3}(
            [kpoints_cart],
            [labels],
            basis,
            Ref(Brillouin.CARTESIAN)
        )

        return BandStructureData(kpath, result.bands, result.num_bands)
    end
end

# Custom QE cell parser removed in favor of AtomsIO.load_system.
