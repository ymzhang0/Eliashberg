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
    parse_quantum_espresso_bands(filename::String)

Parse a Quantum ESPRESSO `bands.x` output file such as `*.bands.dat` and return
a `NamedTuple` with the k-point coordinates and band energies. The returned
`bands` matrix has shape `(num_kpoints, num_bands)`.
"""
function parse_quantum_espresso_bands(filename::String)
    return with_stage_log(
        "Parse Quantum ESPRESSO bands";
        context=(filename=filename,),
        summarize_result=result -> (num_kpoints=result.num_kpoints, num_bands=result.num_bands),
    ) do
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
end

# Custom QE cell parser removed in favor of AtomsIO.load_system.
