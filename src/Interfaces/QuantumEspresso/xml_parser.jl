# src/Interfaces/QuantumEspresso/xml_parser.jl

using StaticArrays
using AtomsBase
using Unitful

"""
    parse_quantum_espresso_xml(filename::String)

Parse a Quantum ESPRESSO data-file-schema.xml (or prefix.xml) and return an 
`AtomsBase.AbstractSystem`. This parser extracts the lattice vectors and 
atomic positions in Bohr and converts them to Angstroms.
"""
function parse_quantum_espresso_xml(filename::String)
    content = read(filename, String)

    # 1. Parse Cell Vectors
    a1 = _parse_xml_vector(content, "a1")
    a2 = _parse_xml_vector(content, "a2")
    a3 = _parse_xml_vector(content, "a3")

    if isnothing(a1) || isnothing(a2) || isnothing(a3)
        error("Could not find lattice vectors (a1, a2, a3) in QE XML file: $filename")
    end

    # 2. Parse Atomic Positions
    atoms_data = _parse_xml_atoms(content)
    if isempty(atoms_data)
        error("No atomic positions found in QE XML file: $filename")
    end

    # 3. Construct periodic cell (Bohr to Angstrom)
    # 1 Bohr = 0.529177210903 Angstrom
    bohr_to_ang = A2Bohr # Uses the constant from our package if available, or just the number
    
    # Actually, we have A2Bohr which is Bohr/Angstrom. 
    # To get Angstrom from Bohr, we divide by A2Bohr or multiply by inverse.
    # Let's use the package constant Ry2eV and other things if we need, 
    # but for geometry, A2Bohr is Bohr / Å.
    
    lattice = [a1 a2 a3] ./ A2Bohr

    # 4. Build AtomsBase system
    atoms = [Atom(species, pos ./ A2Bohr * u"Å") for (species, pos) in atoms_data]
    
    # We use FastSystem or whatever is convenient
    return periodic_system(atoms, [lattice[:, i] .* u"Å" for i in 1:3])
end

function _parse_xml_vector(content::AbstractString, tag::AbstractString)
    m = match(Regex("<$tag>(.*?)</$tag>"), content)
    isnothing(m) && return nothing
    # Clean up whitespace and parse
    vals = parse.(Float64, split(strip(m.captures[1])))
    length(vals) == 3 || error("Expected 3 values for tag <$tag>, found $(length(vals))")
    return SVector{3, Float64}(vals)
end

function _parse_xml_atoms(content::AbstractString)
    atoms = Tuple{Symbol, SVector{3, Float64}}[]
    # Matches <atom name="Nb" index="1">0.0 0.0 0.0</atom>
    for m in eachmatch(r"<atom\s+name=\"(.*?)\"\s+index=\"\d+\">(.*?)</atom>", content)
        species = Symbol(m.captures[1])
        pos_vals = parse.(Float64, split(strip(m.captures[2])))
        push!(atoms, (species, SVector{3, Float64}(pos_vals)))
    end
    return atoms
end
