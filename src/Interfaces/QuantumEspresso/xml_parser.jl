# src/Interfaces/QuantumEspresso/xml_parser.jl

using StaticArrays
using AtomsBase
using XMLDict

"""
    parse_quantum_espresso_xml(xml_path::String)

Read a Quantum ESPRESSO XML file and return it as a dictionary via `XMLDict.jl`.
"""
function parse_quantum_espresso_xml(xml_path::String)
    isfile(xml_path) || error("Quantum ESPRESSO XML file not found at: $xml_path")
    xml_data = read(xml_path, String)
    return xml_dict(xml_data)
end

"""
    build_atoms_from_xml(xml_dict::AbstractDict)

Construct an `AtomsBase` system from a parsed Quantum ESPRESSO XML dictionary.
"""
function build_atoms_from_xml(xml_dict::AbstractDict)
    # 1. Identify Root and Structural Node
    root_key = first(filter(k -> endswith(string(k), "espresso"), keys(xml_dict)))
    root = xml_dict[root_key]
    
    struct_node = if haskey(root, "output") && haskey(root["output"], "atomic_structure")
        root["output"]["atomic_structure"]
    elseif haskey(root, "input") && haskey(root["input"], "atomic_structure")
        root["input"]["atomic_structure"]
    else
        error("Could not find <atomic_structure> in Quantum ESPRESSO XML dictionary.")
    end
    
    # helper for vector parsing
    function _parse_vec(v)
        v_str = v isa AbstractDict ? get(v, "", get(v, "_content", get(v, "__content__", v))) : v
        if !(v_str isa AbstractString)
            error("Expected string content for vector, got $(typeof(v_str)): $v_str")
        end
        parts = split(strip(v_str))
        return SVector{3, Float64}(parse.(Float64, parts))
    end
    
    # 2. Parse Lattice Vectors
    cell_node = get(struct_node, "cell", get(struct_node, "lattice_vectors", nothing))
    isnothing(cell_node) && error("Could not find cell or lattice_vectors in XML structure.")
    
    a1 = _parse_vec(cell_node["a1"])
    a2 = _parse_vec(cell_node["a2"])
    a3 = _parse_vec(cell_node["a3"])
    
    bohr_to_ang = ustrip(A2Bohr)
    lattice_vectors = [a1, a2, a3] .* bohr_to_ang .* u"Å"
    
    # 3. Parse Atomic Positions
    pos_node = struct_node["atomic_positions"]
    atom_list = pos_node["atom"] isa AbstractVector ? pos_node["atom"] : [pos_node["atom"]]
    atoms = map(atom_list) do a
        name = a[:name]
        coords = _parse_vec(a) .* bohr_to_ang .* u"Å"
        return Atom(Symbol(name), coords)
    end
    
    return periodic_system(atoms, lattice_vectors)
end
