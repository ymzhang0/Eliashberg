# src/IO/IO.jl

"""
    save(path::AbstractString, data; kwargs...)

Save an Eliashberg result object to a file. Supports `.h5`, `.hdf5`, and `.jld2`.
"""
function save(path::AbstractString, data; kwargs...)
    ext = lowercase(splitext(path)[2])
    if ext in (".h5", ".hdf5")
        return _save_hdf5(path, data; kwargs...)
    elseif ext == ".jld2"
        return _save_jld2(path, data; kwargs...)
    end
    throw(ArgumentError("Unsupported file extension $(repr(ext)). Use .h5, .hdf5, or .jld2."))
end

"""
    load(path::AbstractString)

Load an Eliashberg result object from a file. Supports `.h5`, `.hdf5`, and `.jld2`.
"""
function load(path::AbstractString)
    ext = lowercase(splitext(path)[2])
    if ext in (".h5", ".hdf5")
        return _load_hdf5(path)
    elseif ext in (".jld2", ".jld")
        return _load_jld2(path)
    end
    throw(ArgumentError("Unsupported file extension $(repr(ext)). Use .h5, .hdf5, or .jld2."))
end

# --- JLD2 Backend ---

function _save_jld2(path::AbstractString, data; kwargs...)
    JLD2.save(path, Dict("result" => data); kwargs...)
end

function _load_jld2(path::AbstractString)
    return JLD2.jldopen(path, "r") do file
        haskey(file, "result") || throw(ArgumentError("JLD2 file $(repr(path)) does not contain a `result` entry."))
        return file["result"]
    end
end

# --- HDF5 Backend ---

function _save_hdf5(path::AbstractString, data; kwargs...)
    h5open(path, "w") do file
        attributes(file)["result_type"] = string(typeof(data))
        res_group = create_group(file, "result")
        _write_hdf5_data(res_group, data)
    end
end

function _load_hdf5(path::AbstractString)
    return h5open(path, "r") do file
        haskey(file, "result") || throw(ArgumentError("HDF5 file $(repr(path)) does not contain a `/result` group."))
        res_group = file["result"]
        res_type = _read_hdf5_attr(file, "result_type", "")
        return _read_hdf5_data(res_group, res_type)
    end
end

# --- HDF5 Writers ---

_write_hdf5_data(group, data::PhaseDiagramData) = _write_phase_diagram(group, data)
_write_hdf5_data(group, data::RenormalizedBandData) = _write_renormalized_bands(group, data)
_write_hdf5_data(group, data::SpectralMapData) = _write_spectral_map(group, data)
_write_hdf5_data(group, data::ZeemanPairingData) = _write_zeeman_pairing(group, data)
_write_hdf5_data(group, data::BandStructureData) = _write_band_structure(group, data)
_write_hdf5_data(group, data::DispersionSurfaceData) = _write_dispersion_surface(group, data)
_write_hdf5_data(group, data::FermiSurfaceData) = _write_fermi_surface(group, data)
_write_hdf5_data(group, data::LandscapeLineData) = _write_landscape_line(group, data)
_write_hdf5_data(group, data::LandscapeSurfaceData) = _write_landscape_surface(group, data)
_write_hdf5_data(group, data::CoexistenceLandscapeData) = _write_coexistence_landscape(group, data)
_write_hdf5_data(group, data::Wannier90BandComparison) = _write_wannier90_comparison(group, data)

function _write_kpath(group, name::AbstractString, kpath::KPath{D}) where {D}
    path_group = create_group(group, name)
    attributes(path_group)["dimension"] = D
    
    # Flatten branches into a single matrix for storage
    all_points = vcat(kpath.branches...)
    points_matrix = zeros(length(all_points), D)
    for (i, p) in enumerate(all_points)
        points_matrix[i, :] .= p
    end
    path_group["points"] = points_matrix
    
    # Branch metadata
    branch_lengths = length.(kpath.branches)
    branch_stop = cumsum(branch_lengths)
    branch_start = [1; branch_stop[1:end-1] .+ 1]
    path_group["branch_start"] = branch_start
    path_group["branch_stop"] = branch_stop
    
    # Node names and indices
    node_indices = Int[]
    node_labels = String[]
    current_offset = 0
    for (b_idx, labels) in enumerate(kpath.nodes)
        for (idx, sym) in labels
            push!(node_indices, idx + current_offset)
            push!(node_labels, string(sym))
        end
        current_offset += branch_lengths[b_idx]
    end
    path_group["node_indices"] = node_indices
    path_group["node_labels"] = node_labels
end

function _write_phase_diagram(group, data::PhaseDiagramData)
    group["phis"] = data.phis
    group["Ts"] = data.Ts
    group["free_energy"] = data.free_energy
    group["condensation_energy"] = data.condensation_energy
    group["order_parameters"] = data.order_parameters
end

function _write_renormalized_bands(group, data::RenormalizedBandData)
    _write_kpath(group, "kpath", data.kpath)
    group["bare_bands"] = data.bare_bands
    group["renormalized_bands"] = data.renormalized_bands
    group["gaps"] = collect(data.gaps)
    group["Ts"] = data.Ts
end

function _write_spectral_map(group, data::SpectralMapData)
    _write_kpath(group, "qpath", data.qpath)
    group["omegas"] = data.omegas
    group["spectral_matrix"] = data.spectral_matrix
    attributes(group)["gap"] = data.gap
    attributes(group)["T"] = data.T
    if !isnothing(data.pair_breaking_edge)
        attributes(group)["pair_breaking_edge"] = data.pair_breaking_edge
    end
end

function _write_zeeman_pairing(group, data::ZeemanPairingData)
    group["qs"] = data.qs
    group["condensation_energy"] = data.condensation_energy
    group["optimal_gaps"] = data.optimal_gaps
    attributes(group)["optimal_q"] = data.optimal_q
    attributes(group)["minimum_index"] = data.minimum_index
end

function _write_band_structure(group, data::BandStructureData)
    _write_kpath(group, "kpath", data.kpath)
    group["bands"] = data.bands
    attributes(group)["num_bands"] = data.num_bands
end

function _write_dispersion_surface(group, data::DispersionSurfaceData)
    group["kxs"] = data.kxs
    group["kys"] = data.kys
    group["energy_matrix"] = data.energy_matrix
end

function _write_fermi_surface(group, data::FermiSurfaceData)
    group["kxs"] = data.kxs
    group["kys"] = data.kys
    group["kzs"] = data.kzs
    group["energy_volume"] = data.energy_volume
end

function _write_landscape_line(group, data::LandscapeLineData)
    group["qs"] = data.qs
    group["values"] = data.values
end

function _write_landscape_surface(group, data::LandscapeSurfaceData)
    group["qxs"] = data.qxs
    group["qys"] = data.qys
    group["landscape_matrix"] = data.landscape_matrix
end

function _write_coexistence_landscape(group, data::CoexistenceLandscapeData)
    group["phis_1"] = data.phis_1
    group["phis_2"] = data.phis_2
    group["free_energy"] = data.free_energy
    attributes(group)["field_1_type"] = data.field_1_type
    attributes(group)["field_2_type"] = data.field_2_type
end

function _write_wannier90_comparison(group, data::Wannier90BandComparison)
    _write_hdf5_data(create_group(group, "reference"), data.reference)
    _write_hdf5_data(create_group(group, "model"), data.model)
    _write_hdf5_data(create_group(group, "shifted_model"), data.shifted_model)
    
    # K-points as matrix
    kpts = zeros(length(data.kpoints_fractional), 3)
    for (i, p) in enumerate(data.kpoints_fractional)
        kpts[i, :] .= p
    end
    group["kpoints_fractional"] = kpts
    
    group["difference"] = data.difference
    attributes(group)["energy_shift"] = data.energy_shift
    attributes(group)["rms_error"] = data.rms_error
    attributes(group)["max_error"] = data.max_error
end

# --- HDF5 Readers (Migrated & Cleaned) ---

function _read_hdf5_data(group, res_type::AbstractString)
    if occursin("PhaseDiagramData", res_type)
        return _read_phase_diagram(group)
    elseif occursin("RenormalizedBandData", res_type)
        return _read_renormalized_bands(group)
    elseif occursin("SpectralMapData", res_type)
        return _read_spectral_map(group)
    elseif occursin("ZeemanPairingData", res_type)
        return _read_zeeman_pairing(group)
    elseif occursin("BandStructureData", res_type)
        return _read_band_structure(group)
    elseif occursin("DispersionSurfaceData", res_type)
        return _read_dispersion_surface(group)
    elseif occursin("FermiSurfaceData", res_type)
        return _read_fermi_surface(group)
    elseif occursin("LandscapeLineData", res_type)
        return _read_landscape_line(group)
    elseif occursin("LandscapeSurfaceData", res_type)
        return _read_landscape_surface(group)
    elseif occursin("CoexistenceLandscapeData", res_type)
        return _read_coexistence_landscape(group)
    elseif occursin("Wannier90BandComparison", res_type)
        return _read_wannier90_comparison(group)
    end
    
    throw(ArgumentError("Unrecognized result type in HDF5 file: $res_type"))
end

function _read_hdf5_attr(obj, name::AbstractString, default=nothing)
    attributes = attrs(obj)
    return haskey(attributes, name) ? read(attributes[name]) : default
end

function _read_kpath(group, name::AbstractString)
    path_group = group[name]
    points_matrix = read(path_group["points"])
    D = Int(_read_hdf5_attr(path_group, "dimension", size(points_matrix, 2)))
    
    all_points = [SVector{D,Float64}(points_matrix[idx, :]) for idx in axes(points_matrix, 1)]
    
    branch_start = read(path_group["branch_start"])
    branch_stop = read(path_group["branch_stop"])
    
    node_indices = read(path_group["node_indices"])
    node_labels = String.(read(path_group["node_labels"]))
    
    branches = Vector{Vector{SVector{D,Float64}}}()
    labels = Vector{Dict{Int,Symbol}}()
    
    for (start_idx, stop_idx) in zip(branch_start, branch_stop)
        push!(branches, all_points[start_idx:stop_idx])
        
        local_labels = Dict{Int,Symbol}()
        for (ni, nl) in zip(node_indices, node_labels)
            if start_idx <= ni <= stop_idx
                local_labels[ni - start_idx + 1] = Symbol(nl)
            end
        end
        push!(labels, local_labels)
    end
    
    cartesian_basis = [SVector{D,Float64}(ntuple(i -> i == axis ? 1.0 : 0.0, D)) for axis in 1:D]
    return KPath{D}(branches, labels, cartesian_basis, Ref(Brillouin.CARTESIAN))
end

_read_phase_diagram(group) = PhaseDiagramData(
    read(group["phis"]),
    read(group["Ts"]),
    read(group["free_energy"]),
    read(group["condensation_energy"]),
    read(group["order_parameters"])
)

_read_renormalized_bands(group) = RenormalizedBandData(
    _read_kpath(group, "kpath"),
    read(group["bare_bands"]),
    read(group["renormalized_bands"]),
    read(group["gaps"]),
    read(group["Ts"])
)

function _read_spectral_map(group)
    pair_breaking_edge = _read_hdf5_attr(group, "pair_breaking_edge", nothing)
    return SpectralMapData(
        _read_kpath(group, "qpath"),
        read(group["omegas"]),
        read(group["spectral_matrix"]),
        Float64(_read_hdf5_attr(group, "gap", 0.0)),
        pair_breaking_edge === nothing ? nothing : Float64(pair_breaking_edge),
        Float64(_read_hdf5_attr(group, "T", 0.0))
    )
end

_read_zeeman_pairing(group) = ZeemanPairingData(
    read(group["qs"]),
    read(group["condensation_energy"]),
    read(group["optimal_gaps"]),
    Float64(_read_hdf5_attr(group, "optimal_q", 0.0)),
    Int(_read_hdf5_attr(group, "minimum_index", 1))
)

_read_band_structure(group) = BandStructureData(
    _read_kpath(group, "kpath"),
    read(group["bands"]),
    Int(_read_hdf5_attr(group, "num_bands", 1))
)

_read_dispersion_surface(group) = DispersionSurfaceData(
    read(group["kxs"]),
    read(group["kys"]),
    read(group["energy_matrix"])
)

_read_fermi_surface(group) = FermiSurfaceData(
    read(group["kxs"]),
    read(group["kys"]),
    read(group["kzs"]),
    read(group["energy_volume"])
)

_read_landscape_line(group) = LandscapeLineData(
    read(group["qs"]),
    read(group["values"])
)

_read_landscape_surface(group) = LandscapeSurfaceData(
    read(group["qxs"]),
    read(group["qys"]),
    read(group["landscape_matrix"])
)

_read_coexistence_landscape(group) = CoexistenceLandscapeData(
    read(group["phis_1"]),
    read(group["phis_2"]),
    read(group["free_energy"]),
    String(_read_hdf5_attr(group, "field_1_type", "")),
    String(_read_hdf5_attr(group, "field_2_type", ""))
)

function _read_wannier90_comparison(group)
    ref = _read_hdf5_data(group["reference"], "BandStructureData")
    model = _read_hdf5_data(group["model"], "BandStructureData")
    shifted = _read_hdf5_data(group["shifted_model"], "BandStructureData")
    
    kpts_matrix = read(group["kpoints_fractional"])
    kpts = [SVector{3,Float64}(kpts_matrix[i, :]) for i in axes(kpts_matrix, 1)]
    
    return Wannier90BandComparison(
        ref, model, shifted, kpts,
        Float64(_read_hdf5_attr(group, "energy_shift", 0.0)),
        Float64(_read_hdf5_attr(group, "rms_error", 0.0)),
        Float64(_read_hdf5_attr(group, "max_error", 0.0)),
        read(group["difference"])
    )
end
