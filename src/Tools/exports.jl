using HDF5
using TOML
using Eliashberg.Config: Configurations

task_type_name(::Eliashberg.Config.SolveGroundStateOption) = "solve_ground_state"
task_type_name(::Eliashberg.Config.ScanInstabilityLandscapeOption) = "scan_instability_landscape"
task_type_name(::Eliashberg.Config.ScanSpectralFunctionOption) = "scan_spectral_function"
task_type_name(::Eliashberg.Config.ComputePhaseTransitionDataOption) = "compute_phase_transition_data"
task_type_name(::Eliashberg.Config.ComputeRenormalizedBandDataOption) = "compute_renormalized_band_data"
task_type_name(::Eliashberg.Config.ComputeZeemanPairingDataOption) = "compute_zeeman_pairing_data"
task_type_name(::Eliashberg.Config.ComputeCollectiveModeSpectralDataOption) = "compute_collective_mode_spectral_data"
task_type_name(task) = string(typeof(task))

function _points_matrix(points)
    isempty(points) && return Matrix{Float64}(undef, 0, 0)
    return reduce(vcat, permutedims.(collect.(points)))
end

function _write_string_vector(group, name::AbstractString, values::AbstractVector{<:AbstractString})
    group[name] = collect(String.(values))
    return nothing
end

function _write_kpath(group, name::AbstractString, kpath)
    path_group = create_group(group, name)
    points = Eliashberg.path_points(kpath)
    node_indices, node_labels = Eliashberg.path_node_metadata(kpath)
    branch_ranges = Eliashberg.path_branch_ranges(kpath)

    path_group["points"] = _points_matrix(points)
    path_group["node_indices"] = Int.(node_indices)
    _write_string_vector(path_group, "node_labels", node_labels)
    path_group["branch_start"] = [first(r) for r in branch_ranges]
    path_group["branch_stop"] = [last(r) for r in branch_ranges]
    attrs(path_group)["dimension"] = isempty(points) ? 0 : length(first(points))
    attrs(path_group)["n_points"] = length(points)
    return path_group
end

function _write_config_group(group, config, toml_path::AbstractString)
    config_group = create_group(group, "config")
    config_group["input_toml"] = read(toml_path, String)
    config_dict = Configurations.to_dict(config)
    config_group["config_dict_repr"] = sprint(show, config_dict)
    attrs(config_group)["type"] = string(typeof(config))
    return config_group, config_dict
end

function _write_phase_diagram(group, result::PhaseDiagramData)
    group["phis"] = result.phis
    group["temperatures"] = result.Ts
    group["free_energy"] = result.free_energy
    group["condensation_energy"] = result.condensation_energy
    group["order_parameters"] = result.order_parameters
    return group
end

function _write_renormalized_bands(group, result::RenormalizedBandData)
    _write_kpath(group, "kpath", result.kpath)
    group["bare_bands"] = result.bare_bands
    group["renormalized_bands"] = result.renormalized_bands
    group["temperatures"] = result.temperatures
    group["gaps"] = result.gaps
    return group
end

function _write_spectral_map(group, result::SpectralMapData)
    _write_kpath(group, "qpath", result.qpath)
    group["omegas"] = result.omegas
    group["spectral_matrix"] = result.spectral_matrix
    attrs(group)["gap"] = result.gap
    attrs(group)["temperature"] = result.temperature
    if !isnothing(result.pair_breaking_edge)
        attrs(group)["pair_breaking_edge"] = result.pair_breaking_edge
    end
    return group
end

function _write_zeeman_pairing(group, result::ZeemanPairingData)
    group["q_vals"] = result.q_vals
    group["condensation_energy"] = result.condensation_energy
    group["optimal_gaps"] = result.optimal_gaps
    attrs(group)["optimal_q"] = result.optimal_q
    attrs(group)["minimum_index"] = result.minimum_index
    return group
end

function _write_hdf5_result(group, result::PhaseDiagramData)
    return _write_phase_diagram(group, result)
end

function _write_hdf5_result(group, result::RenormalizedBandData)
    return _write_renormalized_bands(group, result)
end

function _write_hdf5_result(group, result::SpectralMapData)
    return _write_spectral_map(group, result)
end

function _write_hdf5_result(group, result::ZeemanPairingData)
    return _write_zeeman_pairing(group, result)
end

function write_result_hdf5(save_file::AbstractString, result, config, toml_path::AbstractString; time_seconds::Real)
    h5open(save_file, "w") do file
        _, config_dict = _write_config_group(file, config, toml_path)
        task_type = get(get(config_dict, "task", Dict{String,Any}()), "type", task_type_name(config.task))

        attrs(file)["format"] = "Eliashberg-HDF5"
        attrs(file)["format_version"] = "1.0"
        attrs(file)["result_type"] = string(typeof(result))
        attrs(file)["task_type"] = task_type
        attrs(file)["time_seconds"] = Float64(time_seconds)
        result_group = create_group(file, "result")
        _write_hdf5_result(result_group, result)
    end

    return save_file
end
