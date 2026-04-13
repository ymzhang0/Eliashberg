# src/Tools/runner.jl

using Pkg
Pkg.activate(".") # 确保激活环境
using Distributed
using Dates
using TOML
using LinearAlgebra
using Logging
using TerminalLoggers
using JLD2        # 用于无损保存 Julia 数据结构

# 加载核心物理代码及其中嵌套的 Config 模块
using Eliashberg
using Eliashberg.Config: EliashbergConfig, Configurations, build_from_config
include("exports.jl")

struct TeeLogger{L<:Tuple} <: AbstractLogger
    loggers::L
end

Logging.min_enabled_level(logger::TeeLogger) = minimum(Logging.min_enabled_level(child) for child in logger.loggers)
Logging.catch_exceptions(logger::TeeLogger) = any(Logging.catch_exceptions(child) for child in logger.loggers)
Logging.shouldlog(logger::TeeLogger, level, _module, group, id) =
    any(Logging.shouldlog(child, level, _module, group, id) for child in logger.loggers)

function Logging.handle_message(logger::TeeLogger, level, message, _module, group, id, file, line; kwargs...)
    for child in logger.loggers
        Logging.min_enabled_level(child) <= level || continue
        Logging.shouldlog(child, level, _module, group, id) || continue
        Logging.handle_message(child, level, message, _module, group, id, file, line; kwargs...)
    end
    return nothing
end

struct StageFilterLogger{L<:AbstractLogger} <: AbstractLogger
    logger::L
    min_stage_level::LogLevel
end

Logging.min_enabled_level(logger::StageFilterLogger) = Logging.min_enabled_level(logger.logger)
Logging.catch_exceptions(logger::StageFilterLogger) = Logging.catch_exceptions(logger.logger)

function Logging.shouldlog(logger::StageFilterLogger, level, _module, group, id)
    stage_group = getfield(Eliashberg, :_LOG_STAGE_GROUP)
    group == stage_group && level < logger.min_stage_level && return false
    return Logging.shouldlog(logger.logger, level, _module, group, id)
end

function Logging.handle_message(logger::StageFilterLogger, level, message, _module, group, id, file, line; kwargs...)
    stage_group = getfield(Eliashberg, :_LOG_STAGE_GROUP)
    group == stage_group && level < logger.min_stage_level && return nothing
    return Logging.handle_message(logger.logger, level, message, _module, group, id, file, line; kwargs...)
end

function _configure_blas_threads!(n::Integer=1)
    try
        BLAS.set_num_threads(n)
    catch err
        @warn "Failed to set BLAS thread count." requested_threads=n error=err
    end
    return nothing
end

function _default_plot_filename(task_type::AbstractString)
    return task_type * ".png"
end

function _resolve_job_directory(system, timestamp::AbstractString)
    root = system.output_dir
    stem = isnothing(system.job_name) ? "job" : system.job_name
    return joinpath(root, stem * "_" * timestamp)
end

function _resolve_output_stem(system)
    filename = system.output_filename
    stem, ext = splitext(filename)
    return isempty(ext) ? filename : stem
end

function _resolve_optional_path(out_dir::AbstractString, path::Union{Nothing,AbstractString})
    isnothing(path) && return nothing
    return isabspath(path) ? String(path) : joinpath(out_dir, path)
end

function _resolve_result_paths(out_dir::AbstractString, system)
    system.write_hdf5 || system.write_jld2 || throw(ArgumentError("At least one result format must be enabled in [system]: set write_hdf5=true or write_jld2=true."))

    stem = _resolve_output_stem(system)
    jld2_file = system.write_jld2 ? joinpath(out_dir, stem * ".jld2") : nothing
    hdf5_file = system.write_hdf5 ? joinpath(out_dir, stem * ".h5") : nothing
    log_file = _resolve_optional_path(out_dir, system.log_filename)
    return (; jld2_file, hdf5_file, log_file)
end

function _parse_log_level(level::AbstractString)
    normalized = lowercase(strip(level))
    normalized in ("debug", "trace") && return Logging.Debug
    normalized == "info" && return Logging.Info
    normalized in ("warn", "warning") && return Logging.Warn
    normalized == "error" && return Logging.Error
    throw(ArgumentError("Unsupported system.log_level=$(repr(level)). Use one of: debug, info, warn, error."))
end

function _open_job_logger(system, log_file::Union{Nothing,AbstractString})
    file_level = _parse_log_level(system.log_level)
    console_level = system.quiet ? Logging.Warn : file_level
    stage_level = file_level <= Logging.Debug ? Logging.Debug : Logging.Warn
    console_logger = StageFilterLogger(TerminalLogger(stderr, console_level), stage_level)

    isnothing(log_file) && return (console_logger, nothing)

    mkpath(dirname(log_file))
    io = open(log_file, "w")
    file_logger = StageFilterLogger(SimpleLogger(io, file_level), stage_level)
    return (TeeLogger((console_logger, file_logger)), io)
end

_format_seconds(seconds::Real) = round(Float64(seconds); digits=3)

function _format_bytes(bytes::Integer)
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    value = Float64(bytes)
    unit_idx = 1
    while value >= 1024 && unit_idx < length(units)
        value /= 1024
        unit_idx += 1
    end
    return string(round(value; digits=value >= 100 ? 1 : 2), " ", units[unit_idx])
end

function _safe_maxrss_bytes()
    if isdefined(Sys, :maxrss)
        try
            return Int(Sys.maxrss())
        catch
        end
    end
    return nothing
end

function _safe_live_heap_bytes()
    try
        GC.gc()
        return Int(Base.gc_live_bytes())
    catch
        return nothing
    end
end

function _resource_summary(config, project::AbstractString)
    requested_workers = Int(config.system.n_workers)
    current_workers = Distributed.nworkers()
    planned_workers = if config.system.bootstrap_workers
        max(requested_workers, current_workers)
    else
        current_workers
    end

    return (
        project=project,
        bootstrap_workers=config.system.bootstrap_workers,
        requested_workers=requested_workers,
        available_workers=current_workers,
        planned_workers=planned_workers,
        threads_per_process=Threads.nthreads(),
        blas_threads=BLAS.get_num_threads(),
        restrict=config.system.restrict,
    )
end

function _task_run_summary(task_type::AbstractString, params)
    if task_type == "scan_spectral_function"
        omega_axis = params.task.omegas
        qpath = params.task.qpath
        return (
            task="scan spectral function",
            q_path=Eliashberg.kpath_summary(qpath),
            omega_range=(minimum(omega_axis), maximum(omega_axis)),
            n_omegas=length(omega_axis),
            temperature=Float64(params.task.T_val),
            eta=Float64(params.task.eta),
        )
    elseif task_type == "compute_phase_transition_data"
        return (
            task="compute phase transition data",
            n_phis=length(params.task.phis),
            n_temperatures=length(params.task.Ts),
            approx=string(typeof(params.task.approx)),
        )
    elseif task_type == "compute_renormalized_band_data"
        return (
            task="compute renormalized band data",
            k_path=Eliashberg.kpath_summary(params.task.kpath),
            n_temperatures=length(params.task.Ts),
            approx=string(typeof(params.task.approx)),
        )
    elseif task_type == "compute_collective_mode_spectral_data"
        return (
            task="compute collective mode spectral data",
            q_path=Eliashberg.kpath_summary(params.task.qpath),
            n_omegas=Int(params.task.n_omegas),
            temperature=Float64(params.task.T_val),
            eta=Float64(params.task.eta),
            approx=string(typeof(params.task.approx)),
        )
    elseif task_type == "compute_zeeman_pairing_data"
        return (
            task="compute zeeman pairing data",
            n_q=length(params.task.q_vals),
            field_strength=Float64(params.task.h_val),
            temperature=Float64(params.task.T_val),
            approx=string(typeof(params.task.approx)),
        )
    end

    return (task=task_type,)
end

function _result_summary(result)
    if result isa AbstractArray
        return (data_type=string(typeof(result)), array_shape=size(result), element_type=string(eltype(result)))
    end
    return (data_type=string(typeof(result)),)
end

function _activate_plot_backend!()
    for backend in (:CairoMakie, :GLMakie, :WGLMakie)
        isnothing(Base.find_package(String(backend))) && continue
        try
            Core.eval(Main, :(import $(backend)))
            return backend
        catch err
            @warn "Failed to activate plotting backend." backend=String(backend) error=err
        end
    end
    return nothing
end

function _plot_result_payload(task_type::AbstractString, result, params)
    if task_type == "scan_spectral_function"
        data = SpectralMapData(
            params.task.qpath,
            params.task.omegas,
            result,
            0.0,
            nothing,
            params.task.T_val,
        )
        return plot_spectral_function(data)
    elseif task_type == "compute_phase_transition_data"
        return plot_phase_transition(result)
    elseif task_type == "compute_renormalized_band_data"
        return plot_renormalized_bands(result)
    elseif task_type == "compute_collective_mode_spectral_data"
        return plot_collective_modes(result)
    elseif task_type == "compute_zeeman_pairing_data"
        return plot_zeeman_pairing_landscape(result)
    end
    throw(ArgumentError("No plotter registered for task type $(task_type)."))
end

function _maybe_save_plot(out_dir::AbstractString, task_type::AbstractString, result, params)
    params.plot.enable || return nothing

    figure = _plot_result_payload(task_type, result, params)
    plot_path = isnothing(params.plot.path) ? joinpath(out_dir, _default_plot_filename(task_type)) :
        (isabspath(params.plot.path) ? params.plot.path : joinpath(out_dir, params.plot.path))
    mkpath(dirname(plot_path))
    backend = _activate_plot_backend!()

    if isnothing(backend)
        @warn "Plotting requested but no Makie backend is available; skipping figure export." file=plot_path
        return nothing
    end

    try
        Eliashberg.load_visualization!()
        visualization = Base.invokelatest(getproperty, Eliashberg, :Visualization)
        makie_module = Base.invokelatest(getproperty, visualization, :Makie)
        save_fn = Base.invokelatest(getproperty, makie_module, :save)
        Base.invokelatest(save_fn, plot_path, figure)
        @info "Plot saved successfully." file = plot_path backend = String(backend)
        return plot_path
    catch err
        @warn "Plotting failed; numerical outputs were still saved." file=plot_path backend=String(backend) error=err
        return nothing
    end
end


function submit_job(toml_path::String)
    config = Configurations.from_toml(EliashbergConfig, toml_path)
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    out_dir = _resolve_job_directory(config.system, timestamp)
    mkpath(out_dir)
    paths = _resolve_result_paths(out_dir, config.system)
    logger, log_io = _open_job_logger(config.system, paths.log_file)

    try
        return with_logger(logger) do
            task_type = task_type_name(config.task)
            project = something(config.system.project, Base.active_project(), ".")
            restrict = config.system.restrict
            _configure_blas_threads!(1)
            resource_summary = _resource_summary(config, project)

            @info "Starting Eliashberg job" config_file = toml_path task_type = task_type output_dir = out_dir log_file = paths.log_file
            @info "Execution resources" project = resource_summary.project bootstrap_workers = resource_summary.bootstrap_workers requested_workers = resource_summary.requested_workers available_workers = resource_summary.available_workers planned_workers = resource_summary.planned_workers threads_per_process = resource_summary.threads_per_process blas_threads = resource_summary.blas_threads restrict = resource_summary.restrict

            @info "Building lattice and noninteracting model from input..."
            params = build_from_config(config)
            @info "Physical system ready" lattice = Eliashberg.cell_summary(params.geometry) kgrid = Eliashberg.grid_summary(params.kpoints) model = Eliashberg.model_summary(params.model) interaction = Eliashberg.interaction_summary(params.interaction) field = Eliashberg.field_summary(params.field)
            @info "Running task" summary = _task_run_summary(task_type, params)

            n_requested = config.system.n_workers
            n_current = nworkers()

            if config.system.bootstrap_workers && n_requested > n_current
                @info "Preparing worker pool..." requested_workers = n_requested current_workers = n_current
                addprocs(
                    n_requested - n_current;
                    exeflags="--project=$(project) --threads=1",
                    restrict=restrict,
                    env=Dict(
                        "JULIA_NUM_THREADS" => "1",
                        "OPENBLAS_NUM_THREADS" => "1",
                        "OMP_NUM_THREADS" => "1",
                        "MKL_NUM_THREADS" => "1",
                    ),
                )
            end

            @info "Loading Eliashberg runtime on workers..." worker_count = nworkers()
            for worker_id in workers()
                remotecall_wait(Core.eval, worker_id, Main, quote
                    import Pkg
                    import LinearAlgebra
                    Pkg.activate($project; io=Base.devnull)
                    try
                        LinearAlgebra.BLAS.set_num_threads(1)
                    catch
                    end
                    using Eliashberg
                end)
            end

            config.system.backup_config && cp(toml_path, joinpath(out_dir, "input_backup.toml"); force=true)

            result = nothing
            time_taken = @elapsed begin
                if task_type == "scan_spectral_function"
                    result = scan_spectral_function(
                        params.model, params.interaction, params.field, params.kpoints,
                        params.task.qpath, params.task.omegas;
                        T=params.task.T_val,
                        η=params.task.eta,
                        bootstrap_workers=config.system.bootstrap_workers,
                        n_workers=n_requested,
                        project=project,
                        restrict=restrict,
                    )
                elseif task_type == "compute_phase_transition_data"
                    result = compute_phase_transition_data(
                        params.model, params.interaction, params.field, params.kpoints;
                        params.task...
                    )
                elseif task_type == "compute_renormalized_band_data"
                    result = compute_renormalized_band_data(
                        params.model, params.interaction, params.field, params.kpoints;
                        params.task...
                    )
                elseif task_type == "compute_collective_mode_spectral_data"
                    result = compute_collective_mode_spectral_data(
                        params.model, params.interaction, params.field, params.kpoints;
                        params.task...
                    )
                elseif task_type == "compute_zeeman_pairing_data"
                    result = compute_zeeman_pairing_data(
                        params.model, params.interaction, params.kpoints;
                        params.task...
                    )
                else
                    error("Unknown task type for automated runner: $(task_type)")
                end
            end

            hdf5_result = if task_type == "scan_spectral_function"
                SpectralMapData(
                    params.task.qpath,
                    params.task.omegas,
                    result,
                    0.0,
                    nothing,
                    params.task.T_val,
                )
            else
                result
            end

            !isnothing(paths.jld2_file) && jldsave(paths.jld2_file; result=result, config=config)
            !isnothing(paths.hdf5_file) && write_result_hdf5(paths.hdf5_file, hdf5_result, config, toml_path; time_seconds=time_taken)
            plot_file = _maybe_save_plot(out_dir, task_type, result, params)
            live_heap_bytes = _safe_live_heap_bytes()
            maxrss_bytes = _safe_maxrss_bytes()

            @info "Job completed successfully" wall_time_seconds = _format_seconds(time_taken) worker_processes = nworkers() result_summary = _result_summary(result) output_directory = out_dir jld2_output = paths.jld2_file hdf5_output = paths.hdf5_file run_log = paths.log_file plot_output = plot_file julia_live_heap = isnothing(live_heap_bytes) ? nothing : _format_bytes(live_heap_bytes) peak_resident_memory = isnothing(maxrss_bytes) ? nothing : _format_bytes(maxrss_bytes)
            return (; result, output_dir=out_dir, jld2_file=paths.jld2_file, hdf5_file=paths.hdf5_file, log_file=paths.log_file, plot_file)
        end
    finally
        if !isnothing(log_io)
            flush(log_io)
            close(log_io)
        end
    end
end

# 允许从命令行直接调用
if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) != 1
        println("Usage: julia src/Tools/runner.jl <path_to_config.toml>")
        exit(1)
    end
    submit_job(ARGS[1])
end
