# src/Tools/runner.jl

using Pkg
Pkg.activate(".") # 确保激活环境
using Distributed
using Dates
using TOML
using LinearAlgebra
using Logging
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

function _open_job_logger(log_file::Union{Nothing,AbstractString})
    isnothing(log_file) && return (current_logger(), nothing)

    mkpath(dirname(log_file))
    io = open(log_file, "w")
    file_logger = SimpleLogger(io, Logging.Info)
    return (TeeLogger((current_logger(), file_logger)), io)
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
    logger, log_io = _open_job_logger(paths.log_file)

    try
        return with_logger(logger) do
            @info "Parsing cluster job specification..." file = toml_path output_dir = out_dir log_file = paths.log_file

            params = build_from_config(config)
            task_type = task_type_name(config.task)
            project = something(config.system.project, Base.active_project(), ".")
            restrict = config.system.restrict
            _configure_blas_threads!(1)

            n_requested = config.system.n_workers
            n_current = nworkers()

            if config.system.bootstrap_workers && n_requested > n_current
                @info "Bootstrapping cluster workers..." requested = n_requested
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

            @info "Loading Eliashberg.jl on all $(nworkers()) workers..."
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

            @info "Dispatching task to physics engine..." task_type = task_type

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

            @info "Job completed successfully!" time_seconds = time_taken output_dir = out_dir jld2_file = paths.jld2_file hdf5_file = paths.hdf5_file plot_file = plot_file
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
