# src/Tools/runner.jl

using Pkg
Pkg.activate(".") # 确保激活环境
using Distributed
using Dates
using TOML
using LinearAlgebra
using JLD2        # 用于无损保存 Julia 数据结构

# 加载核心物理代码及其中嵌套的 Config 模块
using Eliashberg
using Eliashberg.Config: EliashbergConfig, Configurations, build_from_config
include("exports.jl")

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
    @info "Parsing cluster job specification..." file = toml_path

    # 1. 解析 TOML
    config = Configurations.from_toml(EliashbergConfig, toml_path)
    params = build_from_config(config)
    task_type = task_type_name(config.task)
    project = something(config.system.project, Base.active_project(), ".")
    restrict = config.system.restrict
    _configure_blas_threads!(1)

    # 2. 自动化分布式集群配置
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

    # 确保所有 worker 都加载了 Eliashberg
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

    # 3. 创建时间戳输出目录
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    out_dir = joinpath("results", "job_" * timestamp)
    mkpath(out_dir)

    # 备份配置文件，保证绝对的可重复性
    cp(toml_path, joinpath(out_dir, "input_backup.toml"))

    # 4. 路由并执行任务 (路由表)
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

    # 5. 保存结果落盘
    jld2_file = joinpath(out_dir, "data.jld2")
    hdf5_file = joinpath(out_dir, "data.h5")
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
    jldsave(jld2_file; result=result, config=config)
    write_result_hdf5(hdf5_file, hdf5_result, config, toml_path; time_seconds=time_taken)
    plot_file = _maybe_save_plot(out_dir, task_type, result, params)

    @info "Job completed successfully!" time_seconds = time_taken output_dir = out_dir jld2_file = jld2_file hdf5_file = hdf5_file plot_file = plot_file
end

# 允许从命令行直接调用
if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) != 1
        println("Usage: julia src/Tools/runner.jl <path_to_config.toml>")
        exit(1)
    end
    submit_job(ARGS[1])
end
