# src/Tools/runner.jl

using Pkg
Pkg.activate(".") # 确保激活环境
using Distributed
using Dates
using TOML
using JLD2        # 用于无损保存 Julia 数据结构

# 加载我们重构好的 Config 模块
include("../Config/Config.jl")
using .Config: EliashbergConfig, Configurations, build_from_config

# 只在主进程加载核心物理代码
using Eliashberg

function submit_job(toml_path::String)
    @info "Parsing cluster job specification..." file = toml_path

    # 1. 解析 TOML
    config = Configurations.from_toml(EliashbergConfig, toml_path)
    params = build_from_config(config)

    # 2. 自动化分布式集群配置
    n_requested = config.system.n_workers
    n_current = nworkers()

    if config.system.bootstrap_workers && n_requested > n_current
        @info "Bootstrapping cluster workers..." requested = n_requested
        addprocs(n_requested - n_current; exeflags="--project=.")
    end

    # 确保所有 worker 都加载了 Eliashberg
    @info "Loading Eliashberg.jl on all $(nworkers()) workers..."
    @everywhere using Eliashberg

    # 3. 创建时间戳输出目录
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    out_dir = joinpath("results", "job_" * timestamp)
    mkpath(out_dir)

    # 备份配置文件，保证绝对的可重复性
    cp(toml_path, joinpath(out_dir, "input_backup.toml"))

    # 4. 路由并执行任务 (路由表)
    @info "Dispatching task to physics engine..." task_type = config.task.type

    result = nothing
    time_taken = @elapsed begin
        if config.task.type == "scan_spectral_function"
            result = scan_spectral_function(
                params.model, params.interaction, params.field, params.kpoints,
                params.task.qpath, params.task.omegas;
                T=params.task.T_val, η=params.task.eta
            )
        elseif config.task.type == "compute_phase_transition_data"
            result = compute_phase_transition_data(
                params.model, params.interaction, params.field, params.kpoints;
                params.task...
            )
        elseif config.task.type == "compute_renormalized_band_data"
            result = compute_renormalized_band_data(
                params.model, params.interaction, params.field, params.kpoints;
                params.task...
            )
        elseif config.task.type == "compute_collective_mode_spectral_data"
            result = compute_collective_mode_spectral_data(
                params.model, params.interaction, params.field, params.kpoints;
                params.task...
            )
        elseif config.task.type == "compute_zeeman_pairing_data"
            result = compute_zeeman_pairing_data(
                params.model, params.interaction, params.kpoints;
                params.task...
            )
        else
            error("Unknown task type for automated runner: $(config.task.type)")
        end
    end

    # 5. 保存结果落盘
    save_file = joinpath(out_dir, "data.jld2")
    jldsave(save_file; result=result, config=config)

    @info "Job completed successfully!" time_seconds = time_taken output_dir = out_dir
end

# 允许从命令行直接调用
if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) != 1
        println("Usage: julia src/Tools/runner.jl <path_to_config.toml>")
        exit(1)
    end
    submit_job(ARGS[1])
end