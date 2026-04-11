"""
    scan_instability_landscape(model::PhysicalModel, kgrid::KGrid{D}, qgrid::KGrid{D}; T=0.001, η=1e-3) where {D}

Map the static external parameter space and reduce each sample over the
internal grid through `GeneralizedSusceptibility`. The solver wrapper only
packages the physics-specific callable and delegates execution to `Engine`.
Set `bootstrap_workers=true` to provision distributed map workers automatically.
"""
function _charge_susceptibility(model::PhysicalModel, kgrid::KGrid{D}, T, η) where {D}
    return GeneralizedSusceptibility(model, kgrid, ChargeDensityWave(zero(SVector{D,Float64})), T, η)
end

struct StaticFluctuationTask{C}
    susceptibility::C
end

function (task::StaticFluctuationTask)(q::SVector{D,Float64}) where {D}
    return real(task.susceptibility(DynamicalFluctuation(q, 0.0)))
end

struct StaticCoordinateFluctuationTask{D,C}
    susceptibility::C
end

function (task::StaticCoordinateFluctuationTask{D})(coords::Vararg{<:Real,D}) where {D}
    q = _point_from_coordinates(Val(D), coords...)
    return real(task.susceptibility(DynamicalFluctuation(q, 0.0)))
end

struct SpectralFunctionTask{C}
    susceptibility::C
end

function (task::SpectralFunctionTask)(q::SVector{D,Float64}, omega::Real) where {D}
    return imag(task.susceptibility(DynamicalFluctuation(q, Float64(omega))))
end

struct RPASpectralFunctionTask{D, C, I<:Interaction}
    susceptibility::C
    interaction::I
end

function (task::RPASpectralFunctionTask{D})(q::SVector{D,Float64}, omega::Real) where {D}
    # 1. 计算复数裸极化率 χ₀(q, ω)
    chi0 = task.susceptibility(DynamicalFluctuation(q, Float64(omega)))
    
    # 2. 计算当前 q 下的库仑排斥力 V(q)
    vq = V(q, task.interaction)
    
    # 3. RPA Dyson 方程：当 1 - V*Re(χ₀) 接近 0 时，虚部会产生极锐的等离激元共振峰
    denominator = 1.0 - vq * chi0
    !isfinite_value(denominator) && @warn "RPA denominator became non-finite during spectral scan." q=q omega=Float64(omega) interaction=interaction_summary(task.interaction) chi0=chi0 denominator=denominator
    abs(denominator) <= 100 * eps(Float64) && @warn "RPA denominator is numerically singular during spectral scan." q=q omega=Float64(omega) interaction=interaction_summary(task.interaction) chi0=chi0 denominator=denominator
    chi_rpa = chi0 / denominator
    !isfinite_value(chi_rpa) && @warn "RPA susceptibility became non-finite during spectral scan." q=q omega=Float64(omega) interaction=interaction_summary(task.interaction) chi0=chi0 denominator=denominator chi_rpa=chi_rpa
    
    return imag(chi_rpa)
end

function scan_instability_landscape(
    model::PhysicalModel,
    kgrid::KGrid{D},
    qgrid::KGrid{D};
    field::AuxiliaryField=ChargeDensityWave(zero(SVector{D,Float64})),
    T=0.001,
    η=1e-3,
    bootstrap_workers::Bool=false,
    n_workers::Integer=max(0, Threads.nthreads() - 1),
    project::Union{Nothing,AbstractString}=Base.active_project(),
    restrict::Bool=true
) where {D}
    return with_stage_log(
        "Scan instability landscape";
        context=(model=model_summary(model), field=field_summary(field), kgrid=grid_summary(kgrid), qgrid=grid_summary(qgrid), T=Float64(T), eta=Float64(η), bootstrap_workers=bootstrap_workers, requested_workers=Int(n_workers)),
        summarize_result=result -> (result_type=string(typeof(result)), size=size(result)),
    ) do
        chi_functor = GeneralizedSusceptibility(model, kgrid, field, T, η)
        axes = _parameter_axes(qgrid)

        @timeit TO "Instability Scan" begin
            if length(axes) == 1 && eltype(axes[1]) <: SVector{D,Float64}
                return Engine.distributed_map_grid(
                    StaticFluctuationTask(chi_functor),
                    axes[1];
                    bootstrap_workers=bootstrap_workers,
                    n_workers=n_workers,
                    project=project,
                    restrict=restrict
                )
            end

            return Engine.distributed_map_grid(
                StaticCoordinateFluctuationTask{D,typeof(chi_functor)}(chi_functor),
                axes...;
                bootstrap_workers=bootstrap_workers,
                n_workers=n_workers,
                project=project,
                restrict=restrict
            )
        end
    end
end

"""
    scan_rpa_spectral_function_hpc(model::PhysicalModel, kgrid::AbstractKGrid{D}, qaxis::AbstractVector{SVector{D,Float64}}, omegas::AbstractVector{<:Real}; T=0.001, η=0.05) where {D}

High-level wrapper that maps a two-parameter space and delegates the internal
weighted reduction to `GeneralizedSusceptibility`. The wrapper only defines the
physics task and passes it to `Engine.distributed_map_grid`. Set
`bootstrap_workers=true` to provision distributed map workers automatically.
"""
function scan_rpa_spectral_function_hpc(
    model::PhysicalModel,
    interaction::Interaction, 
    field::AuxiliaryField,
    kgrid::AbstractKGrid{D},
    qaxis::AbstractVector{SVector{D,Float64}},
    omegas::AbstractVector{<:Real};
    T=0.001,
    η=0.05,
    bootstrap_workers::Bool=false,
    n_workers::Integer=max(0, Threads.nthreads() - 1),
    project::Union{Nothing,AbstractString}=Base.active_project(),
    restrict::Bool=true
) where {D}
    return with_stage_log(
        "Scan RPA spectral function";
        context=(model=model_summary(model), interaction=interaction_summary(interaction), field=field_summary(field), kgrid=grid_summary(kgrid), qaxis=axis_summary(qaxis), omegas=axis_summary(omegas), T=Float64(T), eta=Float64(η), bootstrap_workers=bootstrap_workers, requested_workers=Int(n_workers)),
        summarize_result=result -> (result_type=string(typeof(result)), size=size(result)),
    ) do
        chi_functor = GeneralizedSusceptibility(model, kgrid, field, T, η)

        return Engine.distributed_map_grid(
            RPASpectralFunctionTask{D, typeof(chi_functor), typeof(interaction)}(chi_functor, interaction),
            qaxis,
            omegas;
            bootstrap_workers=bootstrap_workers,
            n_workers=n_workers,
            project=project,
            restrict=restrict
        )
    end
end

function scan_rpa_spectral_function_hpc(
    model::PhysicalModel,
    kgrid::AbstractKGrid{D},
    qaxis::AbstractVector{SVector{D,Float64}},
    omegas::AbstractVector{<:Real};
    field::AuxiliaryField=ChargeDensityWave(zero(SVector{D,Float64})),
    T=0.001,
    η=0.05,
    bootstrap_workers::Bool=false,
    n_workers::Integer=max(0, Threads.nthreads() - 1),
    project::Union{Nothing,AbstractString}=Base.active_project(),
    restrict::Bool=true
) where {D}
    return with_stage_log(
        "Scan spectral function";
        context=(model=model_summary(model), field=field_summary(field), kgrid=grid_summary(kgrid), qaxis=axis_summary(qaxis), omegas=axis_summary(omegas), T=Float64(T), eta=Float64(η), bootstrap_workers=bootstrap_workers, requested_workers=Int(n_workers)),
        summarize_result=result -> (result_type=string(typeof(result)), size=size(result)),
    ) do
        chi_functor = GeneralizedSusceptibility(model, kgrid, field, T, η)

        return Engine.distributed_map_grid(
            SpectralFunctionTask(chi_functor),
            qaxis,
            omegas;
            bootstrap_workers=bootstrap_workers,
            n_workers=n_workers,
            project=project,
            restrict=restrict
        )
    end
end

"""
    scan_spectral_function(model::PhysicalModel, kgrid::AbstractKGrid{D}, qpath::KPath{D}, omegas::AbstractVector{Float64}; T=0.001, η=0.05) where {D}

Map a two-dimensional parameter space over `(q, ω)` and reduce each point over
the internal grid with `GeneralizedSusceptibility`. Set
`bootstrap_workers=true` to provision distributed map workers automatically.
"""
function scan_spectral_function(
    model::PhysicalModel,
    interaction::Interaction, 
    field::AuxiliaryField,
    kgrid::AbstractKGrid{D},
    qpath::KPath{D},
    omegas::AbstractVector{Float64};
    T=0.001,
    η=0.05,
    bootstrap_workers::Bool=false,
    n_workers::Integer=max(0, Threads.nthreads() - 1),
    project::Union{Nothing,AbstractString}=Base.active_project(),
    restrict::Bool=true
) where {D}
    return with_stage_log(
        "Scan spectral function along path";
        context=(model=model_summary(model), interaction=interaction_summary(interaction), field=field_summary(field), kgrid=grid_summary(kgrid), qpath=kpath_summary(qpath), omegas=axis_summary(omegas), T=Float64(T), eta=Float64(η), bootstrap_workers=bootstrap_workers, requested_workers=Int(n_workers)),
        summarize_result=result -> (result_type=string(typeof(result)), size=size(result)),
    ) do
        @timeit TO "Spectral Function Scan" return scan_rpa_spectral_function_hpc(
            model,
            interaction, 
            field,
            kgrid,
            path_points(qpath),
            omegas;
            T=T,
            η=η,
            bootstrap_workers=bootstrap_workers,
            n_workers=n_workers,
            project=project,
            restrict=restrict
        )
    end
end

function scan_spectral_function(
    model::PhysicalModel,
    kgrid::AbstractKGrid{D},
    qpath::KPath{D},
    omegas::AbstractVector{Float64};
    field::AuxiliaryField=ChargeDensityWave(zero(SVector{D,Float64})),
    T=0.001,
    η=0.05,
    bootstrap_workers::Bool=false,
    n_workers::Integer=max(0, Threads.nthreads() - 1),
    project::Union{Nothing,AbstractString}=Base.active_project(),
    restrict::Bool=true
) where {D}
    return with_stage_log(
        "Scan spectral function along path";
        context=(model=model_summary(model), field=field_summary(field), kgrid=grid_summary(kgrid), qpath=kpath_summary(qpath), omegas=axis_summary(omegas), T=Float64(T), eta=Float64(η), bootstrap_workers=bootstrap_workers, requested_workers=Int(n_workers)),
        summarize_result=result -> (result_type=string(typeof(result)), size=size(result)),
    ) do
        return scan_rpa_spectral_function_hpc(
            model,
            kgrid,
            path_points(qpath),
            omegas;
            field=field,
            T=T,
            η=η,
            bootstrap_workers=bootstrap_workers,
            n_workers=n_workers,
            project=project,
            restrict=restrict
        )
    end
end

function _parameter_axes(grid::KGrid{D}) where {D}
    axes = ntuple(dim -> unique(sort([point[dim] for point in grid.points])), D)
    return prod(length.(axes)) == length(grid) ? axes : (grid.points,)
end

function _point_from_coordinates(::Val{D}, coords::Vararg{<:Real,D}) where {D}
    return SVector{D,Float64}(ntuple(dim -> Float64(coords[dim]), Val(D)))
end
