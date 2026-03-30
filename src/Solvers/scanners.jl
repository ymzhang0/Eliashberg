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

function scan_instability_landscape(
    model::PhysicalModel,
    kgrid::KGrid{D},
    qgrid::KGrid{D};
    T=0.001,
    η=1e-3,
    bootstrap_workers::Bool=false,
    n_workers::Integer=max(0, Threads.nthreads() - 1),
    project::Union{Nothing,AbstractString}=Base.active_project(),
    restrict::Bool=true
) where {D}
    chi_functor = _charge_susceptibility(model, kgrid, T, η)
    @info "Scanning instability landscape over $(length(qgrid)) q-points..."
    axes = _parameter_axes(qgrid)

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

"""
    scan_rpa_spectral_function_hpc(model::PhysicalModel, kgrid::AbstractKGrid{D}, qaxis::AbstractVector{SVector{D,Float64}}, omegas::AbstractVector{<:Real}; T=0.001, η=0.05) where {D}

High-level wrapper that maps a two-parameter space and delegates the internal
weighted reduction to `GeneralizedSusceptibility`. The wrapper only defines the
physics task and passes it to `Engine.distributed_map_grid`. Set
`bootstrap_workers=true` to provision distributed map workers automatically.
"""
function scan_rpa_spectral_function_hpc(
    model::PhysicalModel,
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
    chi_functor = _charge_susceptibility(model, kgrid, T, η)
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

"""
    scan_spectral_function(model::PhysicalModel, kgrid::AbstractKGrid{D}, qpath::KPath{D}, omegas::AbstractVector{Float64}; T=0.001, η=0.05) where {D}

Map a two-dimensional parameter space over `(q, ω)` and reduce each point over
the internal grid with `GeneralizedSusceptibility`. Set
`bootstrap_workers=true` to provision distributed map workers automatically.
"""
function scan_spectral_function(
    model::PhysicalModel,
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
    @info "Scanning spectral function over $(length(qpath)) parameter samples and $(length(omegas)) frequency samples..."
    return scan_rpa_spectral_function_hpc(
        model,
        kgrid,
        qpath.points,
        omegas;
        T=T,
        η=η,
        bootstrap_workers=bootstrap_workers,
        n_workers=n_workers,
        project=project,
        restrict=restrict
    )
end

function _parameter_axes(grid::KGrid{D}) where {D}
    axes = ntuple(dim -> unique(sort([point[dim] for point in grid.points])), D)
    return prod(length.(axes)) == length(grid) ? axes : (grid.points,)
end

function _point_from_coordinates(::Val{D}, coords::Vararg{<:Real,D}) where {D}
    return SVector{D,Float64}(ntuple(dim -> Float64(coords[dim]), Val(D)))
end
