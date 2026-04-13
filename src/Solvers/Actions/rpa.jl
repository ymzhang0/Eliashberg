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

struct SpectralFunctionRowTask{C,O}
    susceptibility::C
    omegas::O
end

function (task::SpectralFunctionRowTask)(q::SVector{D,Float64}) where {D}
    return imag.(susceptibility_spectrum(task.susceptibility, q, task.omegas))
end

struct RPASpectralFunctionTask{D,C,I<:Interaction}
    susceptibility::C
    interaction::I
end

const _RPA_SINGULAR_DENOMINATOR_ATOL = 100 * eps(Float64)

function _rpa_spectral_weight(
    chi0,
    vq,
    q,
    omega::Real,
    interaction::Interaction,
)
    denominator = 1.0 - vq * chi0
    !isfinite_value(denominator) && @warn "RPA denominator became non-finite during spectral scan." q = q omega = Float64(omega) interaction = interaction chi0 = chi0 denominator = denominator
    if abs(denominator) <= _RPA_SINGULAR_DENOMINATOR_ATOL
        @warn "RPA denominator is numerically singular during spectral scan. Returning NaN." q = q omega = Float64(omega) interaction = interaction chi0 = chi0 denominator = denominator
        return NaN
    end

    chi_rpa = chi0 / denominator
    !isfinite_value(chi_rpa) && @warn "RPA susceptibility became non-finite during spectral scan." q = q omega = Float64(omega) interaction = interaction chi0 = chi0 denominator = denominator chi_rpa = chi_rpa
    return imag(chi_rpa)
end

function (task::RPASpectralFunctionTask{D})(q::SVector{D,Float64}, omega::Real) where {D}
    chi0 = task.susceptibility(DynamicalFluctuation(q, Float64(omega)))
    vq = V(q, task.interaction)
    return _rpa_spectral_weight(chi0, vq, q, omega, task.interaction)
end

struct RPASpectralFunctionRowTask{D,C,I<:Interaction,O}
    susceptibility::C
    interaction::I
    omegas::O
end

function (task::RPASpectralFunctionRowTask{D})(q::SVector{D,Float64}) where {D}
    chi0_spectrum = susceptibility_spectrum(task.susceptibility, q, task.omegas)
    vq = V(q, task.interaction)
    spectral_row = Vector{Float64}(undef, length(task.omegas))

    for idx in eachindex(task.omegas)
        spectral_row[idx] = _rpa_spectral_weight(
            chi0_spectrum[idx],
            vq,
            q,
            task.omegas[idx],
            task.interaction,
        )
    end

    return spectral_row
end

function _scan_progress_name(kind::AbstractString, n_points::Integer, n_omegas::Integer)
    return string(kind, " (", n_points, " q-points x ", n_omegas, " omegas)")
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
        context=(model=model, field=field, kgrid=kgrid, qgrid=qgrid, T=Float64(T), eta=Float64(η), bootstrap_workers=bootstrap_workers, requested_workers=Int(n_workers)),
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
        context=(model=model, interaction=interaction, field=field, kgrid=kgrid, qaxis=qaxis, omegas=omegas, T=Float64(T), eta=Float64(η), bootstrap_workers=bootstrap_workers, requested_workers=Int(n_workers)),
        summarize_result=result -> (result_type=string(typeof(result)), size=size(result)),
    ) do
        chi_functor = GeneralizedSusceptibility(model, kgrid, field, T, η)
        row_data = Engine.distributed_map_grid(
            RPASpectralFunctionRowTask{D,typeof(chi_functor),typeof(interaction),typeof(omegas)}(chi_functor, interaction, omegas),
            qaxis;
            bootstrap_workers=bootstrap_workers,
            n_workers=n_workers,
            project=project,
            restrict=restrict,
            progress_name=_scan_progress_name("RPA spectral scan along q-path", length(qaxis), length(omegas))
        )

        return _stack_spectral_rows(row_data, length(omegas))
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
        context=(model=model, field=field, kgrid=kgrid, qaxis=qaxis, omegas=omegas, T=Float64(T), eta=Float64(η), bootstrap_workers=bootstrap_workers, requested_workers=Int(n_workers)),
        summarize_result=result -> (result_type=string(typeof(result)), size=size(result)),
    ) do
        chi_functor = GeneralizedSusceptibility(model, kgrid, field, T, η)
        row_data = Engine.distributed_map_grid(
            SpectralFunctionRowTask{typeof(chi_functor),typeof(omegas)}(chi_functor, omegas),
            qaxis;
            bootstrap_workers=bootstrap_workers,
            n_workers=n_workers,
            project=project,
            restrict=restrict,
            progress_name=_scan_progress_name("Spectral scan along q-path", length(qaxis), length(omegas))
        )

        return _stack_spectral_rows(row_data, length(omegas))
    end
end

function _stack_spectral_rows(rows::AbstractVector{<:AbstractVector{<:Real}}, n_omegas::Integer)
    n_q = length(rows)
    matrix = Matrix{Float64}(undef, n_q, n_omegas)

    for idx in eachindex(rows)
        length(rows[idx]) == n_omegas || throw(DimensionMismatch("Spectral row length must match the omega axis length."))
        matrix[idx, :] .= rows[idx]
    end

    return matrix
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
        context=(model=model, interaction=interaction, field=field, kgrid=kgrid, qpath=qpath, omegas=omegas, T=Float64(T), eta=Float64(η), bootstrap_workers=bootstrap_workers, requested_workers=Int(n_workers)),
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
        context=(model=model, field=field, kgrid=kgrid, qpath=qpath, omegas=omegas, T=Float64(T), eta=Float64(η), bootstrap_workers=bootstrap_workers, requested_workers=Int(n_workers)),
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

"""
    compute_collective_mode_spectral_data(T_val, field, model, interaction, kgrid, qpath; omega_max_factor=5.0, n_omegas=100, eta=0.02, approx=ExactTrLn(), phi_guess=0.4, kwargs...)

Compute the dynamical spectral map of a mean-field state and return only pure
array data for downstream visualization.
"""
function compute_collective_mode_spectral_data(
    model::ElectronicDispersion,
    interaction::Interaction,
    field::AuxiliaryField,
    kgrid::AbstractKGrid;
    qpath::KPath,
    T_val::Real,
    omega_max_factor::Real=5.0,
    n_omegas::Integer=100,
    eta::Real=0.02,
    approx::ApproximationLevel=ExactTrLn(),
    phi_guess::Real=0.4,
    bootstrap_workers::Bool=false,
    n_workers::Integer=max(0, Threads.nthreads() - 1),
    project::Union{Nothing,AbstractString}=Base.active_project(),
    restrict::Bool=true
)
    return with_stage_log(
        "Compute collective mode spectral data";
        context=(field=field, model=model, interaction=interaction, kgrid=kgrid, qpath=qpath, T=Float64(T_val), eta=Float64(eta), n_omegas=Int(n_omegas), approx=approx),
        summarize_result=data -> (n_q=length(path_points(data.qpath)), n_omegas=length(data.omegas), gap=data.gap),
    ) do
        phi_gs = solve_ground_state(field, model, interaction, kgrid, approx; phi_guess=Float64(phi_guess), T=T_val, log_level=Logging.Debug)
        phi_gs = phi_gs < 1e-4 ? 0.0 : phi_gs

        bdg_dispersion = MeanFieldDispersion(model, field, phi_gs)
        omega_max = phi_gs > 0.0 ? omega_max_factor * phi_gs : 2.0
        omegas = collect(range(0.0, omega_max, length=n_omegas))
        spectral_matrix = scan_rpa_spectral_function_hpc(
            bdg_dispersion,
            interaction,
            field,
            kgrid,
            path_points(qpath),
            omegas;
            T=T_val,
            η=eta,
            bootstrap_workers=bootstrap_workers,
            n_workers=n_workers,
            project=project,
            restrict=restrict
        )

        pair_breaking_edge = phi_gs > 0.0 ? 2.0 * phi_gs : nothing
        return SpectralMapData(
            qpath=qpath,
            omegas=omegas,
            spectral_matrix=spectral_matrix,
            gap=phi_gs,
            pair_breaking_edge=pair_breaking_edge,
            temperature=Float64(T_val)
        )
    end
end

function compute_collective_mode_spectral_data(
    T_val::Real,
    field::AuxiliaryField,
    model::ElectronicDispersion,
    interaction::Interaction,
    kgrid::AbstractKGrid,
    qpath::KPath;
    kwargs...
)
    return compute_collective_mode_spectral_data(
        model,
        interaction,
        field,
        kgrid;
        qpath=qpath,
        T_val=T_val,
        kwargs...,
    )
end
