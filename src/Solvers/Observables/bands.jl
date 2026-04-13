
_allocate_gap_storage(::AuxiliaryField, n_temperatures::Int) = zeros(Float64, n_temperatures)
_allocate_gap_storage(field::CompositeField, n_temperatures::Int) = zeros(Float64, length(field), n_temperatures)

"""
    compute_renormalized_band_data(Ts, field, model, interaction, kgrid, kpath; approx=ExactTrLn(), phi_guess=0.5)

Compute the mean-field gap and the corresponding renormalized bands for each
temperature sample along a path in parameter space.
"""
function compute_renormalized_band_data(
    model::ElectronicDispersion,
    interaction::Interaction,
    field::AuxiliaryField,
    kgrid::AbstractKGrid;
    kpath::KPath,
    Ts::AbstractVector{<:Real},
    approx::ApproximationLevel=ExactTrLn(),
    phi_guess=0.5,
    warm_start::Bool=true
)
    return with_stage_log(
        "Compute renormalized band data";
        context=(field=field, model=model, interaction=interaction, kgrid=kgrid, kpath=kpath, approx=approx, n_temperatures=length(Ts), warm_start=warm_start),
        summarize_result=data -> (n_temperatures=length(data.temperatures), n_bands=size(data.bare_bands, 2)),
    ) do
        gaps = _allocate_gap_storage(field, length(Ts))
        fallback_guess = _initial_phi_guess(field, phi_guess)
        band_matrices = Vector{Matrix{Float64}}(undef, length(Ts))
        _scan_parameter_axis(
            Ts;
            warm_start=warm_start,
            initial_state=fallback_guess,
            progress_name="Computing Renormalized Bands",
        ) do idx, T, current_guess
            phi_gs = _solve_regularized_ground_state(
                field,
                model,
                interaction,
                kgrid,
                approx;
                phi_guess=current_guess,
                T=T,
                log_level=Logging.Debug,
            )
            _store_gap!(gaps, idx, phi_gs)

            renormalized_dispersion = MeanFieldDispersion(model, field, phi_gs)
            band_matrices[idx] = _band_matrix_along_path(renormalized_dispersion, kpath)
            return _next_phi_guess(phi_gs, fallback_guess)
        end

        renormalized_bands = stack(band_matrices)
        bare_bands = _band_matrix_along_path(model, kpath)

        return RenormalizedBandData(
            kpath=kpath,
            bare_bands=Float64.(bare_bands),
            renormalized_bands=Float64.(renormalized_bands),
            gaps=gaps,
            temperatures=Float64.(Ts)
        )
    end
end

function compute_renormalized_band_data(
    Ts::AbstractVector{<:Real},
    field::AuxiliaryField,
    model::ElectronicDispersion,
    interaction::Interaction,
    kgrid::AbstractKGrid,
    kpath::KPath;
    kwargs...
)
    return compute_renormalized_band_data(
        model,
        interaction,
        field,
        kgrid;
        kpath=kpath,
        Ts=Ts,
        kwargs...,
    )
end

function compute_renormalized_band_data(
    Ts::AbstractVector{<:Real},
    field::CompositeField,
    model::ElectronicDispersion,
    interaction::Interaction,
    kgrid::AbstractKGrid,
    kpath::KPath;
    kwargs...
)
    return compute_renormalized_band_data(
        model,
        interaction,
        field,
        kgrid;
        kpath=kpath,
        Ts=Ts,
        kwargs...,
    )
end
