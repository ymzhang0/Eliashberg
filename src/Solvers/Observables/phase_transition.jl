
function compute_phase_transition_data(
    model::ElectronicDispersion,
    interaction::Interaction,
    field::AuxiliaryField,
    kgrid::AbstractKGrid;
    phis::AbstractVector{<:Real},
    Ts::AbstractVector{<:Real},
    approx::ApproximationLevel=ExactTrLn(),
    phi_guess::Real=0.2,
    warm_start::Bool=true
)
    return with_stage_log(
        "Compute phase transition data";
        context=(field=field, model=model, interaction=interaction, kgrid=kgrid, approx=approx, n_phis=length(phis), n_temperatures=length(Ts), warm_start=warm_start),
        summarize_result=data -> (n_phis=length(data.phis), n_temperatures=length(data.Ts)),
    ) do
        free_energy = zeros(Float64, length(phis), length(Ts))
        condensation_energy = zeros(Float64, length(phis), length(Ts))
        order_parameters = zeros(Float64, length(Ts))
        fallback_guess = _initial_phi_guess(field, phi_guess)
        _scan_parameter_axis(
            Ts;
            warm_start=warm_start,
            initial_state=fallback_guess,
            progress_name="Scanning Temperatures",
        ) do idx, T, current_guess
            energy_curve = _evaluate_action_curve(phis, field, model, interaction, kgrid, approx; T=T)
            free_energy[:, idx] = energy_curve
            condensation_energy[:, idx] = energy_curve .- energy_curve[1]

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
            order_parameters[idx] = phi_gs
            return _next_phi_guess(phi_gs, fallback_guess)
        end

        return PhaseDiagramData(
            phis=Float64.(phis),
            Ts=Float64.(Ts),
            free_energy=free_energy,
            condensation_energy=condensation_energy,
            order_parameters=order_parameters
        )
    end
end

function compute_phase_transition_data(
    phis::AbstractVector{<:Real},
    Ts::AbstractVector{<:Real},
    field::AuxiliaryField,
    model::ElectronicDispersion,
    interaction::Interaction,
    kgrid::AbstractKGrid;
    kwargs...
)
    return compute_phase_transition_data(
        model,
        interaction,
        field,
        kgrid;
        phis=phis,
        Ts=Ts,
        kwargs...,
    )
end

function compute_phase_transition_data(
    model::ElectronicDispersion,
    interaction::Interaction,
    field::CompositeField,
    kgrid::AbstractKGrid;
    phis::AbstractVector{<:Real},
    Ts::AbstractVector{<:Real},
    approx::ApproximationLevel=ExactTrLn(),
    phi_guess=0.2
)
    throw(ArgumentError("compute_phase_transition_data supports a one-dimensional order-parameter scan. Use compute_coexistence_landscape for CompositeField scans."))
end

function compute_phase_transition_data(
    phis::AbstractVector{<:Real},
    Ts::AbstractVector{<:Real},
    field::CompositeField,
    model::ElectronicDispersion,
    interaction::Interaction,
    kgrid::AbstractKGrid;
    kwargs...
)
    return compute_phase_transition_data(
        model,
        interaction,
        field,
        kgrid;
        phis=phis,
        Ts=Ts,
        kwargs...,
    )
end


function compute_coexistence_landscape(
    model::ElectronicDispersion,
    interaction::Interaction,
    comp::CompositeField,
    kgrid::AbstractKGrid;
    phis_1::AbstractVector{<:Real},
    phis_2::AbstractVector{<:Real},
    T::Real,
    approx::ApproximationLevel=ExactTrLn()
)
    length(comp) == 2 || throw(DimensionMismatch("compute_coexistence_landscape requires a CompositeField with exactly two fields."))

    free_energy = zeros(Float64, length(phis_1), length(phis_2))
    temperature = Float64(T)

    Threads.@threads for idx_1 in eachindex(phis_1)
        phi_1 = phis_1[idx_1]
        for (idx_2, phi_2) in enumerate(phis_2)
            phi_point = SVector{2,Float64}(Float64(phi_1), Float64(phi_2))
            free_energy[idx_1, idx_2] = evaluate_action(phi_point, comp, model, interaction, kgrid, approx; T=temperature)
        end
    end

    field_1, field_2 = comp.fields
    return CoexistenceLandscapeData(
        phis_1,
        phis_2,
        free_energy,
        string(typeof(field_1)),
        string(typeof(field_2))
    )
end

function compute_coexistence_landscape(
    phis_1::AbstractVector{<:Real},
    phis_2::AbstractVector{<:Real},
    comp::CompositeField,
    model::ElectronicDispersion,
    interaction::Interaction,
    kgrid::AbstractKGrid;
    kwargs...
)
    return compute_coexistence_landscape(
        model,
        interaction,
        comp,
        kgrid;
        phis_1=phis_1,
        phis_2=phis_2,
        kwargs...,
    )
end

"""
    compute_zeeman_pairing_data(T_val, h_val, q_vals, model, interaction, kgrid; approx=ExactTrLn(), phi_guess=0.4)

Map a one-dimensional external parameter axis to optimized order parameters and
condensation energies for FFLO-style scans.
"""
function compute_zeeman_pairing_data(
    model::ElectronicDispersion,
    interaction::Interaction,
    kgrid::AbstractKGrid;
    T_val::Real,
    h_val::Real,
    q_vals::AbstractVector{<:Real},
    approx::ApproximationLevel=ExactTrLn(),
    phi_guess::Real=0.4,
    warm_start::Bool=true
)
    return with_stage_log(
        "Compute Zeeman pairing data";
        context=(model=model, interaction=interaction, kgrid=kgrid, T=Float64(T_val), h=Float64(h_val), n_q=length(q_vals), approx=approx, warm_start=warm_start),
        summarize_result=data -> (n_q=length(data.q_vals), optimal_q=data.optimal_q, minimum_index=data.minimum_index),
    ) do
        dim = length(first(kgrid.points))
        minimal_energy = zeros(Float64, length(q_vals))
        optimal_gaps = zeros(Float64, length(q_vals))
        fallback_guess = _initial_phi_guess(BCSReducedPairing(), phi_guess)
        _scan_parameter_axis(
            q_vals;
            warm_start=warm_start,
            initial_state=fallback_guess,
            progress_name="Scanning Zeeman Pairing",
        ) do idx, q, current_guess
            q_vector = SVector{dim,Float64}(ntuple(i -> i == 1 ? Float64(q) : 0.0, dim))
            fflo_field = FFLOPairing(q_vector, h_val)
            phi_gs = _solve_regularized_ground_state(
                fflo_field,
                model,
                interaction,
                kgrid,
                approx;
                phi_guess=current_guess,
                T=T_val,
                log_level=Logging.Debug,
            )

            optimal_gaps[idx] = phi_gs
            minimal_energy[idx] = evaluate_action(phi_gs, fflo_field, model, interaction, kgrid, approx; T=T_val)
            return _next_phi_guess(phi_gs, fallback_guess)
        end

        zero_q = zero(SVector{dim,Float64})
        normal_energy = evaluate_action(0.0, FFLOPairing(zero_q, h_val), model, interaction, kgrid, approx; T=T_val)
        condensation_energy = minimal_energy .- normal_energy
        minimum_index = argmin(condensation_energy)

        return ZeemanPairingData(
            q_vals,
            condensation_energy,
            optimal_gaps,
            q_vals[minimum_index],
            minimum_index
        )
    end
end
