
"""
    solve_ground_state(field, model, interaction, kgrid, approx; phi_guess=0.1, T=1e-3)

Finds the order parameter `phi` that minimizes the effective action.
"""
function solve_ground_state(
    field::AuxiliaryField,
    model::ElectronicDispersion,
    interaction::Interaction,
    kgrid::AbstractKGrid,
    approx::ApproximationLevel;
    phi_guess=0.1,
    T=1e-3,
    log_level::LogLevel=Logging.Info
)
    optimization_result = Ref{Any}(nothing)

    return with_stage_log(
        "Solve ground state";
        level=log_level,
        context=(field=field, model=model, interaction=interaction, grid=kgrid, approx=approx, phi_guess=phi_guess, T=Float64(T)),
        summarize_result=phi -> (
            phi=Float64(phi),
            converged=Optim.converged(optimization_result[]),
            iterations=Optim.iterations(optimization_result[]),
            minimum=Optim.minimum(optimization_result[]),
        ),
    ) do
        objective(phi_array) = evaluate_action(
            phi_array[1], field, model, interaction, kgrid, approx; T=T
        )

        options = Optim.Options(callback=_optimization_progress_callback("Solve ground state"))
        optimization_result[] = optimize(objective, [phi_guess], BFGS(), options)
        !Optim.converged(optimization_result[]) && @warn "Ground-state optimization did not report convergence." field = field approx = approx phi_guess = phi_guess iterations = Optim.iterations(optimization_result[]) minimum = Optim.minimum(optimization_result[]) T = Float64(T)
        !isfinite_value(Optim.minimum(optimization_result[])) && @warn "Ground-state optimization produced a non-finite objective value." field = field approx = approx minimum = Optim.minimum(optimization_result[]) T = Float64(T)
        return Optim.minimizer(optimization_result[])[1]
    end
end

function solve_ground_state(
    field::CompositeField,
    model::ElectronicDispersion,
    interaction::Interaction,
    kgrid::AbstractKGrid,
    approx::ApproximationLevel;
    phi_guess=0.1,
    T=1e-3,
    log_level::LogLevel=Logging.Info
)
    optimization_result = Ref{Any}(nothing)

    return with_stage_log(
        "Solve ground state";
        level=log_level,
        context=(field=field, model=model, interaction=interaction, grid=kgrid, approx=approx, phi_guess=phi_guess, T=Float64(T)),
        summarize_result=phis -> (
            phis=Float64.(phis),
            converged=Optim.converged(optimization_result[]),
            iterations=Optim.iterations(optimization_result[]),
            minimum=Optim.minimum(optimization_result[]),
        ),
    ) do
        initial_guess = _initial_phi_guess(field, phi_guess)
        objective(phis) = evaluate_action(phis, field, model, interaction, kgrid, approx; T=T)

        options = Optim.Options(callback=_optimization_progress_callback("Solve ground state"))
        optimization_result[] = optimize(objective, initial_guess, LBFGS(), options)
        !Optim.converged(optimization_result[]) && @warn "Composite ground-state optimization did not report convergence." field = field approx = approx phi_guess = phi_guess iterations = Optim.iterations(optimization_result[]) minimum = Optim.minimum(optimization_result[]) T = Float64(T)
        !isfinite_value(Optim.minimum(optimization_result[])) && @warn "Composite ground-state optimization produced a non-finite objective value." field = field approx = approx minimum = Optim.minimum(optimization_result[]) T = Float64(T)
        return Float64.(Optim.minimizer(optimization_result[]))
    end
end
