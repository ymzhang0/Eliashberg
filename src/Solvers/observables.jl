# observables.jl

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
    T=1e-3
)
    optimization_result = Ref{Any}(nothing)

    return with_stage_log(
        "Solve ground state";
        context=(field=field_summary(field), model=model_summary(model), interaction=interaction_summary(interaction), grid=grid_summary(kgrid), approx=approx_summary(approx), phi_guess=phi_guess, T=Float64(T)),
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

        optimization_result[] = optimize(objective, [phi_guess], BFGS())
        !Optim.converged(optimization_result[]) && @warn "Ground-state optimization did not report convergence." field=field_summary(field) approx=approx_summary(approx) phi_guess=phi_guess iterations=Optim.iterations(optimization_result[]) minimum=Optim.minimum(optimization_result[]) T=Float64(T)
        !isfinite_value(Optim.minimum(optimization_result[])) && @warn "Ground-state optimization produced a non-finite objective value." field=field_summary(field) approx=approx_summary(approx) minimum=Optim.minimum(optimization_result[]) T=Float64(T)
        return Optim.minimizer(optimization_result[])[1]
    end
end

_composite_phi_guess(field::CompositeField, phi_guess::Real) = fill(Float64(phi_guess), length(field))

function _composite_phi_guess(field::CompositeField, phi_guess::AbstractVector{<:Real})
    length(field) == length(phi_guess) || throw(DimensionMismatch("Number of fields must match number of phi guesses."))
    return Float64.(phi_guess)
end

function solve_ground_state(
    field::CompositeField,
    model::ElectronicDispersion,
    interaction::Interaction,
    kgrid::AbstractKGrid,
    approx::ApproximationLevel;
    phi_guess=0.1,
    T=1e-3
)
    optimization_result = Ref{Any}(nothing)

    return with_stage_log(
        "Solve ground state";
        context=(field=field_summary(field), model=model_summary(model), interaction=interaction_summary(interaction), grid=grid_summary(kgrid), approx=approx_summary(approx), phi_guess=phi_guess, T=Float64(T)),
        summarize_result=phis -> (
            phis=Float64.(phis),
            converged=Optim.converged(optimization_result[]),
            iterations=Optim.iterations(optimization_result[]),
            minimum=Optim.minimum(optimization_result[]),
        ),
    ) do
        initial_guess = _composite_phi_guess(field, phi_guess)
        objective(phis) = evaluate_action(phis, field, model, interaction, kgrid, approx; T=T)

        optimization_result[] = optimize(objective, initial_guess, LBFGS())
        !Optim.converged(optimization_result[]) && @warn "Composite ground-state optimization did not report convergence." field=field_summary(field) approx=approx_summary(approx) phi_guess=phi_guess iterations=Optim.iterations(optimization_result[]) minimum=Optim.minimum(optimization_result[]) T=Float64(T)
        !isfinite_value(Optim.minimum(optimization_result[])) && @warn "Composite ground-state optimization produced a non-finite objective value." field=field_summary(field) approx=approx_summary(approx) minimum=Optim.minimum(optimization_result[]) T=Float64(T)
        return Float64.(Optim.minimizer(optimization_result[]))
    end
end
