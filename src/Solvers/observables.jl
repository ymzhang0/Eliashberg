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
    # Build the scalar objective as a one-parameter closure.
    objective(phi_array) = evaluate_action(
        phi_array[1], field, model, interaction, kgrid, approx; T=T
    )

    # Delegate the minimization to Optim.jl.
    res = optimize(objective, [phi_guess], BFGS())

    return Optim.minimizer(res)[1]
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
    initial_guess = _composite_phi_guess(field, phi_guess)
    objective(phis) = evaluate_action(phis, field, model, interaction, kgrid, approx; T=T)

    res = optimize(objective, initial_guess, LBFGS())

    return Float64.(Optim.minimizer(res))
end
