# observables.jl
using Optim

"""
    solve_ground_state(action::EffectiveAction, approx::ApproximationLevel; phi_guess=0.1)

Finds the order parameter `phi` that minimizes the effective action under the chosen approximation.
"""
function solve_ground_state(action::EffectiveAction, approx::ApproximationLevel; phi_guess=0.1)
    # Define the objective function for a scalar parameter
    objective(phi_array) = evaluate(action, phi_array[1], approx)
    
    # We use optimize from Optim.jl to find the minimum
    res = optimize(objective, [phi_guess], BFGS())
    
    return Optim.minimizer(res)[1]
end
