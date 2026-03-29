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
    # 动态构建只依赖于 phi 的目标函数闭包
    objective(phi_array) = evaluate_action(
        phi_array[1], field, model, interaction, kgrid, approx; T=T
    )

    # 扔给 Optim 求解
    res = optimize(objective, [phi_guess], BFGS())

    return Optim.minimizer(res)[1]
end