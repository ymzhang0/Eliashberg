# effective_action.jl
using StaticArrays

"""
    evaluate(action::EffectiveAction, phi::Float64, ::ExactTrLn; T::Float64=1e-3)

Evaluates the non-linear effective action by computing the Tr[ln] term over
the reconstructed energy bands.
"""
function evaluate(action::EffectiveAction, phi::Float64, ::ExactTrLn; T::Float64=1e-3)
    # F(phi) = N * phi^2 / V - T * \\sum_{k, \\lambda} \\ln(1 + e^{-\\beta \\lambda(k)})
    
    term1 = phi^2 / action.V_bare
    
    tr_ln_sum = 0.0
    for i in 1:length(action.grid)
        k = action.grid.points[i]
        w = action.grid.weights[i]
        
        lambdas = reconstructed_bands(k, phi, action.field, action.model)
        
        for lam in lambdas
            if lam < 0
                tr_ln_sum += w * (lam - T * log1p(exp(lam / T)))
            else
                tr_ln_sum += w * (- T * log1p(exp(-lam / T)))
            end
        end
    end
    
    return term1 + tr_ln_sum
end

"""
    evaluate(action::EffectiveAction, phi::Float64, ::RPA; T::Float64=1e-3)

Evaluates the effective action using the quadratic RPA expansion.
F(phi) \\approx (1/V - \\chi_0(Q)) phi^2
"""
function evaluate(action::EffectiveAction, phi::Float64, ::RPA; T::Float64=1e-3)
    chi0 = LindhardSusceptibility(action.model, action.grid, T)
    
    # Evaluate susceptibility at the auxiliary field wavevector
    chi_val = chi0(action.field.Q)
    
    return (1.0 / action.V_bare - chi_val) * phi^2
end
