# susceptibilities.jl
using StaticArrays

"""
    LindhardSusceptibility{M, G}

Functor to evaluate static polarization bubble χ₀(Q) for a given physical model,
k-grid, and temperature.
"""
struct LindhardSusceptibility{M<:PhysicalModel, G<:AbstractKGrid}
    model::M
    grid::G
    T::Float64
end

function (chi::LindhardSusceptibility)(Q::SVector{D, Float64}) where {D}
    # \\chi_0(Q) = - (1/N) \\sum_k (f(\\epsilon_k) - f(\\epsilon_{k+Q})) / (\\epsilon_k - \\epsilon_{k+Q})
    res = 0.0
    N = length(chi.grid)
    for i in 1:N
        k = chi.grid.points[i]
        w = chi.grid.weights[i]
        
        ek = real(ε(k, chi.model)[1,1])
        ek_q = real(ε(k + Q, chi.model)[1,1])
        
        # Fermi-Dirac distribution
        fk = 1.0 / (exp(ek / chi.T) + 1.0)
        fk_q = 1.0 / (exp(ek_q / chi.T) + 1.0)
        
        if abs(ek - ek_q) > 1e-10
            res += w * (fk - fk_q) / (ek - ek_q)
        else
            # derivative of fermi function: - (1/T) * exp(e/T) / (exp(e/T)+1)^2
            df = - (1.0 / chi.T) * exp(ek / chi.T) / (exp(ek / chi.T) + 1.0)^2
            res += w * df
        end
    end
    return -res # Convention: positive susceptibility
end
