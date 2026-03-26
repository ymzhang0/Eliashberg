# susceptibilities.jl
using StaticArrays

"""
    LindhardSusceptibility{M, G}

Functor to evaluate the dynamical polarization bubble χ₀(q, ω) for a given physical model,
k-grid, temperature, and broadening η.
"""
struct LindhardSusceptibility{M<:PhysicalModel, G<:AbstractKGrid}
    model::M
    grid::G
    T::Float64
    η::Float64 # Broadening parameter for the dynamical susceptibility
end

# Default constructor with small broadening
LindhardSusceptibility(model, grid, T) = LindhardSusceptibility(model, grid, T, 1e-3)

"""
    (chi::LindhardSusceptibility)(field::DynamicalFluctuation)

Evaluates the dynamical Lindhard susceptibility:
χ(q, ω) = - (1/N) * ∑_k [f(ε_k) - f(ε_{k+q})] / [ (ε_{k+q} - ε_k) - ω - iη ]
"""
function (chi::LindhardSusceptibility)(field::DynamicalFluctuation{D}) where {D}
    q = field.q
    ω = field.ω
    η = chi.η
    T = chi.T
    
    res = 0.0 + 0.0im
    N = length(chi.grid)
    
    for i in 1:N
        k = chi.grid.points[i]
        w = chi.grid.weights[i]
        
        # Dispersion ε(k) from the model
        ek = real(ε(k, chi.model)[1,1])
        ek_q = real(ε(k + q, chi.model)[1,1])
        
        # Fermi-Dirac distribution f(ε)
        fk = 1.0 / (exp(ek / T) + 1.0)
        fk_q = 1.0 / (exp(ek_q / T) + 1.0)
        
        # Dynamical Lindhard denominator: (ε_{k+q} - ε_k) - ω - iη
        denominator = (ek_q - ek) - ω - 1im * η
        
        res += w * (fk - fk_q) / denominator
    end
    
    return res # Return the complex susceptibility; peak is typically positive real part
end

# Support for static evaluation via SVector for backward compatibility or convenience
function (chi::LindhardSusceptibility)(Q::SVector{D, <:Real}) where {D}
    return chi(DynamicalFluctuation(Q, 0.0))
end
