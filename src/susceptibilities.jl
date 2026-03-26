# susceptibilities.jl
using StaticArrays, LinearAlgebra

# --- Vertex Matrix Interface ---
"""
    vertex_matrix(field::AuxiliaryField)

Returns the symmetry-breaking coupling (vertex) matrix for computing quantum 
coherence factors. Default is the identity matrix for density channels.
"""
vertex_matrix(::AuxiliaryField) = 1.0

"""
    vertex_matrix(::SuperconductingPairing)

For superconducting pairing, the order parameter couples electron and hole 
components in Nambu space, corresponding to the Pauli-X matrix.
"""
vertex_matrix(::SuperconductingPairing) = SA[0.0 1.0; 1.0 0.0]

# --- Generalized Susceptibility ---
"""
    GeneralizedSusceptibility{M, G, F}

Functor to evaluate the generalized polarization bubble χ₀(q, ω) for a given 
physical model, k-grid, auxiliary field (determining the vertex matrix), temperature, 
and broadening η.
"""
struct GeneralizedSusceptibility{M<:PhysicalModel,G<:AbstractKGrid,F<:AuxiliaryField}
    model::M
    grid::G
    field::F
    T::Float64
    η::Float64 # Broadening parameter for the dynamical susceptibility
end

# Default constructor with small broadening
GeneralizedSusceptibility(model, grid, field, T) = GeneralizedSusceptibility(model, grid, field, T, 1e-3)

"""
    (chi::GeneralizedSusceptibility)(fluct::DynamicalFluctuation)

Evaluates the generalized dynamical susceptibility, incorporating exact eigensystems 
and coherence factors based on the auxiliary field's vertex matrix.
"""
function (chi::GeneralizedSusceptibility)(fluct::DynamicalFluctuation{D}) where {D}
    # Ensure fluctuation momentum dimension matches grid dimension
    if !(chi.grid isa AbstractKGrid{D})
        throw(DimensionMismatch("Fluctuation momentum dimension ($D) does not match grid dimension"))
    end

    q = fluct.q
    ω = fluct.ω
    η = chi.η
    T = chi.T

    res = 0.0im
    N = length(chi.grid)

    V_mat = vertex_matrix(chi.field)

    for i in 1:N
        k = chi.grid.points[i]
        w = chi.grid.weights[i]

        # Calculate full eigensystem
        eig_k = band_structure(chi.model, k)
        eig_kq = band_structure(chi.model, k + q)

        n_bands_k = length(eig_k.values)
        n_bands_kq = length(eig_kq.values)

        # Loop over ALL band combinations m (for k) and n (for k+q)
        for m in 1:n_bands_k
            for n in 1:n_bands_kq
                E_m = real(eig_k.values[m])
                E_n = real(eig_kq.values[n])

                # Coherence Factor (Matrix Element)
                u_m = eig_k.vectors[:, m]
                u_n = eig_kq.vectors[:, n]
                M_mn = abs2(dot(u_n, V_mat * u_m))

                if M_mn > 1e-10
                    f_m = (E_m / T > 50.0) ? 0.0 : 1.0 / (exp(E_m / T) + 1.0)
                    f_n = (E_n / T > 50.0) ? 0.0 : 1.0 / (exp(E_n / T) + 1.0)

                    # Check for static intra-band limit to avoid 0/0 NaN
                    if abs(E_n - E_m) < 1e-8 && abs(ω) < 1e-8
                        # Use L'Hopital's rule (derivative of Fermi function)
                        # df/dE = -exp(E_m/T) / (T * (exp(E_m/T) + 1)^2)
                        # df_dE = -exp(E_m / T) / (T * (exp(E_m / T) + 1.0)^2)
                        df_dE = f_m * (f_m - 1.0) / T

                        # In the Lindhard formula, term is -(df/dE)
                        res += w * M_mn * (-df_dE)
                    else
                        denominator = (E_n - E_m) - ω - 1im * η
                        res += w * M_mn * (f_m - f_n) / denominator
                    end
                end
            end
        end
    end

    return res # Return the complex susceptibility
end

# Support for static evaluation via SVector
function (chi::GeneralizedSusceptibility)(Q::SVector{D,<:Real}) where {D}
    return chi(DynamicalFluctuation(Q, 0.0))
end
