# susceptibilities.jl

# --- Generalized Susceptibility ---
"""
    GeneralizedSusceptibility{M, G, F}

Functor that maps external parameters to a response value while reducing over an
internal weighted grid. The auxiliary field selects the vertex operator, while
`T` and `η` parameterize the kernel evaluation.
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

Evaluate the response by reducing over the internal grid with
`Engine.integrate_grid`. The outer call only specifies the external parameters;
the weighted reduction is delegated to the pure execution engine.
"""
function (chi::GeneralizedSusceptibility)(fluct::DynamicalFluctuation{D}) where {D}
    if !(chi.grid isa AbstractKGrid{D})
        throw(DimensionMismatch("Fluctuation momentum dimension ($D) does not match grid dimension"))
    end

    return Engine.integrate_grid(SusceptibilityReductionKernel(chi, fluct), chi.grid)
end

# Support for static evaluation via SVector
function (chi::GeneralizedSusceptibility)(Q::SVector{D,<:Real}) where {D}
    return chi(DynamicalFluctuation(Q, 0.0))
end

struct SusceptibilityReductionKernel{C,F}
    chi::C
    fluct::F
end

SusceptibilityReductionKernel(chi::GeneralizedSusceptibility, fluct::DynamicalFluctuation) =
    SusceptibilityReductionKernel{typeof(chi),typeof(fluct)}(chi, fluct)

function (kernel::SusceptibilityReductionKernel)(k::SVector{D,Float64}) where {D}
    response_sum = 0.0im

    eig_k = band_structure(kernel.chi.model, k)
    eig_kq = band_structure(kernel.chi.model, k + kernel.fluct.q)
    vertex = _susceptibility_vertex(kernel.chi.field, k)

    for m in eachindex(eig_k.values)
        for n in eachindex(eig_kq.values)
            energy_m = real(eig_k.values[m])
            energy_n = real(eig_kq.values[n])

            vec_m = eig_k.vectors[:, m]
            vec_n = eig_kq.vectors[:, n]
            coherence = abs2(dot(vec_n, vertex * vec_m))

            coherence <= 1e-10 && continue

            occ_m = _fermi_weight(energy_m, kernel.chi.T)
            occ_n = _fermi_weight(energy_n, kernel.chi.T)

            if abs(energy_n - energy_m) < 1e-8 && abs(kernel.fluct.ω) < 1e-8
                response_sum += coherence * (-_fermi_derivative(occ_m, kernel.chi.T))
            else
                denominator = (energy_n - energy_m) - kernel.fluct.ω - 1im * kernel.chi.η
                response_sum += coherence * (occ_m - occ_n) / denominator
            end
        end
    end

    return response_sum
end

_susceptibility_vertex(field::ParticleHoleChannel, k::SVector) = vertex_matrix(field)
_susceptibility_vertex(field::BCSReducedPairing, k::SVector) = vertex_matrix(k, field)
_susceptibility_vertex(field::FFLOPairing, k::SVector) = vertex_matrix(k, field)
_susceptibility_vertex(field::PairDensityWave, k::SVector) = vertex_matrix(k, field)

function _fermi_weight(energy::Float64, temperature::Float64)
    scaled = energy / temperature

    if scaled > 50.0
        return 0.0
    elseif scaled < -50.0
        return 1.0
    end

    return 1.0 / (exp(scaled) + 1.0)
end

_fermi_derivative(occupation::Float64, temperature::Float64) = occupation * (occupation - 1.0) / temperature
