# susceptibilities.jl

# --- Vertex Matrix Interface ---
"""
    vertex_matrix(field::AuxiliaryField)

Returns the symmetry-breaking coupling (vertex) matrix for computing quantum 
coherence factors. Default is the identity matrix for density channels.
"""
vertex_matrix(::AuxiliaryField) = 1.0

"""
    vertex_matrix(::BCSReducedPairing)

For standard BCS pairing, the order parameter couples electron and hole 
components in a 2x2 Nambu space, corresponding to the Pauli-X matrix.
"""
vertex_matrix(::BCSReducedPairing) = SA[0.0 1.0; 1.0 0.0]

"""
    vertex_matrix(::FFLOPairing)

For FFLO pairing, the order parameter couples the electron at `k` to the hole at `-k+q`
in a 2x2 Nambu space. The vertex is identical to the BCS case in structure.
"""
vertex_matrix(::FFLOPairing) = SA[0.0 1.0; 1.0 0.0]

"""
    vertex_matrix(::PairDensityWave)

For a standing-wave Pair Density Wave, the order parameter couples the electron at `k` 
to holes at both `-k+q` and `-k-q` in a 3x3 extended Nambu space.
"""
vertex_matrix(::PairDensityWave) = SA[
    0.0 1.0 1.0;
    1.0 0.0 0.0;
    1.0 0.0 0.0
]

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

struct SusceptibilityReductionKernel{C,F,V}
    chi::C
    fluct::F
    vertex::V
end

SusceptibilityReductionKernel(chi::GeneralizedSusceptibility, fluct::DynamicalFluctuation) =
    SusceptibilityReductionKernel(chi, fluct, vertex_matrix(chi.field))

function (kernel::SusceptibilityReductionKernel)(k::SVector{D,Float64}) where {D}
    response_sum = 0.0im

    eig_k = band_structure(kernel.chi.model, k)
    eig_kq = band_structure(kernel.chi.model, k + kernel.fluct.q)

    for m in eachindex(eig_k.values)
        for n in eachindex(eig_kq.values)
            energy_m = real(eig_k.values[m])
            energy_n = real(eig_kq.values[n])

            vec_m = eig_k.vectors[:, m]
            vec_n = eig_kq.vectors[:, n]
            coherence = abs2(dot(vec_n, kernel.vertex * vec_m))

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
