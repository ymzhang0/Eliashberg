
function renormalized_dispersion(disp::ElectronicDispersion{D}, Σ::SelfEnergy, k::SVector{D,Float64}, ω::Float64) where {D}
    ε0 = ε(k, disp)
    Σval = Σ(k, ω)
    return Hermitian(ε0 + Σval)
end

"""
    solve_bcs(kgrid::AbstractKGrid{D}, dispersion_model::ElectronicDispersion, interaction_model::Interaction) where {D}

Solve the generalized linearized BCS gap equation over a `KGrid`. 
Assumes a single dominant band closest to the Fermi level for the constructed N x N eigenvalue problem.
"""
function solve_bcs(
    kgrid::AbstractKGrid{D},
    dispersion_model::ElectronicDispersion{D},
    interaction_model::Interaction,
) where {D}
    N = length(kgrid)
    H = zeros(Float64, N, N)

    # Precompute kinetic energies (taking the first band for simplistic 1-band gap equation)
    Ek = zeros(Float64, N)
    for i in 1:N
        # We assume single band near Fermi surface for the scalar gap equation
        Ek[i] = band_structure(dispersion_model, kgrid[i]).values[1]
    end

    for i in 1:N
        for j in 1:N
            if i == j
                H[i, j] += Ek[i]
            end
            # In proper KGrid integration, we multiply the interaction by the integration weight of j
            H[i, j] += V(kgrid[i], kgrid[j], interaction_model, dispersion_model) * kgrid.weights[j]
        end
    end

    eig = eigen(H)
    return eig.values, eig.vectors
end