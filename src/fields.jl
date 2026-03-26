# fields.jl
using LinearAlgebra
using StaticArrays

"""
    reconstructed_bands(k, phi, field::AuxiliaryField, model::PhysicalModel)

Returns the eigenvalues of the reconstructed band structure in the presence of symmetry breaking
parameterized by `phi` and `field`.
"""
function reconstructed_bands(k::SVector{D, Float64}, phi::Float64, field::ChargeDensityWave{D}, model::PhysicalModel) where {D}
    # For CDW, the original band couples k and k+Q modes via the order parameter phi.
    # We construct the 2x2 Hamiltonian matrix.
    
    # original dispersion at k
    H11 = real(ε(k, model)[1,1])
    
    # original dispersion at k+Q 
    # (Note: here we assume ε expects SVector, so k + field.q works directly)
    k_plus_q = k + field.q
    H22 = real(ε(k_plus_q, model)[1,1])
    
    H12 = phi
    H21 = phi
    
    H_cdw = Hamiltonian2x2(H11, H22, H12, H21)
    
    # We just return the eigenvalues of this 2x2 matrix analytically
    return eigenvals_2x2(H_cdw)
end

# Helper to compute eigenvalues for 2x2 symmetric real matrix
struct Hamiltonian2x2
    H11::Float64
    H22::Float64
    H12::Float64
    H21::Float64
end

function eigenvals_2x2(H::Hamiltonian2x2)
    # Tr H = H11 + H22
    # Det H = H11*H22 - H12*H21
    # lambda = (Tr ± sqrt(Tr^2 - 4*Det)) / 2
    tr = H.H11 + H.H22
    det = H.H11 * H.H22 - H.H12 * H.H21
    
    discriminant = sqrt(max(0.0, tr^2 - 4 * det)) # max to prevent domain error due to floating point
    
    lam1 = (tr - discriminant) / 2
    lam2 = (tr + discriminant) / 2
    
    return SVector{2, Float64}(lam1, lam2)
end
