# Pauli matrices
using LinearAlgebra

export σ₀, σ₁, σ₂, σ₃, pauli_matrices, γ⁰, γ¹, γ², γ³, gamma_matrices, commutator, anticommutator

# σ₀ (identity matrix)
const σ₀ = Complex{Float64}[1 0; 0 1]
const σ₁ = Complex{Float64}[0 1; 1 0]
const σ₂ = Complex{Float64}[0 -im; im 0]
const σ₃ = Complex{Float64}[1 0; 0 -1]

const pauli_matrices = [σ₁, σ₂, σ₃]

# Gamma matrices
const Z₂ = identity(Complex{Float64}, 2, 2)

const γ⁰ = Z₂ * σ₃

const γ¹ = [
    Z₂   σ₁;
    -σ₁   Z₂
    ]

const γ² = [
    Z₂   σ₂;
    -σ₂   Z₂
    ]

const γ³ = [
    Z₂   σ₃;
    -σ₃   Z₂
    ]

const γ⁵ = [
    σ₀   Z₂;
    Z₂   σ₀
    ]

const γ⁰¹ = commutator(γ⁰, γ¹) / (2 * im)
const γ⁰² = commutator(γ⁰, γ²) / (2 * im)
const γ⁰³ = commutator(γ⁰, γ³) / (2 * im)
const γ¹² = commutator(γ¹, γ²) / (2 * im)
const γ¹³ = commutator(γ¹, γ³) / (2 * im)
const γ²³ = commutator(γ², γ³) / (2 * im)

const gamma_matrices = [γ⁰, γ¹, γ², γ³, γ⁰¹, γ⁰², γ⁰³, γ¹², γ¹³, γ²³]

function commutator(A::Matrix{Complex{Float64}}, B::Matrix{Complex{Float64}})
    return A * B - B * A
end

function anticommutator(A::Matrix{Complex{Float64}}, B::Matrix{Complex{Float64}})
    return A * B + B * A
end

