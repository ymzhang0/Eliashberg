# src/Models/Hamiltonian.jl

using StaticArrays
using LinearAlgebra

# ---------------------------------------------------------
# Electronic Hamiltonian Operators (H)
# ---------------------------------------------------------

"""
    H(k::SVector{D, Float64}, model::ElectronicDispersion{D}) where {D}

Evaluate the Hermitian Hamiltonian matrix for an electronic dispersion at momentum `k`.
"""
function H end

# Fallback for generic vectors/tuples
function H(k::Union{AbstractVector{<:Real},Tuple{Vararg{Real}}}, model::ElectronicDispersion{D}) where {D}
    length(k) == D || throw(DimensionMismatch("Expected k-point of dimension $D, got $(length(k))"))
    return H(SVector{D,Float64}(k...), model)
end

function H(k::SVector{D,Float64}, model::FreeElectron{D}) where {D}
    E_k = dot(k, k) / (2 * model.mass) - model.EF
    return Hermitian(hcat(E_k))
end

function H(k::SVector{D,Float64}, model::TightBinding{D}) where {D}
    E_k = -model.EF
    for (R_idx, t_hop) in model.hoppings
        R = model.lattice * R_idx
        E_k += 2 * t_hop * cos(dot(k, R))
    end
    return Hermitian(hcat(E_k))
end

function H(k::SVector{D,Float64}, model::MultiOrbitalTightBinding{D}) where {D}
    lattice = primitive_vectors(model)
    n_basis = model.num_orbitals
    hamiltonian = zeros(ComplexF64, n_basis, n_basis)

    for site in 1:n_basis
        hamiltonian[site, site] -= model.EF
    end

    for (atom_i, atom_j, cell_offset_R, hopping) in model.hoppings
        delta_r = lattice * SVector{D,Float64}(cell_offset_R)
        phase = exp(im * dot(k, delta_r))
        hamiltonian[atom_i, atom_j] += hopping * phase
    end

    return Hermitian((hamiltonian + hamiltonian') / 2)
end

function H(k::SVector{D,Float64}, model::SpinorDispersion{D}) where {D}
    bare_H = _matrix_data(H(k, model.bare))
    return _block_diagonal(bare_H, bare_H)
end

function H(k::SVector{D,Float64}, model::RenormalizedDispersion{D}) where {D}
    bare_H = H(k, model.bare_dispersion)
    Σ_H = Σ(k, model.self_energy)
    return Hermitian(bare_H + Σ_H)
end

# ---------------------------------------------------------
# Electronic Energy Spectrum (ε)
# ---------------------------------------------------------

"""
    ε(k, model::ElectronicDispersion)

Returns the energy eigenvalues (spectrum) of the electronic model at momentum `k`.
For 1x1 models, returns a scalar Float64. For multi-orbital models, returns a Vector.
"""
function ε(k::SVector{D,Float64}, model::ElectronicDispersion{D}) where {D}
    matrix = H(k, model)
    if size(matrix) == (1, 1)
        return real(matrix[1, 1])
    else
        return eigen(matrix).values
    end
end

"""
    diagonalize(k, model::ElectronicDispersion)

Returns the full eigenvalue decomposition of the electronic model at momentum `k`.
Returns an `Eigen` object containing `.values` and `.vectors`.
"""
function diagonalize(k::SVector{D,Float64}, model::ElectronicDispersion{D}) where {D}
    return eigen(H(k, model))
end

function diagonalize(k::Union{AbstractVector{<:Real},Tuple{Vararg{Real}}}, model::ElectronicDispersion{D}) where {D}
    length(k) == D || throw(DimensionMismatch("Expected k-point of dimension $D, got $(length(k))"))
    return diagonalize(SVector{D,Float64}(k...), model)
end

# Convenience fallback
function ε(k::Union{AbstractVector{<:Real},Tuple{Vararg{Real}}}, model::ElectronicDispersion{D}) where {D}
    length(k) == D || throw(DimensionMismatch("Expected k-point of dimension $D, got $(length(k))"))
    return ε(SVector{D,Float64}(k...), model)
end

# ---------------------------------------------------------
# Phonon Dynamical Operators (D)
# ---------------------------------------------------------

"""
    D(q::SVector{D,Float64}, model::PhononDispersion{D}) where {D}

Evaluate the Dynamical Matrix for a phonon dispersion at momentum `q`.
"""
function D end

function D(q::SVector{D,Float64}, model::MonoatomicLatticeModel{D}) where {D}
    freq = ω(q, model)
    return Hermitian(hcat(freq^2))
end

function D(q::SVector{D,Float64}, model::EinsteinModel) where {D}
    return Hermitian(hcat(model.ωE^2))
end

function D(q::SVector{D,Float64}, model::DebyeModel) where {D}
    return Hermitian(hcat((model.vs * norm(q))^2))
end

# ---------------------------------------------------------
# Phonon Frequency Spectrum (ω)
# ---------------------------------------------------------

"""
    ω(q, model::PhononDispersion)

Returns the phonon frequency eigenvalues (spectrum) at momentum `q`.
"""
function ω(q::SVector{D,Float64}, d::MonoatomicLatticeModel{D}) where {D}
    val = 0.0
    for i in 1:D
        a_vec = d.lattice[:, i]
        val += 1.0 - cos(dot(q, a_vec))
    end
    return sqrt(abs(2 * d.K / d.M * val))
end

ω(q::SVector{D,Float64}, model::EinsteinModel) where {D} = model.ωE
ω(q::SVector{D,Float64}, model::DebyeModel) where {D} = model.vs * norm(q)
ω(q::SVector{D,Float64}, d::PolaritonModel) where {D} = sqrt(d.ωE^2 + (d.vs * norm(q))^2)

# Convenience fallback
function ω(q::Union{AbstractVector{<:Real},Tuple{Vararg{Real}}}, model::PhononDispersion{D}) where {D}
    length(q) == D || throw(DimensionMismatch("Expected q-point of dimension $D, got $(length(q))"))
    return ω(SVector{D,Float64}(q...), model)
end

# ---------------------------------------------------------
# Internal Utilities
# ---------------------------------------------------------

_matrix_data(H::Hermitian) = parent(H)
_matrix_data(H::AbstractMatrix) = H

function _block_diagonal(A::StaticMatrix{N,N,TA}, B::StaticMatrix{N,N,TB}) where {N,TA,TB}
    T = promote_type(TA, TB)
    return Hermitian(SMatrix{2N,2N,T,4N * N}(ntuple(idx -> begin
            row = (idx - 1) % (2N) + 1
            col = (idx - 1) ÷ (2N) + 1
            if row <= N && col <= N
                return T(A[row, col])
            elseif row > N && col > N
                return T(B[row-N, col-N])
            end
            return zero(T)
        end, 4N * N)))
end

function _block_diagonal(A::AbstractMatrix{TA}, B::AbstractMatrix{TB}) where {TA,TB}
    size(A, 1) == size(A, 2) || throw(DimensionMismatch("Left block must be square."))
    size(B, 1) == size(B, 2) || throw(DimensionMismatch("Right block must be square."))

    T = promote_type(TA, TB)
    if size(A) == (1, 1) && size(B) == (1, 1)
        return Hermitian(@SMatrix [T(A[1, 1]) zero(T); zero(T) T(B[1, 1])])
    end

    n_left = size(A, 1)
    n_right = size(B, 1)
    matrix = zeros(T, n_left + n_right, n_left + n_right)
    matrix[1:n_left, 1:n_left] = A
    matrix[n_left+1:end, n_left+1:end] = B
    return Hermitian(matrix)
end
