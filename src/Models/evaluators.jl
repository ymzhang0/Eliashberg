# Models/evaluators.jl

# ---------------------------------------------------------
# Electronic Hamiltonian Evaluators (ε)
# ---------------------------------------------------------

"""
    ε(k::Union{AbstractVector{<:Real}, Tuple{Vararg{Real}}}, model::Dispersion{D}) where {D}

Convenience fallback method. Converts a standard array or tuple `k` into an `SVector{D, Float64}`
and delegates to the highly optimized core `ε` methods. This allows users to evaluate
the Hamiltonian interactively without importing StaticArrays.
"""
function ε(k::Union{AbstractVector{<:Real}, Tuple{Vararg{Real}}}, model::Dispersion{D}) where {D}
    length(k) == D || throw(DimensionMismatch("Expected k-point of dimension $D, got $(length(k))"))
    return ε(SVector{D, Float64}(k...), model)
end

"""
    ε(k::SVector{D, Float64}, model::TightBinding{D}) where {D}

Evaluates the tight-binding Hamiltonian at momentum `k` using Fourier transform
over the explicit real-space hopping vectors.
"""
function ε(k::SVector{D,Float64}, model::FreeElectron{D}) where {D}
    E_k = dot(k, k) / (2 * model.mass) - model.EF
    return Hermitian(hcat(E_k))
end

"""
    ε(k::SVector{D, Float64}, model::TightBinding{D}) where {D}

Evaluates the tight-binding Hamiltonian at momentum `k` using Fourier transform
over the explicit real-space hopping vectors.
"""
function ε(k::SVector{D,Float64}, model::TightBinding{D}) where {D}
    # Start with the on-site energy (Fermi level shift)
    E_k = -model.EF

    # Sum over all hopping vectors
    for (R_idx, t_hop) in model.hoppings
        # Calculate real-space vector R = n1*a1 + n2*a2 ...
        R = model.lattice.vectors * R_idx

        # Add Hermitian pair contribution: t * exp(ikR) + t * exp(-ikR) = 2t * cos(k·R)
        E_k += 2 * t_hop * cos(dot(k, R))
    end

    return Hermitian(hcat(E_k))
end

"""
    ε(k::SVector{D,Float64}, model::MultiOrbitalTightBinding{D}) where {D}

Evaluate the Bloch Hamiltonian for a multi-atom-basis tight-binding model.
For each real-space hopping `(i, j, R, t)`, the phase factor is computed from
the exact Cartesian displacement between basis atom `i` in the home cell and
basis atom `j` in the translated cell `R`.
"""
function ε(k::SVector{D,Float64}, model::MultiOrbitalTightBinding{D}) where {D}
    crystal = model.crystal
    lattice = primitive_vectors(crystal)
    basis = crystal.fractional_positions
    n_basis = length(basis)
    hamiltonian = zeros(ComplexF64, n_basis, n_basis)

    for site in 1:n_basis
        hamiltonian[site, site] -= model.EF
    end

    for (atom_i, atom_j, cell_offset_R, hopping) in model.hoppings
        fractional_displacement = basis[atom_j] + SVector{D,Float64}(cell_offset_R) - basis[atom_i]
        delta_r = lattice * fractional_displacement
        phase = exp(im * dot(k, delta_r))
        contribution = hopping * phase

        if atom_i == atom_j && iszero(cell_offset_R)
            hamiltonian[atom_i, atom_i] += real(contribution)
        elseif atom_i == atom_j
            hamiltonian[atom_i, atom_i] += contribution + conj(contribution)
        else
            hamiltonian[atom_i, atom_j] += contribution
            hamiltonian[atom_j, atom_i] += conj(contribution)
        end
    end

    return Hermitian(hamiltonian)
end

function ε(k::SVector{2,Float64}, model::KagomeLattice)
    a1 = model.lattice.vectors[:, 1]
    a2 = model.lattice.vectors[:, 2]

    # Fully vector-based, invariant to lattice scaling/rotation
    h12 = -2 * model.t * cos(dot(k, a1) / 2.0)
    h13 = -2 * model.t * cos(dot(k, a2) / 2.0)
    h23 = -2 * model.t * cos(dot(k, a1 - a2) / 2.0)

    H = @SMatrix [
        -model.EF h12 h13;
        h12 -model.EF h23;
        h13 h23 -model.EF
    ]
    return Hermitian(H)
end

"""
    ε(k::SVector{2, Float64}, model::Graphene)

Evaluates the 2x2 Dirac Hamiltonian for Graphene.
Uses the phase shifts strictly derived from the hexagonal primitive vectors.
"""
function ε(k::SVector{2,Float64}, model::Graphene)
    a1 = model.lattice.vectors[:, 1]
    a2 = model.lattice.vectors[:, 2]

    # The off-diagonal element connects the A and B sublattices
    f_k = 1.0 + exp(-im * dot(k, a1)) + exp(-im * dot(k, a2))
    h_AB = -model.t * f_k

    H = @SMatrix [
        -model.EF h_AB;
        conj(h_AB) -model.EF
    ]
    return Hermitian(H)
end

"""
    ε(k::SVector{1, Float64}, model::SSHModel)

Evaluates the 2x2 Hamiltonian for the 1D SSH topological chain.
"""
function ε(k::SVector{1,Float64}, model::SSHModel)
    a1 = model.lattice.vectors[:, 1]

    # t1 is intra-cell (no phase), t2 is inter-cell (phase shift by a1)
    h_AB = model.t1 + model.t2 * exp(-im * dot(k, a1))

    H = @SMatrix [
        -model.EF h_AB;
        conj(h_AB) -model.EF
    ]
    return Hermitian(H)
end

_matrix_data(H::Hermitian) = parent(H)
_matrix_data(H::AbstractMatrix) = H

function _block_diagonal(A::StaticMatrix{N,N,TA}, B::StaticMatrix{N,N,TB}) where {N,TA,TB}
    T = promote_type(TA, TB)
    return Hermitian(SMatrix{2N,2N,T,4N*N}(ntuple(idx -> begin
        row = (idx - 1) % (2N) + 1
        col = (idx - 1) ÷ (2N) + 1
        if row <= N && col <= N
            return T(A[row, col])
        elseif row > N && col > N
            return T(B[row - N, col - N])
        end
        return zero(T)
    end, 4N*N)))
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

function ε(k::SVector{D,Float64}, model::SpinorDispersion{D}) where {D}
    bare_H = _matrix_data(ε(k, model.bare))
    return _block_diagonal(bare_H, bare_H)
end

function ε(
    k::SVector{D,Float64},
    model::RenormalizedDispersion{D}
) where {D}
    bare_H = ε(k, model.bare_dispersion)
    Σ_H = Σ(k, model.self_energy)
    return Hermitian(bare_H + Σ_H)
end


# ---------------------------------------------------------
# Phonon Dispersions Evaluators (ω)
# ---------------------------------------------------------

"""
    ω(q::SVector{D,Float64}, d::MonoatomicLatticeModel{D})

Computes the acoustic phonon dispersion strictly using the real-space lattice vectors.
"""
function ω(q::SVector{D,Float64}, d::MonoatomicLatticeModel{D}) where {D}
    val = 0.0
    # Sum over the primitive lattice vectors dynamically
    for i in 1:D
        a_vec = d.lattice.vectors[:, i]
        val += 1.0 - cos(dot(q, a_vec))
    end
    return sqrt(2 * d.K / d.M * val)
end

ω(q::SVector{D,Float64}, model::EinsteinModel) where {D} = model.ωE
ω(q::SVector{D,Float64}, model::DebyeModel) where {D} = model.vs * norm(q)
ω(q::SVector{D,Float64}, d::PolaritonModel) where {D} = sqrt(d.ωE^2 + (d.vs * norm(q))^2)


# ---------------------------------------------------------
# General Band Structure
# ---------------------------------------------------------

"""
    band_structure(disp, k)

Returns an `Eigen` object for the Hamiltonian matrix at momentum `k`.
"""
function band_structure(disp::ElectronicDispersion{D}, k::SVector{D,Float64}) where {D}
    H = ε(k, disp)
    return eigen(H)
end
