using LinearAlgebra
using StaticArrays

struct FreeElectron{D} <: ElectronicDispersion{D}
    EF::Float64
    mass::Float64
end
FreeElectron{D}(EF::Float64) where D = FreeElectron{D}(EF, 1.0) # default mass=1

struct TightBinding{D} <: ElectronicDispersion{D}
    t::Float64
    tp::Float64
    EF::Float64
end
TightBinding{D}(t, EF) where D = TightBinding{D}(t, 0.0, EF)

struct EinsteinModel{D} <: PhononDispersion{D}
    ωE::Float64
end

struct DebyeModel{D} <: PhononDispersion{D}
    vs::Float64
    ωD::Float64
end

struct PolaritonModel{D} <: PhononDispersion{D}
    ωE::Float64  # Einstein frequency
    vs::Float64  # sound velocity
end

struct MonoatomicLatticeModel{D} <: PhononDispersion{D}
    K::Float64  # spring constant
    M::Float64  # mass
    a::Float64  # lattice constant
end

struct RenormalizedDispersion{D,M<:ElectronicDispersion{D},S<:SelfEnergy} <: ElectronicDispersion{D}
    bare_dispersion::M
    self_energy::S
end

struct MeanFieldDispersion{D,M<:ElectronicDispersion{D},F<:AuxiliaryField} <: ElectronicDispersion{D}
    bare_dispersion::M
    field::F
    phi::Float64
end

function MeanFieldDispersion(bare::M, field::ChargeDensityWave{D}, phi::Real) where {D,M<:ElectronicDispersion{D}}
    return MeanFieldDispersion{D,M,typeof(field)}(bare, field, Float64(phi))
end

function MeanFieldDispersion(bare::M, field::SuperconductingPairing, phi::Float64) where {D,M<:ElectronicDispersion{D}}
    return MeanFieldDispersion{D,M,SuperconductingPairing}(bare, field, phi)
end


# Evaluate dispersion at momentum k as a Hermitian matrix
function ε(
    k::SVector{D,Float64},
    model::FreeElectron{D}
) where {D}
    val = sum(abs2, k) / (2 * model.mass) - model.EF
    return Hermitian(hcat(val)) # 1x1 Hermitian matrix
end

function ε(
    k::SVector{1,Float64},
    model::TightBinding{1}
)
    val = -2 * model.t * cos(k[1]) - model.EF
    return Hermitian(hcat(val))
end

function ε(
    k::SVector{2,Float64},
    model::TightBinding{2}
)
    val = -2 * model.t * (cos(k[1]) + cos(k[2])) - 4 * model.tp * cos(k[1]) * cos(k[2]) - model.EF
    return Hermitian(hcat(val))
end

function ε(
    k::SVector{3,Float64},
    model::TightBinding{3}
)
    val = -2 * model.t * (cos(k[1]) + cos(k[2]) + cos(k[3])) - model.EF
    return Hermitian(hcat(val))
end

function ε(
    k::SVector{D,Float64},
    model::RenormalizedDispersion{D}
) where {D}
    # Simplified placeholder for self energy renormalization
    bare_H = ε(k, model.bare_dispersion)
    Σ_H = Σ(k, model.self_energy)
    return Hermitian(bare_H + Σ_H)
end

function ε(
    k::SVector{D,Float64},
    model::MeanFieldDispersion{D,M,<:ChargeDensityWave{D}}
) where {D,M}

    bare_disp = model.bare_dispersion
    field = model.field
    phi = model.phi

    # Original dispersion at k
    H11 = real(ε(k, bare_disp)[1, 1])

    # Original dispersion at k + Q
    H22 = real(ε(k + field.q, bare_disp)[1, 1])

    # Off-diagonal coupling
    H12 = phi

    # Construct 2x2 Hermitian matrix
    H = @SMatrix [H11 H12;
                  H12 H22]

    return Hermitian(H)
end

# Phonon dispersions
ω(q::SVector{D,Float64}, model::EinsteinModel) where {D} = model.ωE
ω(q::SVector{D,Float64}, model::DebyeModel) where {D} = model.vs * norm(q)
ω(q::SVector{D,Float64}, d::PolaritonModel) where {D} = sqrt(d.ωE^2 + (d.vs * norm(q))^2)

function ω(q::SVector{D,Float64}, d::MonoatomicLatticeModel) where {D}
    val = sum(1 - cos(d.a * qi) for qi in q)
    return sqrt(2 * d.K / d.M * val)
end

"""
    band_structure(disp, k)

Returns an `Eigen` object containing eigenvalues and eigenvectors for the 
Hamiltonian matrix at momentum `k`.
This abstracts away the diagonalization needed for Lindhard bubble calculations 
in generalized many-body systems.
"""
function band_structure(disp::ElectronicDispersion{D}, k::SVector{D,Float64}) where {D}
    H = ε(k, disp)
    return eigen(H)
end



# Define the form factor of the superconducting gap in momentum space
function gap_form_factor(k::SVector{2,Float64}, field::SuperconductingPairing)
    if field.symmetry == :s_wave
        return 1.0 # Isotropic
    elseif field.symmetry == :d_wave
        # Iconic d_{x^2-y^2} symmetry for cuprate high-T_c superconductors
        return cos(k[1]) - cos(k[2])
    else
        error("Unsupported pairing symmetry: $(field.symmetry)")
    end
end

# Handle 1D and 3D s-wave fallback
gap_form_factor(k::SVector, field::SuperconductingPairing) = 1.0

"""
    ε(k::SVector, model::MeanFieldDispersion{D, M, SuperconductingPairing})

Constructs the 2x2 Bogoliubov-de Gennes (BdG) Hamiltonian in Nambu space.
"""
function ε(k::SVector{D,Float64}, model::MeanFieldDispersion{D,M,SuperconductingPairing}) where {D,M}
    # 1. Get the bare normal-state dispersion
    ek = real(ε(k, model.bare_dispersion)[1, 1])

    # 2. Calculate the momentum-dependent superconducting gap Δ_k
    Δ_k = model.phi * gap_form_factor(k, model.field)

    # 3. Construct the BdG matrix (In Nambu space, hole kinetic energy is -ε_{-k}; 
    # assuming spatial inversion symmetry, typically -ε_{-k} = -ε_k)
    H11 = ek
    H22 = -ek
    H12 = Δ_k
    H21 = Δ_k # Assuming real gap for simplicity

    # Return Hermitian matrix for eigenvalue solver to use natively
    return Hermitian(@SMatrix [H11 H12; 
                               H12 H22])
end

# ------------------------------------------------------------------
# 广义正常态基底提升 (Normal State Basis Promotion)
# ------------------------------------------------------------------

"""
    normal_state_basis(model::ElectronicDispersion, field::AuxiliaryField)

根据你要引入的辅助场，返回计算极化率所需的正确“正常态”物理基底。
默认情况下（如 CDW, SDW），正常态就是原始模型本身。
"""
normal_state_basis(model::ElectronicDispersion{D}, field::AuxiliaryField) where {D} = model

# --- 专门为超导准备的 Nambu 空间包装器 ---
struct NormalNambuDispersion{D,M<:ElectronicDispersion{D}} <: ElectronicDispersion{D}
    bare::M
end

"""
对于超导（SuperconductingPairing），即使序参量为 0，正常态也必须
被放置在 2x2 的南部空间（Nambu Space）中，以便泡利矩阵顶点能够作用于它。
"""
normal_state_basis(model::ElectronicDispersion{D}, field::SuperconductingPairing) where {D} = NormalNambuDispersion{D,typeof(model)}(model)

function ε(k::SVector{D,Float64}, model::NormalNambuDispersion) where {D}
    # 提取裸带能量
    ek = real(ε(k, model.bare)[1, 1])
    # 保证时间反演对称性，空穴带的能量为 -ε(-k)
    e_minus_k = real(ε(-k, model.bare)[1, 1])

    # 返回无超导能隙的 2x2 对角 Nambu 矩阵
    return Hermitian(@SMatrix [ek 0.0; 0.0 -e_minus_k])
end