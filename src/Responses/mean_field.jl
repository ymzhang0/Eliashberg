# ----------------------------------------------------------------------------
# Mean-Field Dispersion Structs & Constructors
# ----------------------------------------------------------------------------

struct MeanFieldDispersion{D,M<:ElectronicDispersion{D},F<:AuxiliaryField} <: ElectronicDispersion{D}
    bare_dispersion::M
    field::F
    phi::Float64
end

struct NormalNambuDispersion{D,M<:ElectronicDispersion{D}} <: ElectronicDispersion{D}
    bare::M
end

struct ParticleHoleNormalDispersion{D,M<:ElectronicDispersion{D},F<:ParticleHoleChannel{D}} <: ElectronicDispersion{D}
    bare::M
    field::F
end

struct FFLONormalDispersion{D,M<:ElectronicDispersion{D}} <: ElectronicDispersion{D}
    bare::M
    q::SVector{D,Float64}
end

struct PDWNormalDispersion{D,M<:ElectronicDispersion{D}} <: ElectronicDispersion{D}
    bare::M
    q::SVector{D,Float64}
end

# Generic constructor to ensure phi is Float64
function MeanFieldDispersion(bare::M, field::F, phi::Real) where {D,M<:ElectronicDispersion{D},F<:AuxiliaryField}
    return MeanFieldDispersion{D,M,F}(bare, field, Float64(phi))
end


function MeanFieldDispersion(bare::M, field::ParticleHoleChannel{D}, phi::Float64) where {D,M<:ElectronicDispersion{D}}
    return MeanFieldDispersion{D,M,typeof(field)}(bare, field, phi)
end

function MeanFieldDispersion(bare::M, field::BCSReducedPairing, phi::Float64) where {D,M<:ElectronicDispersion{D}}
    return MeanFieldDispersion{D,M,BCSReducedPairing}(bare, field, phi)
end

function MeanFieldDispersion(bare::M, field::FFLOPairing{D}, phi::Float64) where {D,M<:ElectronicDispersion{D}}
    return MeanFieldDispersion{D,M,FFLOPairing{D}}(bare, field, phi)
end

function MeanFieldDispersion(bare::M, field::PairDensityWave{D}, phi::Float64) where {D,M<:ElectronicDispersion{D}}
    return MeanFieldDispersion{D,M,PairDensityWave{D}}(bare, field, phi)
end

# ----------------------------------------------------------------------------
# Basis Promotion Logic
# ----------------------------------------------------------------------------

normal_state_basis(model::ElectronicDispersion{D}, field::AuxiliaryField) where {D} = model
normal_state_basis(model::ElectronicDispersion{D}, field::ChargeDensityWave{D}) where {D} =
    ParticleHoleNormalDispersion{D,typeof(model),typeof(field)}(model, field)
normal_state_basis(model::ElectronicDispersion{D}, field::BCSReducedPairing) where {D} = NormalNambuDispersion{D,typeof(model)}(model)
normal_state_basis(model::ElectronicDispersion{D}, field::FFLOPairing{D}) where {D} = FFLONormalDispersion{D,typeof(model)}(model, field.q)
normal_state_basis(model::ElectronicDispersion{D}, field::PairDensityWave{D}) where {D} = PDWNormalDispersion{D,typeof(model)}(model, field.q)

function ε(k::SVector{D,Float64}, model::ParticleHoleNormalDispersion{D}) where {D}
    H11 = real(ε(k, model.bare)[1, 1])
    H22 = real(ε(k + model.field.q, model.bare)[1, 1])
    return Hermitian(@SMatrix [H11 0.0; 0.0 H22])
end

function ε(k::SVector{D,Float64}, model::NormalNambuDispersion{D}) where {D}
    ek = real(ε(k, model.bare)[1, 1])
    e_minus_k = real(ε(-k, model.bare)[1, 1])
    return Hermitian(@SMatrix [ek 0.0; 0.0 -e_minus_k])
end

function ε(::SVector{D,Float64}, ::FFLONormalDispersion{D}) where {D}
    @warn "FFLO normal-state basis is not fully implemented yet."
    throw(ErrorException("FFLO normal-state basis is not fully implemented yet."))
end

function ε(::SVector{D,Float64}, ::PDWNormalDispersion{D}) where {D}
    @warn "PDW normal-state basis is not fully implemented yet."
    throw(ErrorException("PDW normal-state basis is not fully implemented yet."))
end

function ε(k::SVector{D,Float64}, model::MeanFieldDispersion{D,M,F}) where {D,M,F<:AuxiliaryField}
    # 1. 提取正常态的本底哈密顿量 H_0 (通过 normal_state_basis 降级回去调用)
    # 这会返回 2x2 或 3x3 的只有对角线的矩阵 (对于超导，还自带了 -e_{-k} 和塞曼场)
    normal_basis_model = normal_state_basis(model.bare_dispersion, model.field)
    H_0 = ε(k, normal_basis_model)
    
    # 2. 提取顶点矩阵 Γ
    Γ = vertex_matrix(k, model.field)
    
    # 3. 终极组装：Schwinger-Dyson Equation
    return H_0 + model.phi * Γ * I
end

