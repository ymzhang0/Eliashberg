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
normal_state_basis(model::SpinorDispersion{D}, field::SpinDensityWave{D}) where {D} =
    ParticleHoleNormalDispersion{D,typeof(model),typeof(field)}(model, field)
normal_state_basis(model::ElectronicDispersion{D}, field::SpinDensityWave{D}) where {D} =
    normal_state_basis(SpinorDispersion(model), field)
normal_state_basis(model::ElectronicDispersion{D}, field::BCSReducedPairing) where {D} = NormalNambuDispersion{D,typeof(model)}(model)
normal_state_basis(model::ElectronicDispersion{D}, field::FFLOPairing{D}) where {D} = FFLONormalDispersion{D,typeof(model)}(model, field.q)
normal_state_basis(model::ElectronicDispersion{D}, field::PairDensityWave{D}) where {D} = PDWNormalDispersion{D,typeof(model)}(model, field.q)

function ε(k::SVector{D,Float64}, model::ParticleHoleNormalDispersion{D}) where {D}
    Hk = _matrix_data(ε(k, model.bare))
    Hkq = _matrix_data(ε(k + model.field.q, model.bare))
    return _block_diagonal(Hk, Hkq)
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

function vertex_matrix(model::ParticleHoleNormalDispersion{D}, k::SVector{D,Float64}, ::ChargeDensityWave{D}) where {D}
    return _particle_hole_order_vertex(vertex_matrix(model.bare, k, DirectChannel()))
end

function vertex_matrix(model::ParticleHoleNormalDispersion{D}, k::SVector{D,Float64}, field::SpinDensityWave{D,Dir}) where {D,Dir}
    return _particle_hole_order_vertex(vertex_matrix(model.bare, k, ExchangeChannel(Dir)))
end

_spin_vertex_style(::NormalNambuDispersion{D,M}) where {D,M<:SpinorDispersion{D}} = SpinorVertexStyle()
_spin_operator_for_model(model::NormalNambuDispersion{D,M}, k::SVector{D,Float64}, pauli::StaticMatrix{2,2}) where {D,M<:SpinorDispersion{D}} =
    _spin_operator_for_model(model.bare, k, pauli)

function _particle_hole_order_vertex(coupling::StaticMatrix{N,N,T}) where {N,T}
    return Hermitian(SMatrix{2N,2N,T,4N*N}(ntuple(idx -> begin
        row = (idx - 1) % (2N) + 1
        col = (idx - 1) ÷ (2N) + 1
        if row <= N && col > N
            return coupling[row, col - N]
        elseif row > N && col <= N
            return coupling[row - N, col]
        end
        return zero(T)
    end, 4N*N)))
end

function _particle_hole_order_vertex(coupling::Number)
    T = typeof(coupling)
    return Hermitian(@SMatrix [zero(T) coupling; coupling zero(T)])
end

function _particle_hole_order_vertex(coupling::AbstractMatrix{T}) where {T}
    size(coupling, 1) == size(coupling, 2) || throw(DimensionMismatch("Particle-hole vertex requires a square coupling matrix."))

    n = size(coupling, 1)
    if n == 1
        return _particle_hole_order_vertex(coupling[1, 1])
    end

    matrix = zeros(T, 2 * n, 2 * n)
    matrix[1:n, n+1:end] = coupling
    matrix[n+1:end, 1:n] = coupling
    return Hermitian(matrix)
end

function ε(k::SVector{D,Float64}, model::MeanFieldDispersion{D,M,F}) where {D,M,F<:AuxiliaryField}
    # 1. 提取正常态的本底哈密顿量 H_0 (通过 normal_state_basis 降级回去调用)
    # 这会返回 2x2 或 3x3 的只有对角线的矩阵 (对于超导，还自带了 -e_{-k} 和塞曼场)
    normal_basis_model = normal_state_basis(model.bare_dispersion, model.field)
    H_0 = ε(k, normal_basis_model)
    
    # 2. 提取顶点矩阵 Γ
    Γ = vertex_matrix(normal_basis_model, k, model.field)
    
    # 3. 终极组装：Schwinger-Dyson Equation
    return H_0 + model.phi * Γ * I
end
