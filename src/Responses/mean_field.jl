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

function MeanFieldDispersion(
    bare::ElectronicDispersion,
    comp::CompositeField,
    phis::AbstractVector{<:Real}
)
    length(comp) == length(phis) || throw(DimensionMismatch("Number of fields must match number of phis."))
    return _nest_mean_field_dispersion(bare, comp.fields, phis, 1)
end

_nest_mean_field_dispersion(model::ElectronicDispersion, ::Tuple{}, ::AbstractVector{<:Real}, ::Int) = model

function _nest_mean_field_dispersion(
    model::ElectronicDispersion,
    fields::Tuple{F,Vararg{AuxiliaryField}},
    phis::AbstractVector{<:Real},
    idx::Int
) where {F<:AuxiliaryField}
    next_model = MeanFieldDispersion(model, first(fields), Float64(phis[idx]))
    return _nest_mean_field_dispersion(next_model, Base.tail(fields), phis, idx + 1)
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

# this is not a general construction for spinor superconducting states.
function ε(k::SVector{D,Float64}, model::NormalNambuDispersion{D}) where {D}
    # 动态获取底层模型 (可能是 1x1, 2x2 或更多) 的哈密顿量矩阵
    Hk_mat = _matrix_data(ε(k, model.bare))
    H_minus_k_mat = _matrix_data(ε(-k, model.bare))
    
    # 根据 Bogoliubov-de Gennes (BdG) 理论
    # Nambu 空间的空穴支哈密顿量必须是 -H(-k)^T
    return _block_diagonal(Hk_mat, -transpose(H_minus_k_mat))
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

# 获取超导的顶点矩阵
function vertex_matrix(model::NormalNambuDispersion{D}, k::SVector{D,Float64}, field::BCSReducedPairing) where {D}
    fk = gap_form_factor(k, field)

    bare_block = _matrix_data(ε(k, model.bare))
    return _bcs_nambu_vertex(fk, bare_block)
end

function _bcs_nambu_vertex(fk::Number, ::StaticMatrix{N,N,TB}) where {N,TB}
    T = promote_type(typeof(fk), TB)
    return Hermitian(SMatrix{2N,2N,T,4N*N}(ntuple(idx -> begin
        row = (idx - 1) % (2N) + 1
        col = (idx - 1) ÷ (2N) + 1

        if row <= N && col == N + row
            return T(fk)
        elseif row > N && col == row - N
            return conj(T(fk))
        end

        return zero(T)
    end, 4N*N)))
end

function _bcs_nambu_vertex(fk::Number, block::AbstractMatrix{TB}) where {TB}
    size(block, 1) == size(block, 2) || throw(DimensionMismatch("BCS vertex requires a square normal-state block."))

    n = size(block, 1)
    T = promote_type(typeof(fk), TB)

    if n == 1
        return Hermitian(@SMatrix [zero(T) T(fk); conj(T(fk)) zero(T)])
    end

    matrix = zeros(T, 2n, 2n)
    @inbounds for i in 1:n
        matrix[i, n + i] = T(fk)
        matrix[n + i, i] = conj(T(fk))
    end
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
