# src/Responses/vertices.jl
# ============================================================================
# 相互作用顶点矩阵 (Vertex Matrices for Mean-Field Condensates)
# 作用：定义序参量 phi 是如何耦合到正常态基矢 (Normal State Basis) 上的。
# 泛型公式：H_MF = H_0 + phi * Γ
# ============================================================================

const _σ₀_static = @SMatrix [σ₀[1, 1] σ₀[1, 2]; σ₀[2, 1] σ₀[2, 2]]
const _σ₁_static = @SMatrix [σ₁[1, 1] σ₁[1, 2]; σ₁[2, 1] σ₁[2, 2]]
const _σ₂_static = @SMatrix [σ₂[1, 1] σ₂[1, 2]; σ₂[2, 1] σ₂[2, 2]]
const _σ₃_static = @SMatrix [σ₃[1, 1] σ₃[1, 2]; σ₃[2, 1] σ₃[2, 2]]

abstract type SpinVertexStyle end
struct SpinlessVertexStyle <: SpinVertexStyle end
struct SpinfulVertexStyle <: SpinVertexStyle end

_spin_vertex_style(::ElectronicDispersion) = SpinlessVertexStyle()
_spin_vertex_style(::SpinfulDispersion) = SpinfulVertexStyle()

spin_direction(::ExchangeChannel{Dir}) where {Dir} = Dir
spin_direction(::SpinDensityWave{D,Dir}) where {D,Dir} = Dir

vertex_matrix(::DirectChannel) = 1.0
vertex_matrix(::ChargeDensityWave) = 1.0
vertex_matrix(::ExchangeChannel{:z}) = 1.0
vertex_matrix(::SpinDensityWave{D,:z}) where {D} = 1.0
vertex_matrix(field::ExchangeChannel) = _spinless_spin_flip_error(spin_direction(field))
vertex_matrix(field::SpinDensityWave) = _spinless_spin_flip_error(spin_direction(field))

vertex_matrix(k::SVector, field::DirectChannel) = vertex_matrix(field)
vertex_matrix(k::SVector, field::ChargeDensityWave) = vertex_matrix(field)
vertex_matrix(k::SVector, field::ExchangeChannel) = vertex_matrix(field)
vertex_matrix(k::SVector, field::SpinDensityWave) = vertex_matrix(field)

function vertex_matrix(model::ElectronicDispersion{D}, field::ParticleHoleChannel) where {D}
    return vertex_matrix(model, zero(SVector{D,Float64}), field)
end

function vertex_matrix(model::ElectronicDispersion, k::SVector, ::DirectChannel)
    return _density_vertex(_spin_vertex_style(model), model, k)
end

function vertex_matrix(model::ElectronicDispersion, k::SVector, ::ChargeDensityWave)
    return vertex_matrix(model, k, DirectChannel())
end

function vertex_matrix(model::ElectronicDispersion, k::SVector, field::ExchangeChannel{Dir}) where {Dir}
    return _exchange_vertex(_spin_vertex_style(model), model, k, Val(Dir))
end

function vertex_matrix(model::ElectronicDispersion, k::SVector, field::SpinDensityWave{D,Dir}) where {D,Dir}
    return _exchange_vertex(_spin_vertex_style(model), model, k, Val(Dir))
end

_density_vertex(::SpinlessVertexStyle, ::ElectronicDispersion, ::SVector) = 1.0
_density_vertex(::SpinfulVertexStyle, model::ElectronicDispersion, k::SVector) =
    _spin_operator_for_model(model, k, _σ₀_static)

_exchange_vertex(::SpinlessVertexStyle, ::ElectronicDispersion, ::SVector, ::Val{:z}) = 1.0
_exchange_vertex(::SpinlessVertexStyle, ::ElectronicDispersion, ::SVector, ::Val{:x}) = _spinless_spin_flip_error(:x)
_exchange_vertex(::SpinlessVertexStyle, ::ElectronicDispersion, ::SVector, ::Val{:y}) = _spinless_spin_flip_error(:y)
_exchange_vertex(::SpinlessVertexStyle, ::ElectronicDispersion, ::SVector, ::Val{:transverse}) = _spinless_spin_flip_error(:transverse)
_exchange_vertex(::SpinfulVertexStyle, model::ElectronicDispersion, k::SVector, ::Val{:z}) =
    _spin_operator_for_model(model, k, _σ₃_static)
_exchange_vertex(::SpinfulVertexStyle, model::ElectronicDispersion, k::SVector, ::Val{:x}) =
    _spin_operator_for_model(model, k, _σ₁_static)
_exchange_vertex(::SpinfulVertexStyle, model::ElectronicDispersion, k::SVector, ::Val{:y}) =
    _spin_operator_for_model(model, k, _σ₂_static)
_exchange_vertex(::SpinfulVertexStyle, model::ElectronicDispersion, k::SVector, ::Val{:transverse}) =
    _spin_operator_for_model(model, k, _σ₁_static)

function _spin_operator_for_model(model::SpinfulDispersion{D}, k::SVector{D,Float64}, pauli::StaticMatrix{2,2}) where {D}
    bare_block = _matrix_data(ε(k, model.bare))
    return _spin_operator(bare_block, pauli)
end

function _spin_operator(block::StaticMatrix{N,N,TB}, pauli::StaticMatrix{2,2,TP}) where {N,TB,TP}
    T = promote_type(TB, TP)
    return Hermitian(SMatrix{2N,2N,T,4N*N}(ntuple(idx -> begin
        row = (idx - 1) % (2N) + 1
        col = (idx - 1) ÷ (2N) + 1
        spin_row = row <= N ? 1 : 2
        spin_col = col <= N ? 1 : 2
        orbital_row = row <= N ? row : row - N
        orbital_col = col <= N ? col : col - N
        orbital_row == orbital_col || return zero(T)
        return T(pauli[spin_row, spin_col])
    end, 4N*N)))
end

function _spin_operator(block::AbstractMatrix{TB}, pauli::StaticMatrix{2,2,TP}) where {TB,TP}
    n_orbitals = size(block, 1)
    size(block, 1) == size(block, 2) || throw(DimensionMismatch("Spin operator requires a square orbital block."))

    T = promote_type(TB, TP)
    if n_orbitals == 1
        return Hermitian(@SMatrix [T(pauli[1, 1]) T(pauli[1, 2]); T(pauli[2, 1]) T(pauli[2, 2])])
    end

    matrix = zeros(T, 2 * n_orbitals, 2 * n_orbitals)
    for spin_row in 1:2, spin_col in 1:2
        coeff = T(pauli[spin_row, spin_col])
        iszero(coeff) && continue
        row_offset = (spin_row - 1) * n_orbitals
        col_offset = (spin_col - 1) * n_orbitals
        @inbounds for orbital in 1:n_orbitals
            matrix[row_offset + orbital, col_offset + orbital] = coeff
        end
    end
    return Hermitian(matrix)
end

function _spinless_spin_flip_error(direction::Symbol)
    throw(ArgumentError("Spin direction $direction requires a SpinfulDispersion basis."))
end

"""
2. Standard BCS Reduced Pairing
基矢: (c_k, c_{-k}^†)^T (2x2 Nambu 旋量)
物理：超导能隙，携带有特定的配对对称性形状因子 f(k)。
"""
function vertex_matrix(k::SVector{D,Float64}, field::BCSReducedPairing) where {D}
    fk = gap_form_factor(k, field)
    # Hermitian 默认取上三角进行共轭对称，完美兼容 p 波等复数形状因子
    return Hermitian(@SMatrix [0.0 fk; 
                               fk  0.0])
end

"""
3. FFLO Pairing
基矢: (c_{k↑}, c_{-k+q↓}^†)^T (2x2 平移 Nambu 旋量)
物理：由于库珀对整体带有动量 q，两个费米子的相对动量变为 k - q/2。
"""
function vertex_matrix(k::SVector{D,Float64}, field::FFLOPairing{D}) where {D}
    # 相对动量
    rel_k = k - field.q / 2.0
    fk = gap_form_factor(rel_k, field)
    
    return Hermitian(@SMatrix [0.0 fk; 
                               fk  0.0])
end

"""
4. Pair Density Wave (PDW)
基矢: (c_k, c_{-k+q}^†, c_{-k-q}^†)^T (3x3 扩展 Nambu 旋量)
物理：驻波超导序，同时向 +q 和 -q 两个方向散射空穴。
"""
function vertex_matrix(k::SVector{D,Float64}, field::PairDensityWave{D}) where {D}
    rel_k_plus  = k - field.q / 2.0
    rel_k_minus = k + field.q / 2.0
    
    # PDW 的振幅在正负 q 处平分，所以附带 1/2 系数
    fk_plus  = 0.5 * gap_form_factor(rel_k_plus, field)
    fk_minus = 0.5 * gap_form_factor(rel_k_minus, field)
    
    # 构造 3x3 顶点矩阵，第一行/列是电子，后两行/列是空穴
    return Hermitian(@SMatrix [
        0.0      fk_plus  fk_minus;
        fk_plus  0.0      0.0;
        fk_minus 0.0      0.0
    ])
end

vertex_matrix(::ElectronicDispersion, k::SVector{D,Float64}, field::BCSReducedPairing) where {D} = vertex_matrix(k, field)
vertex_matrix(::ElectronicDispersion, k::SVector{D,Float64}, field::FFLOPairing{D}) where {D} = vertex_matrix(k, field)
vertex_matrix(::ElectronicDispersion, k::SVector{D,Float64}, field::PairDensityWave{D}) where {D} = vertex_matrix(k, field)

# ----------------------------------------------------------------------------
# Gap Form Factors
# ----------------------------------------------------------------------------

function gap_form_factor(k::SVector{2,Float64}, field::AuxiliaryField)
    if field.symmetry == :s_wave
        return 1.0
    elseif field.symmetry == :d_wave
        return cos(k[1]) - cos(k[2])
    elseif field.symmetry == :p_wave
        return sin(k[1]) + im * sin(k[2])
    elseif field.symmetry == :s_plus_minus_wave
        return cos(k[1]) * cos(k[2])
    else
        error("Unsupported pairing symmetry: $(field.symmetry)")
    end
end

gap_form_factor(k::SVector, field::AuxiliaryField) = 1.0
