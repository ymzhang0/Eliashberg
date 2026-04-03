# src/Responses/vertices.jl
# ============================================================================
# 相互作用顶点矩阵 (Vertex Matrices for Mean-Field Condensates)
# 作用：定义序参量 phi 是如何耦合到正常态基矢 (Normal State Basis) 上的。
# 泛型公式：H_MF = H_0 + phi * Γ
# ============================================================================

"""
    vertex_matrix(k, field)

1. Particle-Hole Channel Condensates (CDW, SDW 等)
基矢: (c_k, c_{k+q})^T  (2x2 BZ 折叠基矢)
物理：序参量 phi 直接将 k 和 k+q 态耦合，无额外动量依赖。
"""
function vertex_matrix(k::SVector{D,Float64}, field::ParticleHoleChannel{D}) where {D}
    # 纯粹的非对角耦合
    return Hermitian(@SMatrix [0.0 1.0; 
                               1.0 0.0])
end

# 粒子-空穴响应探针在 susceptibility 泡泡图里仍然退化为标量权重
vertex_matrix(::ParticleHoleChannel) = 1.0

# 电荷密度探针 (c^† c)，在单带正常态下顶点为 1.0
vertex_matrix(::DirectChannel) = 1.0

# 自旋密度探针 (S_z)。在单带且无自旋显式矩阵的模型下，泡泡图的标量权重也为 1.0
vertex_matrix(::ExchangeChannel) = 1.0

# 兜底方法：以防未来 susceptibilities.jl 被修改为强制传入 k
vertex_matrix(k::SVector, ::DirectChannel) = 1.0
vertex_matrix(k::SVector, ::ExchangeChannel) = 1.0

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
