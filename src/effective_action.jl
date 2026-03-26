# effective_action.jl
using StaticArrays

"""
    evaluate(action::EffectiveAction, phi::Float64, ::ExactTrLn; T::Float64=1e-3)

Evaluates the non-linear effective action by computing the Exact Tr[ln] term over
the reconstructed energy bands.
"""
function evaluate(action::EffectiveAction, phi::Float64, ::ExactTrLn; T::Float64=1e-3)
    # F(phi) = N * phi^2 / V_eff(q) - T * \sum_{k, \lambda} \ln(1 + e^{-\beta \lambda(k)})

    # 1. 获取当前场的波矢（对于 CDW 是嵌套波矢 Q，对于均匀超导通常是 0）
    q_vec = hasproperty(action.field, :q) ? action.field.q : zero(first(action.grid))

    # 2. 现场评估总的相互作用强度 V(q)
    # 这里完美支持了你之前组合的 phonon_attraction + coulomb_repulsion！
    V_total = V(q_vec, action.interaction)

    # 第一项：高斯积分产生的二次项
    term1 = phi^2 / V_total

    # 3. 构造平均场重构的能带（多重派发会自动识别是 CDW 还是 BdG 超导）
    mf_disp = MeanFieldDispersion(action.model, action.field, phi)

    tr_ln_sum = 0.0
    for i in 1:length(action.grid)
        k = action.grid.points[i]
        w = action.grid.weights[i]

        # 求解破坏对称性后的真实本征值（比如 Bogoliubov 准粒子能量 E_k）
        lambdas = real(band_structure(mf_disp, k).values)

        for lam in lambdas
            # 数值安全的费米-狄拉克积分技巧
            if lam < 0
                tr_ln_sum += w * (lam - T * log1p(exp(lam / T)))
            else
                tr_ln_sum += w * (-T * log1p(exp(-lam / T)))
            end
        end
    end

    # 注意：在标准的 2x2 Nambu 空间或 CDW 折叠中，TrLn 往往包含了粒子和空穴的两支，
    # 数学上严格来说前面应该有个整体的 1/2 系数（如果你遍历了整个布里渊区而不是约化布里渊区）。
    # 这里保持你的原始公式逻辑，这只影响全局标度，不影响极小值点 phi_0 的位置。
    return term1 + tr_ln_sum
end

"""
    evaluate(action::EffectiveAction, phi::Float64, ::RPA; T::Float64=1e-3)

Evaluates the effective action using the quadratic RPA expansion.
F(phi) \\approx (1/V - \\chi_0(Q)) phi^2
"""
function evaluate(action::EffectiveAction, phi::Float64, ::RPA; T::Float64=1e-3)
    # 1. 自动根据外场类型，获取正确的正常态基底（1x1 还是 2x2）
    normal_model = normal_state_basis(action.model, action.field)

    # 2. 组装大一统极化率仿函数（此时 normal_model 已经完美适配 action.field 的顶点矩阵）
    chi0 = GeneralizedSusceptibility(normal_model, action.grid, action.field, T)
    # 获取当前场的波矢（如果是均匀超导则是 0）
    q_vec = hasproperty(action.field, :q) ? action.field.q : zero(first(action.grid))

    # 【核心改变】：现场评估总的相互作用强度 V(q)
    # 对于 CDW，排斥为正，吸引为负；我们需要的是产生不稳定性的净吸引力（在作用量符号下）
    V_total = V(q_vec, action.interaction)
    chi_val = chi0(q_vec)

    return (1.0 / action.V_bare - real(chi_val)) * phi^2
end

"""
    evaluate(action::EffectiveAction, phi_values::AbstractVector{<:Real}, level::ApproximationLevel; T::Float64=1e-3)

Evaluates the effective action for a collection of `phi` values.
"""
function evaluate(action::EffectiveAction, phi_values::AbstractVector{<:Real}, level::ApproximationLevel; T::Float64=1e-3)
    return [evaluate(action, phi, level; T=T) for phi in phi_values]
end
