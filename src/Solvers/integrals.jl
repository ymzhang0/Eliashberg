
"""
计算积分 I = ∫ exp(-ax^2 + bx^4) dx

方法：
1. 直接数值积分 (Direct): 作为验证基准
2. HS 变换 (Hubbard-Stratonovich): 先解析积掉 x，再数值积 y
"""
function compare_integrals(a::Number, b::Number)
    println("--- 计算参数: a=$a, b=$b ---")

    # --- 安全性检查 ---
    if b > 0
        println("⚠️ 警告: b > 0 时，原积分 exp(bx^4) 在无穷远处发散。")
        println("计算结果将没有物理意义或为 Inf。")
        # 我们可以继续尝试，但结果很可能是发散的
    end

    # ==========================================
    # 方法 1: 直接对 x 进行数值积分 (作为标准答案)
    # ==========================================
    f_direct(x) = exp(-a * x^2 + b * x^4)

    # 积分范围 (-Inf, Inf)
    # 只有当 b <= 0 时这个积分才收敛
    val_direct, err_direct = try
        quadgk(f_direct, -Inf, Inf)
    catch e
        (NaN, NaN)
    end

    # ==========================================
    # 方法 2: 使用 Hubbard-Stratonovich 变换
    # I = (1/sqrt(2π)) * ∫ dy * exp(-y^2/2) * sqrt(π / (a - y*sqrt(2b)))
    # ==========================================

    # 注意：如果 b < 0，sqrt(2b) 是虚数，我们需要用 Complex 类型
    # 系数 c = sqrt(2b)
    c = sqrt(Complex(2 * b))

    function integrand_hs(y)
        # 1. 计算 exp(-y^2/2)
        weight = exp(-0.5 * y^2)

        # 2. 计算 x 积分带来的项: sqrt(π / A_eff)
        # A_eff 是 x^2 的有效系数: a - y*c
        A_eff = a - y * c

        # 如果 A_eff 的实部 <= 0，说明 x 的高斯积分发散
        if real(A_eff) <= 0
            # 这种情况通常只在 b > 0 时发生
            return Inf
        end

        inner_x = sqrt(π / A_eff)

        return weight * inner_x
    end

    # 对 y 从 -Inf 到 Inf 积分
    val_hs_complex, err_hs = quadgk(integrand_hs, -Inf, Inf)

    # 最终乘上前面的系数 1/sqrt(2π)
    result_hs = (1 / sqrt(2π)) * val_hs_complex

    # ==========================================
    # 输出结果
    # ==========================================
    println("1. 直接积分结果 (Direct): $val_direct")
    println("2. HS 变换结果 (HS)    : $(real(result_hs))") # 取实部，虚部理论上为0

    # 验证虚部是否足够小（应该接近机器精度）
    if abs(imag(result_hs)) > 1e-10
        println("   (注意: HS结果含有不可忽略的虚部: $(imag(result_hs))im，请检查参数)")
    end

    return real(result_hs)
end

