# src/Visualization/responses.jl

function visualize_phase_transition(
    phis::AbstractVector{<:Real},
    Ts::AbstractVector{<:Real},
    field::AuxiliaryField,
    model::ElectronicDispersion,
    interaction::Interaction,
    kgrid::AbstractKGrid,
)

    fig = Figure(size=(1000, 500), fontsize=16)

    # 左图：自由能地貌
    ax1 = Axis(fig[1, 1],
        xlabel=L"Order Parameter $\phi$",
        ylabel=L"Condensation Energy $\mathcal{F}(\phi) - \mathcal{F}(0)$",
        title="Free Energy Landscape vs T")

    # 右图：能隙随温度演化
    ax2 = Axis(fig[1, 3],
        xlabel=L"Temperature $T$",
        ylabel=L"Superconducting Gap $\Delta(T)$",
        title="Order Parameter vs T")

    # ==========================================
    # 4. 执行双重扫描与绘图
    # ==========================================
    colormap_choice = :plasma
    colors = cgrad(colormap_choice, length(Ts))

    phi_gs_list = Float64[] # 用于记录每个温度下的基态序参量

    println("🚀 开始扫描温度相图...")
    for (i, T) in enumerate(Ts)
        # 4.1 扫描自由能曲线
        F_vals = evaluate_action(phis, field, model, interaction, kgrid, ExactTrLn(); T=T)

        # 减去 phi=0 时的能量，得到纯粹的凝聚能
        F_norm = F_vals .- F_vals[1]

        # 画出当前温度的自由能曲线
        lines!(ax1, phis, F_norm, color=colors[i], linewidth=2.5)

        # 4.2 精确求解当前温度的基态能隙 (使用我们在 observables 里的优化器)
        # 初始猜测值设为上一步的结果，可以极大加速优化收敛！
        guess = i == 1 ? 0.2 : phi_gs_list[end]
        phi_gs = solve_ground_state(field, model, interaction, kgrid, ExactTrLn(); phi_guess=guess, T=T)

        # 如果能隙非常小（比如小于 1e-4），物理上认为已经相变为正常态
        push!(phi_gs_list, phi_gs < 1e-4 ? 0.0 : phi_gs)
    end

    # 添加自由能图的温度 Colorbar
    Colorbar(fig[1, 2], limits=extrema(Ts), colormap=colormap_choice, label="Temperature T")

    # 画出右侧的序参量演化图
    scatterlines!(ax2, Ts, phi_gs_list, color=:crimson, markersize=10, linewidth=2.5)

    # 标出零线，辅助视觉
    hlines!(ax1, [0.0], color=:black, linestyle=:dash, linewidth=1)
    hlines!(ax2, [0.0], color=:black, linestyle=:dash, linewidth=1)

    println("✅ 扫描完成！")
    return fig
end


# ============================================================================
# Susceptibility & Spectral Visualization
# ============================================================================

function visualize_landscape(::Val{1}, qgrid::KGrid{1}, landscape_vector::Vector{Float64}; axis=(;), kwargs...)
    qs = [q[1] for q in qgrid.points]
    perm = sortperm(qs)
    qs_sorted = qs[perm]
    vals_sorted = landscape_vector[perm]

    fig = Figure(size=(800, 500))
    ax = Axis(fig[1, 1]; xlabel=L"q", ylabel=L"\chi_0(q, \omega=0)", title="1D Instability Landscape", axis...)

    lines!(ax, qs_sorted, vals_sorted; linewidth=2, color=:crimson, kwargs...)
    hlines!(ax, [0.0], color=:black, alpha=0.3)
    return fig
end

function visualize_landscape(::Val{2}, qgrid::KGrid{2}, landscape_matrix::Matrix{Float64}; axis=(;), kwargs...)
    kxs = unique(sort([k[1] for k in qgrid.points]))
    kys = unique(sort([k[2] for k in qgrid.points]))

    fig = Figure(size=(800, 600))
    ax = Axis(fig[1, 1]; xlabel=L"q_x", ylabel=L"q_y", title="Instability Landscape (Static Susceptibility)", axis...)

    hm = heatmap!(ax, kxs, kys, landscape_matrix; colormap=:magma, kwargs...)
    Colorbar(fig[1, 2], hm, label=L"\chi_0(q, \omega=0)")
    lines!(ax, [-π, π, π, -π, -π], [-π, -π, π, π, -π], color=:white, linestyle=:dash, alpha=0.5)
    return fig
end

function visualize_spectral_function(qpath::KPath{D}, omegas::AbstractVector{Float64}, spectral_matrix::Matrix{Float64}; axis=(;), kwargs...) where {D}
    dist = 0.0
    distances = zeros(length(qpath.points))
    for i in 2:length(qpath.points)
        dist += norm(qpath.points[i] - qpath.points[i-1])
        distances[i] = dist
    end

    tick_positions = distances[qpath.node_indices]
    tick_labels = qpath.node_labels

    fig = Figure(size=(900, 600))
    ax = Axis(fig[1, 1]; ylabel=L"\omega", title="Dynamic Spectral Function " * L"A(q, \omega)",
        xticks=(tick_positions, tick_labels), xgridvisible=false, axis...)

    hm = heatmap!(ax, distances, omegas, spectral_matrix; colormap=:inferno, kwargs...)
    Colorbar(fig[1, 2], hm, label=L"\text{Im}[\chi(q, \omega)]")
    vlines!(ax, tick_positions, color=:white, linestyle=:dash, linewidth=0.8, alpha=0.6)
    xlims!(ax, distances[1], distances[end])
    return fig
end


function visualize_zeeman_pairing_landscape(T_val, h_val, q_vals, model, interaction, kgrid)
    # 动态获取系统的真实物理维度 D (1D, 2D, 或 3D)
    D = length(first(kgrid.points))

    # 记录每个 q 下，优化出来的最佳能隙 phi 和对应的自由能 F
    F_min_list = Float64[]
    phi_opt_list = Float64[]

    println("🚀 开始扫描外加磁场下的配对动量 q_x ... (h = $h_val, T = $T_val)")
    current_guess = 0.4

    for q in q_vals
        # 【核心修复】：根据维度 D 自动生成动量向量，默认在 qx 方向扫描
        # 如果 D=2，自动生成 [q, 0.0]；如果 D=3，自动生成 [q, 0.0, 0.0]
        q_vec = SVector{D,Float64}(ntuple(i -> i == 1 ? Float64(q) : 0.0, D))

        fflo_field = FFLOPairing(q_vec, h_val)

        # 对当前固定的 q，寻找使能量最低的 Δ(phi)
        phi_gs = solve_ground_state(fflo_field, model, interaction, kgrid, ExactTrLn(); phi_guess=current_guess, T=T_val)

        # 如果 phi 已经衰减到 0，说明这个 q 下系统不支持超导
        if phi_gs < 1e-4
            phi_gs = 0.0
        end

        # 计算这个 q 和对应的最佳 phi 下的自由能
        F_val = evaluate_action(phi_gs, fflo_field, model, interaction, kgrid, ExactTrLn(); T=T_val)

        push!(phi_opt_list, phi_gs)
        push!(F_min_list, F_val)

        # 绝热追踪：防止优化器在 0 附近迷路
        current_guess = phi_gs > 0.05 ? phi_gs : 0.05
    end

    # 【核心修复】：同样对正常态的 q=0 向量进行维度适配
    zero_q_vec = zero(SVector{D,Float64})
    F_normal = evaluate_action(0.0, FFLOPairing(zero_q_vec, h_val), model, interaction, kgrid, ExactTrLn(); T=T_val)

    # 减去正常态(非超导)的能量，画凝聚能
    F_norm = F_min_list .- F_normal

    # ================= 绘图 =================
    fig = Figure(size=(1000, 450), fontsize=16)

    # 左图：自由能随 q 的变化
    ax1 = Axis(fig[1, 1],
        xlabel=L"Center-of-Mass Momentum $q_x$",
        ylabel=L"Condensation Energy $\mathcal{F}(q) - \mathcal{F}_{\mathrm{normal}}$",
        title="Magnetic Pairing Landscape (h = $h_val)")

    lines!(ax1, q_vals, F_norm, color=:royalblue, linewidth=3)
    hlines!(ax1, [0.0], color=:gray, linestyle=:dash)

    # 寻找理论上的 FFLO 动量 (能量最低点)
    min_idx = argmin(F_norm)
    q_fflo = q_vals[min_idx]
    scatter!(ax1, [q_fflo], [F_norm[min_idx]], color=:crimson, markersize=12, label="Global Minimum")
    axislegend(ax1, position=:lt)

    # 右图：能隙振幅随 q 的变化
    ax2 = Axis(fig[1, 2],
        xlabel=L"Center-of-Mass Momentum $q_x$",
        ylabel=L"Optimal Gap $\Delta(q)$",
        title="Order Parameter vs Momentum")

    lines!(ax2, q_vals, phi_opt_list, color=:crimson, linewidth=3)
    scatter!(ax2, [q_fflo], [phi_opt_list[min_idx]], color=:royalblue, markersize=12)

    return fig
end


function visualize_collective_modes(
    T_val::Real,
    field::AuxiliaryField,
    model::ElectronicDispersion,
    interaction::Interaction,
    kgrid::AbstractKGrid,
    qpath::KPath;
    omega_max_factor::Real=5.0, # 扫描到 3.5 倍的能隙
    n_omegas::Integer=100,             # 频率空间的网格数
    eta::Real=0.02              # 谱线展宽 (决定洛伦兹峰的锐度)
)
    # ==========================================
    # 1. 求解基态能隙
    # ==========================================
    println("⏳ 正在求解基态能隙 (T = $T_val)...")
    phi_gs = solve_ground_state(field, model, interaction, kgrid, ExactTrLn(); phi_guess=0.4, T=T_val)
    if phi_gs < 1e-4
        phi_gs = 0.0
        println("⚠️ 警告：当前参数下系统未进入超导态 (Δ ≈ 0)。")
    else
        println("✅ 基态能隙求得: Δ₀ = $(round(phi_gs, digits=4))")
    end

    # ==========================================
    # 2. 构造超导 BdG 色散
    # ==========================================
    bdg_dispersion = MeanFieldDispersion(model, field, phi_gs)

    # ==========================================
    # 3. 提取 qpath 信息与频率网格
    # ==========================================
    labels = qpath.node_labels
    distances = zeros(length(qpath.points))
    dist = 0.0
    for i in 2:length(qpath.points)
        dist += norm(qpath.points[i] - qpath.points[i-1])
        distances[i] = dist
    end
    tick_positions = distances[qpath.node_indices]

    # 动态生成频率扫描范围：最高扫到 omega_max_factor * Δ₀
    # 如果 Δ₀ 为 0 (正常态)，则保底扫到 2.0
    w_max = phi_gs > 0 ? omega_max_factor * phi_gs : 2.0
    omegas = range(0.0, w_max, length=n_omegas)

    # ==========================================
    # 4. 执行动态扫描
    # ==========================================
    println("🚀 开始扫描集体激发谱 (q, ω)... 这可能需要一点时间")

    chi_im_matrix = scan_rpa_spectral_function_hpc(bdg_dispersion, kgrid, qpath.points, omegas; T=T_val, η=eta)

    # ==========================================
    # 5. 画图
    # ==========================================
    fig = Figure(size=(800, 500), fontsize=16)
    ax = Axis(fig[1, 1],
        title=L"Superconducting Excitation Spectrum $\mathrm{Im}\chi(q, \omega)$",
        xlabel="Momentum Transfer q",
        ylabel=L"Frequency $\omega$",
        xticks=(tick_positions, labels)
    )

    # 画出热力图
    hm = heatmap!(ax, distances, omegas, chi_im_matrix, colormap=:magma)
    Colorbar(fig[1, 2], hm, label=L"Spectral Weight $\mathrm{Im}\chi$")

    # 如果系统在超导态，标出 2Δ₀ 的理论分界线
    if phi_gs > 0
        hlines!(ax, [2 * phi_gs], color=:cyan, linestyle=:dash, linewidth=2, label=L"Pair-breaking edge $2\Delta_0$")
        axislegend(ax, position=:lt)
    end

    return fig
end
