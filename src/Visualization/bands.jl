# src/Visualization/bands.jl

# ============================================================================
# Generic Plotting Pipeline for Dispersions
# ============================================================================

"""
    visualize_dispersion(disp::Dispersion, [kgrid::AbstractKGrid]; kwargs...)

Generates the appropriate Makie plot for the given dispersion model.
Automatically infers the dimension via the `dimensionality(disp)` trait.
"""
function visualize_dispersion(disp::Dispersion, kgrid::Union{AbstractKGrid,Nothing}=nothing; kwargs...)
    D = dimensionality(disp)
    grid = isnothing(kgrid) ? default_kgrid(Val(D)) : kgrid
    return visualize_dispersion(Val(D), disp, grid; kwargs...)
end

# ----------------- 1D Dispatch -----------------
function visualize_dispersion(::Val{1}, disp::Dispersion, kgrid::KGrid{1}; E_Fermi=0.0, axis=(;), kwargs...)
    k_vals = [k[1] for k in kgrid.points]
    perm = sortperm(k_vals)
    k_sorted = k_vals[perm]
    bands = [real(band_structure(disp, kgrid.points[i]).values) for i in perm]
    num_bands = length(bands[1])

    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel="k", ylabel="E(k)", title="1D Dispersion", axis...)

    for b in 1:num_bands
        E_b = [E[b] for E in bands]
        lines!(ax, k_sorted, E_b; label="Band $b", kwargs...)
    end
    hlines!(ax, [E_Fermi], color=:black, linestyle=:dash, label="Fermi Level")
    return fig
end

# ----------------- 2D Dispatch -----------------
function visualize_dispersion(::Val{2}, disp::Dispersion, kgrid::KGrid{2}; E_Fermi=0.0, axis=(;), kwargs...)
    kxs = unique(sort([k[1] for k in kgrid.points]))
    kys = unique(sort([k[2] for k in kgrid.points]))
    E_matrix = zeros(Float64, length(kxs), length(kys))

    for (i, kx) in enumerate(kxs)
        for (j, ky) in enumerate(kys)
            vals = real(band_structure(disp, SVector(kx, ky)).values)
            E_matrix[i, j] = vals[1]
        end
    end

    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel=L"k_x", ylabel=L"k_y", title="2D Dispersion", axis...)
    hm = heatmap!(ax, kxs, kys, E_matrix; colormap=:viridis, kwargs...)
    Colorbar(fig[1, 2], hm, label="Energy")
    contour!(ax, kxs, kys, E_matrix; levels=[E_Fermi], color=:red, linewidth=2, labels=true)
    lines!(ax, [-π, π, π, -π, -π], [-π, -π, π, π, -π], color=:white, linestyle=:dash)
    return fig
end

# ----------------- Band Structure along KPath (Any Dimension) -----------------
function visualize_dispersion(::Val{D}, disp::ElectronicDispersion, kpath::KPath; E_Fermi=0.0, axis=(;), kwargs...) where {D}
    dist = 0.0
    distances = zeros(length(kpath.points))
    for i in 2:length(kpath.points)
        dist += norm(kpath.points[i] - kpath.points[i-1])
        distances[i] = dist
    end
    bands = [real(band_structure(disp, k).values) for k in kpath.points]
    num_bands = length(bands[1])
    tick_positions = distances[kpath.node_indices]
    tick_labels = kpath.node_labels

    fig = Figure(size=(800, 600))
    ax = Axis(fig[1, 1]; ylabel="Energy E(k)", title="$(D)D Band Structure",
        xticks=(tick_positions, tick_labels), xgridvisible=false, axis...)

    vlines!(ax, tick_positions, color=:gray, linestyle=:dot, linewidth=1.5)

    for b in 1:num_bands
        E_b = [E[b] for E in bands]
        lines!(ax, distances, E_b; linewidth=2.5, kwargs...)
    end

    hlines!(ax, [E_Fermi], color=:black, linestyle=:dash, linewidth=1.5)
    xlims!(ax, distances[1], distances[end])
    return fig
end

# ----------------- 3D Isosurface (Fermi Surface) -----------------
function visualize_dispersion(::Val{3}, disp::ElectronicDispersion, kgrid::KGrid{3}; E_Fermi=0.0, axis=(;), kwargs...)
    n_pts = 100
    kxs = range(-π, π, length=n_pts)
    kys = range(-π, π, length=n_pts)
    kzs = range(-π, π, length=n_pts)
    E_volume = zeros(Float32, length(kxs), length(kys), length(kzs))

    for (i, kx) in enumerate(kxs)
        for (j, ky) in enumerate(kys)
            for (k, kz) in enumerate(kzs)
                vals = real(band_structure(disp, SVector{3,Float64}(kx, ky, kz)).values)
                E_volume[i, j, k] = vals[1]
            end
        end
    end

    fig = Figure(size=(900, 800), fontsize=16)
    ax = Axis3(fig[1, 1]; xlabel=L"k_x", ylabel=L"k_y", zlabel=L"k_z",
        title="Interactive 3D Fermi Surface", elevation=π / 6, azimuth=π / 4, axis...)

    E_min, E_max = minimum(E_volume), maximum(E_volume)
    sg = SliderGrid(fig[2, 1], (label="μ (Fermi level)", range=range(E_min, E_max, length=300), startvalue=Float64(E_Fermi)))
    mu_slider = sg.sliders[1].value
    iso_level = lift(μ -> Float32[μ], mu_slider)

    contour!(ax, (-π, π), (-π, π), (-π, π), E_volume; levels=iso_level, colormap=:viridis, alpha=0.5, transparency=true, kwargs...)
    return fig
end

function visualize_renormalized_bands(
    Ts::AbstractVector{<:Real},
    field::AuxiliaryField,
    model::ElectronicDispersion,
    interaction::Interaction,
    kgrid::AbstractKGrid,
    kpath::KPath
)

    # 提取画图用的横坐标距离
    distances = zeros(length(kpath.points))
    dist = 0.0
    for i in 2:length(kpath.points)
        dist += norm(kpath.points[i] - kpath.points[i-1])
        distances[i] = dist
    end
    tick_positions = distances[kpath.node_indices]

    # 修复：从传入的 kpath 中提取高对称点标签
    labels = kpath.node_labels

    # ==========================================
    # 3. 绘图循环
    # ==========================================
    fig = Figure(size=(1400, 420), fontsize=16) # 稍微增加一点高度以容纳换行标题

    for (i, T) in enumerate(Ts)
        # 3.1 求解当前温度下的真实基态能隙
        phi_gs = solve_ground_state(field, model, interaction, kgrid, ExactTrLn(); phi_guess=0.5, T=T)
        if phi_gs < 1e-4
            phi_gs = 0.0
        end # 超过 Tc 后抹平数值噪音

        # 3.2 将求得的能隙代入，构造此时的平均场 BdG 色散矩阵
        mf_disp = MeanFieldDispersion(model, field, phi_gs)

        # 3.3 沿 K-path 对角化 BdG 矩阵，提取粒子和空穴的两支能量
        bands = [real(band_structure(mf_disp, k).values) for k in kpath.points]
        E_hole = [b[1] for b in bands]    # 空穴支 (能量通常为负)
        E_part = [b[2] for b in bands]    # 粒子支 (能量通常为正)

        # 3.4 算一下原来的裸能带，作为背景对比
        bare_bands = [real(band_structure(model, k).values)[1] for k in kpath.points]

        # --- 核心修改 1：通过换行(\n)将标题分为上下两行，彻底解决横向重叠 ---
        title_str = "T = $(round(T, digits=2))   Δ = $(round(phi_gs, digits=2))"

        # --- 核心修改 2：只保留第一张图的 Y 轴刻度和标签 ---
        is_first = (i == 1)

        # --- 开始画这个子图 ---
        ax = Axis(fig[1, i],
            title=title_str,
            titlesize=16,
            xticks=(tick_positions, labels),
            ylabel=is_first ? "Energy E(k)" : "",
            xgridvisible=false,
            yticksvisible=is_first,       # 隐藏其余图的 Y 轴刻度线短横
            yticklabelsvisible=is_first   # 隐藏其余图的 Y 轴数字
        )

        # 画出费米能级和高对称点分隔线
        hlines!(ax, [0.0], color=:gray, linestyle=:dash, linewidth=1)
        vlines!(ax, tick_positions, color=:lightgray, linestyle=:dot, linewidth=1.5)

        # 画裸能带 (黑色虚线)
        lines!(ax, distances, bare_bands, color=:black, linestyle=:dot, alpha=0.4, linewidth=2, label="Bare Band")

        # 画重整化后的 BdG 准粒子能带
        lines!(ax, distances, E_hole, color=:royalblue, linewidth=3, label="Hole Band")
        lines!(ax, distances, E_part, color=:crimson, linewidth=3, label="Particle Band")

        # 统一 Y 轴范围，横向对比将非常完美
        ylims!(ax, -2.5, 2.5)
        xlims!(ax, distances[1], distances[end])

        if is_first
            axislegend(ax, position=:lt, framevisible=false)
        end
    end

    # --- 核心修改 3：强制调整子图之间的横向间距 ---
    # 因为隐藏了Y轴标签，子图可能会自动靠得太近，给个 15 的间距让图面保持呼吸感
    colgap!(fig.layout, 15)

    return fig
end