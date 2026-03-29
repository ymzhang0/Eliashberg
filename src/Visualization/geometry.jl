# ============================================================================
# 6. Lattice & Brillouin Zone Visualization
# ============================================================================


# ----------------------------------------------------------------------------
# Real Space Lattice Visualizations
# ----------------------------------------------------------------------------

"""
    visualize_lattice(lattice::Lattice{1}; extent=3, kwargs...)

Visualizes a 1D real-space lattice chain.
"""
function visualize_lattice(lattice::Lattice{1}; extent=3, axis=(;), kwargs...)
    # 把高度调低一点，因为是一维的，太高了显得空旷
    fig = Figure(size=(600, 200))

    # 核心修改：通过 Makie 的 Axis 属性彻底干掉外框和不必要的网格
    ax = Axis(fig[1, 1];
        title="1D Real Space Lattice",
        xlabel="x",
        yticksvisible=false,
        yticklabelsvisible=false,
        ygridvisible=false,        # 去掉水平网格线
        leftspinevisible=false,    # 去掉左外框
        rightspinevisible=false,   # 去掉右外框
        topspinevisible=false,     # 去掉上外框
        bottomspinevisible=false,  # 去掉下外框（因为我们要把轴画在原子上）
        axis...
    )

    a1 = lattice.vectors[1, 1]

    # 1. 画一条贯穿所有原子的中心轴线（充当真正的坐标轴）
    hlines!(ax, [0.0], color=:black, linewidth=1.5)

    # Generate points
    xs = [i * a1 for i in -extent:extent]
    ys = zeros(length(xs))

    # 2. 画原子：建议把颜色改成白底黑边（类似空心圆），压在黑线上会极其好看和专业
    scatter!(ax, xs, ys;
        color=:white, strokecolor=:gray, strokewidth=2,
        markersize=15, label="Lattice Sites", kwargs...
    )

    # Draw primitive vector
    arrows2d!(ax, [0.0], [0.0], [a1], [0.0]; color=:red, shaftwidth=4, tipwidth=15, tiplength=15)

    # Add origin marker (原点用实心黑点标出)
    scatter!(ax, [0.0], [0.0]; color=:black, markersize=15)

    # 3. 锁定 Y 轴范围，防止图表上下乱飘，确保链条绝对居中
    ylims!(ax, -1, 1)

    return fig
end

"""
    visualize_lattice(lattice::Lattice{2}; extent=2, kwargs...)

Visualizes a 2D real-space lattice with hollow sites, primitive vectors, 
magnitudes, and the angle between them (without bounding boxes or grid lines).
"""
function visualize_lattice(lattice::Lattice{2}; extent=2, axis=(;), kwargs...)
    fig = Figure(size=(600, 600))

    # 1. 彻底干掉边框、网格和坐标轴刻度
    ax = Axis(fig[1, 1];
        aspect=DataAspect(),
        title="2D Real Space Lattice",
        xgridvisible=false, ygridvisible=false,
        xticksvisible=false, yticksvisible=false,
        xticklabelsvisible=false, yticklabelsvisible=false,
        leftspinevisible=false, rightspinevisible=false,
        topspinevisible=false, bottomspinevisible=false,
        axis...
    )

    a1 = lattice.vectors[:, 1]
    a2 = lattice.vectors[:, 2]

    # Generate surrounding lattice points
    pts = [lattice.vectors * SVector(i, j) for i in -extent:extent for j in -extent:extent]
    xs = [p[1] for p in pts]
    ys = [p[2] for p in pts]

    # 2. 画空心原子：白底，灰色描边
    scatter!(ax, xs, ys;
        color=:white, strokecolor=:gray, strokewidth=2,
        markersize=12, label="Lattice Sites", kwargs...
    )

    # Draw primitive basis vectors from origin
    arrows2d!(ax, [0.0, 0.0], [0.0, 0.0], [a1[1], a2[1]], [a1[2], a2[2]];
        color=[:red, :blue], shaftwidth=4, tipwidth=15, tiplength=15)

    # 3. 标注原点
    scatter!(ax, [0.0], [0.0]; color=:black, markersize=12)

    # ---------------------------------------------------------
    # 4. 自动标注基矢长度和夹角
    # ---------------------------------------------------------
    len_a1 = norm(a1)
    len_a2 = norm(a2)

    # 在基矢的中点附近标上长度
    text!(ax, a1[1] / 2, a1[2] / 2; text="|a₁| = $(round(len_a1, digits=2))", color=:red, offset=(5, 5), align=(:left, :bottom))
    text!(ax, a2[1] / 2, a2[2] / 2; text="|a₂| = $(round(len_a2, digits=2))", color=:blue, offset=(5, 5), align=(:left, :bottom))

    # 计算两个向量与 x 轴的绝对极角
    ang1 = atan(a1[2], a1[1])
    ang2 = atan(a2[2], a2[1])

    # 确定画弧线的起止角度 (保证画的是较小的那个夹角)
    start_ang, end_ang = min(ang1, ang2), max(ang1, ang2)
    if end_ang - start_ang > π
        start_ang, end_ang = end_ang, start_ang + 2π
    end

    # 计算度数
    theta_deg = round(abs(end_ang - start_ang) * 180 / π, digits=1)

    # 画夹角弧线 (半径取较短基矢的 30%)
    arc_radius = min(len_a1, len_a2) * 0.3
    arc!(ax, Point2f(0.0, 0.0), arc_radius, start_ang, end_ang, color=:black, linewidth=1.5)

    # 把角度数字写在弧线外侧一点点的位置
    mid_ang = (start_ang + end_ang) / 2
    text!(ax, arc_radius * 1.4 * cos(mid_ang), arc_radius * 1.4 * sin(mid_ang);
        text="$(theta_deg)°", color=:black, align=(:center, :center))

    return fig
end

"""
    visualize_lattice(lattice::Lattice{3}; extent=1, kwargs...)

Visualizes a 3D real-space lattice with hollow sites, primitive vectors, 
magnitudes, and the angles between them (without bounding boxes or grids).
"""
function visualize_lattice(lattice::Lattice{3}; extent=1, axis=(;), kwargs...)
    fig = Figure(size=(800, 800))

    # 1. 彻底干掉 3D 边框、网格和刻度标签
    ax = Axis3(fig[1, 1];
        aspect=:data,
        title="3D Real Space Lattice",
        xgridvisible=false, ygridvisible=false, zgridvisible=false,
        xticksvisible=false, yticksvisible=false, zticksvisible=false,
        xticklabelsvisible=false, yticklabelsvisible=false, zticklabelsvisible=false,
        xspinesvisible=false, yspinesvisible=false, zspinesvisible=false,
        axis...
    )

    a1, a2, a3 = lattice.vectors[:, 1], lattice.vectors[:, 2], lattice.vectors[:, 3]

    # Generate surrounding lattice points
    pts = [lattice.vectors * SVector(i, j, k) for i in -extent:extent for j in -extent:extent for k in -extent:extent]

    # 2. 空心原子：白底灰边
    scatter!(ax, [p[1] for p in pts], [p[2] for p in pts], [p[3] for p in pts];
        color=:white, strokecolor=:gray, strokewidth=2, markersize=12, label="Lattice Sites", kwargs...)

    # 3. 三维箭头：使用 shaftradius 和 tipradius，它们使用的是数据坐标空间的大小
    arrows3d!(ax,
        [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
        [a1[1], a2[1], a3[1]], [a1[2], a2[2], a3[2]], [a1[3], a2[3], a3[3]];
        color=[:red, :blue, :green],
        shaftradius=0.03, tipradius=0.08, tiplength=0.25)

    # 原点标注
    scatter!(ax, [0.0], [0.0], [0.0]; color=:black, markersize=12)

    # ---------------------------------------------------------
    # 4. 自动标注基矢长度和三维空间夹角
    # ---------------------------------------------------------
    len_a1, len_a2, len_a3 = norm(a1), norm(a2), norm(a3)

    # 标注长度 (文字放在基矢中点)
    text!(ax, a1[1] / 2, a1[2] / 2, a1[3] / 2; text="|a₁| = $(round(len_a1, digits=2))", color=:red, align=(:left, :bottom))
    text!(ax, a2[1] / 2, a2[2] / 2, a2[3] / 2; text="|a₂| = $(round(len_a2, digits=2))", color=:blue, align=(:left, :bottom))
    text!(ax, a3[1] / 2, a3[2] / 2, a3[3] / 2; text="|a₃| = $(round(len_a3, digits=2))", color=:green, align=(:left, :bottom))

    # 内部辅助函数：在三维空间任意两个向量之间画圆弧并标注角度
    function draw_angle3d!(v1, v2)
        u1, u2 = normalize(v1), normalize(v2)
        ang = acos(clamp(dot(u1, u2), -1.0, 1.0)) # 计算夹角

        if ang > 1e-5
            # 计算这两个向量构成的平面的法向量
            n = normalize(cross(u1, u2))
            # 计算面内与 u1 垂直的向量
            w = cross(n, u1)

            # 弧线半径取基矢最短长度的 30%
            r = min(len_a1, len_a2, len_a3) * 0.3

            # 生成三维圆弧上的点
            arc_pts = [Point3f(r * (cos(t) * u1 + sin(t) * w)) for t in range(0, ang, length=30)]
            lines!(ax, arc_pts; color=:black, linewidth=2)

            # 在圆弧外侧标上角度文字
            mid_t = ang / 2
            mid_pos = r * 1.3 * (cos(mid_t) * u1 + sin(mid_t) * w)
            text!(ax, mid_pos[1], mid_pos[2], mid_pos[3];
                text="$(round(ang*180/π, digits=1))°", color=:black, align=(:center, :center))
        end
    end

    # 分别画出 a1-a2, a2-a3, a3-a1 之间的夹角
    draw_angle3d!(a1, a2)
    draw_angle3d!(a2, a3)
    draw_angle3d!(a3, a1)

    return fig
end

# ----------------------------------------------------------------------------
# Reciprocal Space & K-Grid Visualizations
# ----------------------------------------------------------------------------

"""
    visualize_reciprocal_space(lattice::Lattice{1}, kgrid=nothing; kwargs...)

Visualizes the 1D reciprocal lattice and 1st Brillouin Zone.
"""
function visualize_reciprocal_space(lattice::Lattice{1}, kgrid::Union{AbstractKGrid,Nothing}=nothing; axis=(;), kwargs...)
    fig = Figure(size=(600, 300))
    ax = Axis(fig[1, 1];
        title="1D Reciprocal Space & K-Grid",
        xlabel=L"k",
        yticksvisible=false,
        yticklabelsvisible=false,
        ygridvisible=false,        # 去掉水平网格线
        leftspinevisible=false,    # 去掉左外框
        rightspinevisible=false,   # 去掉右外框
        topspinevisible=false,     # 去掉上外框
        bottomspinevisible=false,  # 去掉下外框（因为我们要把轴画在原子上）
        axis...
    )
    b1 = reciprocal_vectors(lattice)[1, 1]

    # Draw 1st Brillouin Zone boundaries
    vlines!(ax, [-b1 / 2, b1 / 2]; color=:black, linestyle=:dash, linewidth=2, label="1st BZ Boundaries")

    if !isnothing(kgrid)
        kxs = [k[1] for k in kgrid.points]
        scatter!(ax, kxs, zeros(length(kxs)); color=:purple, markersize=10, alpha=0.7, label="K-Grid Points")
    end

    arrows2d!(ax, [0.0], [0.0], [b1], [0.0]; color=:darkred, shaftwidth=3, tipwidth=15, tiplength=15)
    scatter!(ax, [0.0], [0.0]; color=:gold, markersize=15, marker=:star5, label=L"\Gamma")

    axislegend(ax; position=:rt)
    return fig
end

"""
    visualize_reciprocal_space(lattice::Lattice{2}, kgrid=nothing; kwargs...)

Visualizes the 2D reciprocal lattice vectors and the primitive Brillouin Zone parallelogram.
"""
function visualize_reciprocal_space(lattice::Lattice{2}, kgrid::Union{AbstractKGrid,Nothing}=nothing; axis=(;), kwargs...)
    fig = Figure(size=(700, 700))
    ax = Axis(fig[1, 1]; aspect=DataAspect(), title="2D Reciprocal Space & K-Grid",
        xlabel=L"k_x", ylabel=L"k_y", axis...)

    B = reciprocal_vectors(lattice)
    b1, b2 = B[:, 1], B[:, 2]

    # Centered primitive cell (Parallelogram spanning exactly one Brillouin Zone volume)
    cell_corners_x = [0.0, b1[1], b1[1] + b2[1], b2[1], 0.0] .- (b1[1] + b2[1]) / 2
    cell_corners_y = [0.0, b1[2], b1[2] + b2[2], b2[2], 0.0] .- (b1[2] + b2[2]) / 2
    lines!(ax, cell_corners_x, cell_corners_y; color=:black, linestyle=:dash, linewidth=2, label="Primitive Cell (1 BZ Volume)")

    if !isnothing(kgrid)
        kxs = [k[1] for k in kgrid.points]
        kys = [k[2] for k in kgrid.points]
        scatter!(ax, kxs, kys; color=:purple, markersize=6, alpha=0.7, label="K-Grid Points")
    end

    arrows2d!(ax, [0.0, 0.0], [0.0, 0.0], [b1[1], b2[1]], [b1[2], b2[2]];
        color=[:darkred, :darkblue], shaftwidth=3, tipwidth=15, tiplength=15)

    scatter!(ax, [0.0], [0.0]; color=:gold, markersize=15, marker=:star5, label=L"\Gamma")

    axislegend(ax; position=:rt)
    return fig
end

"""
    visualize_reciprocal_space(lattice::Lattice{3}, kgrid=nothing; kwargs...)

Visualizes the 3D reciprocal primitive vectors and the grid points inside the primitive cell.
"""
function visualize_reciprocal_space(lattice::Lattice{3}, kgrid::Union{AbstractKGrid,Nothing}=nothing; axis=(;), kwargs...)
    fig = Figure(size=(800, 800))
    ax = Axis3(fig[1, 1]; aspect=:data, title="3D Reciprocal Space & K-Grid",
        xlabel=L"k_x", ylabel=L"k_y", zlabel=L"k_z", axis...)

    B = reciprocal_vectors(lattice)
    b1, b2, b3 = B[:, 1], B[:, 2], B[:, 3]

    # Draw the 12 edges of the centered primitive parallelepiped
    # Vertices combinations: ±b1/2 ±b2/2 ±b3/2
    corners = [(i * b1 + j * b2 + k * b3) / 2 for i in [-1, 1] for j in [-1, 1] for k in [-1, 1]]
    # Connect edges
    for i in 1:8, j in i+1:8
        diff = corners[i] - corners[j]
        # If the difference is roughly equal to one of the basis vectors, it's an edge
        if any(v -> norm(abs.(diff) - abs.(v)) < 1e-5, [b1, b2, b3])
            lines!(ax, [corners[i][1], corners[j][1]],
                [corners[i][2], corners[j][2]],
                [corners[i][3], corners[j][3]]; color=:black, linestyle=:dash, alpha=0.5)
        end
    end

    if !isnothing(kgrid)
        kxs = [k[1] for k in kgrid.points]
        kys = [k[2] for k in kgrid.points]
        kzs = [k[3] for k in kgrid.points]
        scatter!(ax, kxs, kys, kzs; color=:purple, markersize=4, alpha=0.5)
    end

    arrows3d!(ax, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
        [b1[1], b2[1], b3[1]], [b1[2], b2[2], b3[2]], [b1[3], b2[3], b3[3]];
        color=[:darkred, :darkblue, :darkgreen], shaftwidth=4, tipwidth=15, tiplength=15)

    scatter!(ax, [0.0], [0.0], [0.0]; color=:gold, markersize=15, marker=:star5, label=L"\Gamma")

    return fig
end