using StaticArrays

# Physics pipeline abstract types
abstract type PhysicalModel end
abstract type AuxiliaryField end
abstract type ApproximationLevel end

# Concrete approximation levels
struct ExactTrLn <: ApproximationLevel end
struct RPA <: ApproximationLevel end

# Concrete Auxiliary fields
# Represents a macroscopic, frozen condensate (e.g., T=0 CDW ground state)
struct StaticMeanField{D} <: AuxiliaryField
    q::SVector{D,Float64}
end
StaticMeanField(q::SVector{D, <:Real}) where D = StaticMeanField{D}(SVector{D, Float64}(q))

# Represents a propagating bosonic fluctuation with momentum q and frequency ω
struct DynamicalFluctuation{D} <: AuxiliaryField
    q::SVector{D,Float64}
    ω::Float64
end
DynamicalFluctuation(q::SVector{D, <:Real}, ω::Real) where D = DynamicalFluctuation{D}(SVector{D, Float64}(q), Float64(ω))

# Specific types can subtype or wrapper these if needed, 
# for now we provide these as the main concrete implementations.
struct ChargeDensityWave{D} <: AuxiliaryField
    q::SVector{D,Float64}
end

abstract type Dispersion <: PhysicalModel end
abstract type ElectronicDispersion <: Dispersion end
abstract type PhononDispersion <: Dispersion end

abstract type Smearing end

abstract type Interaction end
abstract type CoulombInteraction <: Interaction end
abstract type ElectronPhononInteraction <: Interaction end
abstract type ScreenedInteraction <: Interaction end
abstract type Polarization end

abstract type Propagator end

abstract type PhononPropagator <: Propagator end
abstract type ElectronPropagator <: Propagator end
abstract type GorkovPropagator <: Propagator end

abstract type SelfEnergy end

abstract type SpectralFunction end
abstract type ElectronSpectralFunction <: SpectralFunction end
abstract type PhononSpectralFunction <: SpectralFunction end

abstract type GapFunction end

abstract type AbstractKGrid{D} end

Base.length(g::AbstractKGrid) = length(g.points)
Base.iterate(g::AbstractKGrid, state=1) = iterate(g.points, state)
Base.getindex(g::AbstractKGrid, i::Int) = g.points[i]
Base.firstindex(g::AbstractKGrid) = 1
Base.lastindex(g::AbstractKGrid) = length(g.points)

# Effective Action struct
struct EffectiveAction{M<:PhysicalModel,F<:AuxiliaryField,G<:AbstractKGrid}
    model::M
    field::F
    grid::G
    V_bare::Float64
end

"""
    KGrid{D} <: AbstractKGrid{D}

A concrete generic implementation of a D-dimensional K-grid.
Contains the grid `points` as `SVector{D, Float64}` and corresponding 
integration `weights`.
"""
struct KGrid{D} <: AbstractKGrid{D}
    points::Vector{SVector{D,Float64}}
    weights::Vector{Float64}
end

Base.eltype(::Type{<:AbstractKGrid{D}}) where {D} = SVector{D,Float64}

"""
    KPath{D} <: AbstractKGrid{D}

用于绘制能带图的高对称线路径。
不仅包含 k 点，还记录了高对称点（拐点）的索引和标签。
"""
struct KPath{D} <: AbstractKGrid{D}
    points::Vector{SVector{D,Float64}}
    weights::Vector{Float64}       # 画图不需要权重，随便填即可
    node_indices::Vector{Int}      # 记录高对称点在 points 数组中的索引
    node_labels::Vector{String}    # 高对称点的名字，如 ["Γ", "X", "M", "Γ"]
end

# 辅助函数：根据给定的一系列高对称点生成 KPath
function generate_kpath(nodes::Vector{SVector{D,Float64}}, labels::Vector{String}; n_pts_per_segment=50) where {D}
    points = SVector{D,Float64}[]
    node_indices = Int[]

    current_idx = 1
    push!(node_indices, current_idx)

    for i in 1:(length(nodes)-1)
        start_node = nodes[i]
        end_node = nodes[i+1]

        # 线性插值
        for j in 1:n_pts_per_segment
            t = (j - 1) / n_pts_per_segment
            k = start_node + t * (end_node - start_node)
            push!(points, k)
        end

        current_idx += n_pts_per_segment
        push!(node_indices, current_idx)
    end

    # 补上最后一个点
    push!(points, nodes[end])
    weights = zeros(length(points))

    return KPath{D}(points, weights, node_indices, labels)
end
