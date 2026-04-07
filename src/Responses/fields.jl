# Responses/fields.jl

# ----------------------------------------------------------------------------
# Auxiliary Field Concrete implementations
# ----------------------------------------------------------------------------

"""
    DirectChannel{D} <: ParticleHoleChannel{D}

代表粒子-空穴直接通道 (Particle-Hole Direct Channel)。
物理上对应电荷密度涨落 (Charge Density Fluctuations)。
由长程库仑排斥力 V(q) 驱动，
在 q=0 处最强。
"""
struct DirectChannel <: ParticleHoleChannel{Any} end

"""
    ExchangeChannel{Dir} <: ParticleHoleChannel{Any}

代表粒子-空穴交换通道 (Particle-Hole Exchange Channel)。
对应自旋密度涨落 (Spin Density Fluctuations)，
耦合到自旋算符 `S^α`。`direction` 可取 `:z`, `:x`, `:y`, `:transverse`，
默认是纵向的 `:z` 通道。
"""
struct ExchangeChannel{Dir} <: ParticleHoleChannel{Any} end
ExchangeChannel(direction::Symbol=:z) = ExchangeChannel{_validate_spin_direction(direction)}()
"""
    ChargeDensityWave{D} <: ParticleHoleChannel{D}

代表宏观凝聚的电荷密度波序。
(物理上它是 DirectChannel 发生相变后的产物，基矢结构完全一致)
"""
struct ChargeDensityWave{D} <: ParticleHoleChannel{D}
    q::SVector{D,Float64}
end

_to_static_momentum(q::AbstractVector{<:Real}) = SVector{length(q),Float64}(q...)
_to_static_momentum(q::Tuple{Vararg{Real}}) = SVector{length(q),Float64}(q...)

ChargeDensityWave(q::AbstractVector{<:Real}) = ChargeDensityWave(_to_static_momentum(q))
ChargeDensityWave(q::Tuple{Vararg{Real}}) = ChargeDensityWave(_to_static_momentum(q))

"""
    SpinDensityWave{D,Dir} <: ParticleHoleChannel{D}

代表宏观凝聚的自旋密度波序。
它耦合到动量 `q` 处的自旋算符 `S^α(q)`，其中 `direction` 指定序参量方向，
可取 `:z`, `:x`, `:y`, `:transverse`，默认 `:z`。
"""
struct SpinDensityWave{D,Dir} <: ParticleHoleChannel{D}
    q::SVector{D,Float64}
end

SpinDensityWave(q::SVector{D,Float64}, direction::Symbol=:z) where {D} =
    SpinDensityWave{D,_validate_spin_direction(direction)}(q)
SpinDensityWave(q::SVector{D,<:Real}, direction::Symbol=:z) where {D} =
    SpinDensityWave(SVector{D,Float64}(q), direction)
SpinDensityWave(q::AbstractVector{<:Real}, direction::Symbol=:z) =
    SpinDensityWave(_to_static_momentum(q), direction)
SpinDensityWave(q::Tuple{Vararg{Real}}, direction::Symbol=:z) =
    SpinDensityWave(_to_static_momentum(q), direction)

"""
    BCSReducedPairing <: AuxiliaryField

Represents a standard uniform superconducting condensate (q=0).
Pairs electrons with momenta (k, -k).
"""
struct BCSReducedPairing <: AuxiliaryField
    symmetry::Symbol
end
BCSReducedPairing() = BCSReducedPairing(:s_wave)

"""
    CompositeField{T<:Tuple} <: AuxiliaryField

Container that groups multiple auxiliary fields into a single composite order.
The fields are applied sequentially when constructing the nested mean-field
Hamiltonian.
"""
struct CompositeField{T<:Tuple} <: AuxiliaryField
    fields::T
end
CompositeField(fields::Vararg{AuxiliaryField}) = CompositeField(fields)

Base.length(comp::CompositeField) = length(comp.fields)
Base.iterate(comp::CompositeField, state=1) = iterate(comp.fields, state)
Base.getindex(comp::CompositeField, i::Int) = comp.fields[i]

"""
    FFLOPairing{D} <: AuxiliaryField

Represents the Fulde-Ferrell (FF) state with a single center-of-mass momentum `q`.
Pairs electrons with momenta (k, -k+q).
"""
struct FFLOPairing{D} <: AuxiliaryField
    q::SVector{D,Float64}
    symmetry::Symbol
    h::Float64 # Zeeman magnetic field strength
end
FFLOPairing(q::SVector{D,Float64}, h::Float64=0.0) where {D} = FFLOPairing{D}(q, :s_wave, h)
FFLOPairing(q::SVector{D,<:Real}, h::Real=0.0) where {D} = FFLOPairing(SVector{D,Float64}(q), Float64(h))
FFLOPairing(q::AbstractVector{<:Real}, h::Real=0.0) = FFLOPairing(_to_static_momentum(q), Float64(h))
FFLOPairing(q::Tuple{Vararg{Real}}, h::Real=0.0) = FFLOPairing(_to_static_momentum(q), Float64(h))

"""
    PairDensityWave{D} <: AuxiliaryField

Represents a commensurate Pair Density Wave (LO-like state) with standing wave modulation.
Couples the electron at `k` to holes at both `-k+q` and `-k-q`.
"""
struct PairDensityWave{D} <: AuxiliaryField
    q::SVector{D,Float64}
    symmetry::Symbol
end
PairDensityWave(q::SVector{D,Float64}) where {D} = PairDensityWave{D}(q, :s_wave)
PairDensityWave(q::SVector{D,<:Real}) where {D} = PairDensityWave(SVector{D,Float64}(q))
PairDensityWave(q::AbstractVector{<:Real}) = PairDensityWave(_to_static_momentum(q))
PairDensityWave(q::Tuple{Vararg{Real}}) = PairDensityWave(_to_static_momentum(q))

"""
    MomentumDependentPairing{D, T} <: AuxiliaryField

代表一个完全由微观相互作用驱动的、在动量网格上具有连续分布的超导能隙。
它不再依赖于任何先验的对称性假设 (如 :d_wave)。
"""
struct MomentumDependentPairing{D, T} <: AuxiliaryField
    # 存储网格上每一个点的能隙值
    gap_values::Vector{T} 
end

# 初始化的便利构造器：传入一个全 0 数组，或者给一点微小的随机噪声/d波种子破缺对称性
function MomentumDependentPairing(kgrid::AbstractKGrid{D}; seed=:d_wave, amp=0.01) where {D}
    gaps = zeros(ComplexF64, length(kgrid))
    if seed == :random
        for i in eachindex(gaps)
            gaps[i] = amp * (rand() - 0.5 + im * (rand() - 0.5)) # 小随机复数扰动
        end
    elseif seed == :s_wave
        for (i, k) in enumerate(kgrid.points)
            gaps[i] = amp # s-wave 形状的初始能隙分布
        end
    elseif seed == :d_wave && D == 2
        for (i, k) in enumerate(kgrid.points)
            gaps[i] = amp * (cos(k[1]) - cos(k[2]))
        end
    else
        @warn "Unsupported seed type $D dimension $seed for MomentumDependentPairing. Initializing with zeros."
    end
    return MomentumDependentPairing{D, ComplexF64}(gaps)
end

"""
    StaticMeanField{D} <: AuxiliaryField

Represents a macroscopic, frozen condensate (e.g., T=0 CDW ground state).
"""
struct StaticMeanField{D} <: AuxiliaryField
    q::SVector{D,Float64}
end
StaticMeanField(q::SVector{S,<:Real}) where S = StaticMeanField{S}(SVector{S,Float64}(q))
StaticMeanField(q::AbstractVector{<:Real}) = StaticMeanField(_to_static_momentum(q))
StaticMeanField(q::Tuple{Vararg{Real}}) = StaticMeanField(_to_static_momentum(q))

"""
    DynamicalFluctuation{D} <: AuxiliaryField

Represents a propagating bosonic fluctuation with momentum q and frequency ω.
"""
struct DynamicalFluctuation{D} <: AuxiliaryField
    q::SVector{D,Float64}
    ω::Float64
end
DynamicalFluctuation(q::SVector{D,<:Real}, ω::Real) where D = DynamicalFluctuation{D}(SVector{D,Float64}(q), Float64(ω))
DynamicalFluctuation(q::AbstractVector{<:Real}, ω::Real) = DynamicalFluctuation(_to_static_momentum(q), Float64(ω))
DynamicalFluctuation(q::Tuple{Vararg{Real}}, ω::Real) = DynamicalFluctuation(_to_static_momentum(q), Float64(ω))

function _validate_spin_direction(direction::Symbol)
    direction in (:z, :x, :y, :transverse) && return direction
    throw(ArgumentError("Unsupported spin direction $direction. Use :z, :x, :y, or :transverse."))
end
