
struct ConstantInteraction <: ElectronPhononInteraction
    V0::Float64
end

struct LocalInteraction <: ElectronPhononInteraction
    V0::Float64
    threshold::Float64
    fsthick::Float64
end

struct YukawaInteraction <: ElectronPhononInteraction
    V0::Float64
    λ::Float64
end

struct LimitedConstantInteraction{D,M<:ElectronicDispersion{D}} <: ElectronPhononInteraction
    V0::Float64
    ωc::Float64
    dispersion::M
end

struct FanMigdalInteraction{
    G<:ElectronPropagator,
    D<:PhononPropagator
} <: ElectronPhononInteraction
    electron::G
    phonon::D
end

struct BareCoulombInteraction <: CoulombInteraction
    cutoff::Float64
end

# Legacy polarization types removed. GeneralizedSusceptibility is the new source of truth.

struct ScreenedCoulombInteraction{
    I<:CoulombInteraction,
    P<:Polarization
} <: ScreenedInteraction
    bare::I
    polarization::P
end

"""
    MediatedInteraction{B, C, V<:Interaction} <: Interaction

A universal interface for effective electron-electron interactions mediated by a
bosonic field with the form

    V_eff(q) = V_bare(q) + |g(q)|^2 * D_boson(q)
"""
struct MediatedInteraction{B,C,V<:Interaction} <: Interaction
    bare_V::V
    boson_field::B
    coupling::C
end

struct CompositeInteraction{T<:Tuple} <: Interaction
    interactions::T
end
CompositeInteraction(ints::Vararg{Interaction}) = CompositeInteraction(ints)

Base.length(comp::CompositeInteraction) = length(comp.interactions)

# Legacy chi methods removed.

function _nearest_grid_index(q::SVector{D,Float64}, qgrid::AbstractKGrid{D}) where {D}
    length(qgrid) > 0 || throw(ArgumentError("Cannot evaluate a boson propagator on an empty q-grid."))

    best_idx = firstindex(qgrid)
    best_distance = norm(q - qgrid[best_idx])

    for idx in (best_idx + 1):lastindex(qgrid)
        distance = norm(q - qgrid[idx])
        if distance < best_distance
            best_idx = idx
            best_distance = distance
        end
    end

    return best_idx
end

function evaluate_boson_propagator(q::SVector{D,Float64}, boson_field) where {D}
    applicable(boson_field, q) ||
        throw(ArgumentError("No boson propagator evaluator is defined for $(typeof(boson_field))."))
    return boson_field(q)
end

function evaluate_boson_propagator(
    q::SVector{D,Float64},
    boson_field::Tuple{<:AbstractKGrid{D},<:AbstractArray},
) where {D}
    qgrid, chi_array = boson_field
    length(qgrid) == length(chi_array) ||
        throw(DimensionMismatch("Susceptibility data length must match the q-grid length."))

    idx = _nearest_grid_index(q, qgrid)
    return chi_array[idx]
end

function evaluate_boson_propagator(
    q::SVector{D,Float64},
    boson_field::PhononDispersion{D},
) where {D}
    return -2.0 / ω(q, boson_field)
end

evaluate_coupling(::SVector{D,Float64}, coupling::Number) where {D} = coupling

function evaluate_coupling(q::SVector{D,Float64}, coupling) where {D}
    applicable(coupling, q) ||
        throw(ArgumentError("Interaction coupling $(typeof(coupling)) must be a number or callable as g(q)."))
    return coupling(q)
end

function V(
    interaction::ConstantInteraction,
)
    return interaction.V0
end

function V(
    q::SVector{D,Float64},
    interaction::ConstantInteraction
) where {D}
    return V(interaction)
end

function V(
    k::SVector{D,Float64},
    kp::SVector{D,Float64},
    interaction::LocalInteraction,
    dispersion_model::ElectronicDispersion{D}
) where {D}
    # Uses band structure energies nearest to Fermi surface
    Ek = band_structure(dispersion_model, k).values
    Ekp = band_structure(dispersion_model, kp).values

    # Check if any band is within fsthick
    if any(abs.(Ek) .≤ interaction.fsthick) && any(abs.(Ekp) .≤ interaction.fsthick)
        return norm(kp + k) < interaction.threshold ? interaction.V0 : 0.0
    else
        return 0.0
    end
end

function V(
    q::SVector{D,Float64},
    interaction::LocalInteraction
) where {D}
    return norm(q) < interaction.threshold ? interaction.V0 : 0.0
end

function V(
    k::SVector{D,Float64},
    kp::SVector{D,Float64},
    interaction::YukawaInteraction,
) where {D}
    return V(k - kp, interaction)
end

function V(
    q::SVector{D,Float64},
    interaction::YukawaInteraction
) where {D}
    return interaction.V0 / (norm(q)^2 + interaction.λ^2)
end

function V(
    k::SVector{D,Float64},
    kp::SVector{D,Float64},
    interaction::LimitedConstantInteraction{D},
    dispersion_model::ElectronicDispersion{D}
) where {D}
    Ek = band_structure(dispersion_model, k).values
    Ekp = band_structure(dispersion_model, kp).values
    if any(abs.(Ek) .≤ interaction.ωc) && any(abs.(Ekp) .≤ interaction.ωc)
        return interaction.V0
    else
        return 0.0
    end
end

function V(
    q::SVector{D,Float64},
    interaction::LimitedConstantInteraction{D}
) where {D}
    return interaction.V0
end

function V(
    q::SVector{D,Float64},
    interaction::BareCoulombInteraction
) where {D}
    q_norm = norm(q)
    return q_norm < interaction.cutoff ? 0.0 : 1.0 / (q_norm^2)
end

function V(
    q::SVector{D,Float64},
    kgrid::AbstractKGrid{D},
    screened_interaction::ScreenedCoulombInteraction
) where {D}
    return V(q, screened_interaction)
end

function V(
    q::SVector{D,Float64},
    screened_interaction::ScreenedCoulombInteraction
) where {D}
    bare_V = V(q, screened_interaction.bare)
    chi_val = screened_interaction.polarization(q)
    return bare_V / (1 + bare_V * chi_val)
end

# FanMigdal placeholder implementations 
function V(
    q::SVector{D,Float64},
    kgrid::AbstractKGrid{D},
    interaction::FanMigdalInteraction
) where {D}
    # Placeholder screened interaction for FanMigdal
    return V(q, interaction)
end

function V(
    q::SVector{D,Float64},
    interaction::FanMigdalInteraction
) where {D}
    return 0.0
end

function V(
    q::SVector{D,Float64},
    interaction::MediatedInteraction
) where {D}
    bare_V = V(q, interaction.bare_V)
    g = evaluate_coupling(q, interaction.coupling)
    D_q = evaluate_boson_propagator(q, interaction.boson_field)
    return bare_V + abs2(g) * D_q
end

function Σ(
    k::SVector{D,Float64},
    ω::Float64,
    qgrid::AbstractKGrid{D},
    interaction::FanMigdalInteraction{G,Ph},
) where {D,G,Ph}
    # Placeholder generic integral over qgrid
    # Needs g2 matrix element integration correctly adapted for D-dimensions
    return 0.0
end

# ============================================================================
# 复合相互作用 (Composite Interaction)
# 允许将多个相互作用（如声子吸引 + 库仑排斥）线性叠加
# ============================================================================

struct CombinedInteraction{T<:Tuple} <: Interaction
    interactions::T
end

# 重载 Julia 的 '+' 运算符！
Base.:+(a::Interaction, b::Interaction) = CombinedInteraction((a, b))
Base.:+(a::CombinedInteraction, b::Interaction) = CombinedInteraction((a.interactions..., b))

# 当需要计算动量 q 处的矩阵元时，直接把所有子相互作用的值加起来
function V(q::SVector{D,Float64}, interaction::CombinedInteraction) where {D}
    return sum(V(q, int) for int in interaction.interactions)
end

function V(k::SVector{D,Float64}, kp::SVector{D,Float64}, interaction::CombinedInteraction, disp::ElectronicDispersion{D}) where {D}
    return sum(V(k, kp, int, disp) for int in interaction.interactions)
end

# ============================================================================
# Universal Interaction Interface Fallbacks
# 保证任何相互作用都能安全响应 V(k, kp, interaction, dispersion) 的大一统调用
# ============================================================================

function V(k::SVector{D,Float64}, kp::SVector{D,Float64}, interaction::Interaction, disp::ElectronicDispersion{D}) where {D}
    # 1. 如果它实现了 V(q, int) (如 Yukawa, Coulomb)，自动转换 k - kp
    if hasmethod(V, Tuple{SVector{D,Float64},typeof(interaction)})
        return V(k - kp, interaction)
        # 2. 如果它实现了 V(int) (如 Constant)，直接调用
    elseif hasmethod(V, Tuple{typeof(interaction)})
        return V(interaction)
    else
        error("Interaction $(typeof(interaction)) must implement a specific V(...) method.")
    end
end
