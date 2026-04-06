struct ExactTrLn <: ApproximationLevel end
struct RPA <: ApproximationLevel end

struct ExactTrLnContributionKernel{M}
    dispersion::M
    temperature::Float64
end

function (kernel::ExactTrLnContributionKernel)(k::SVector)
    tr_ln_contribution = 0.0

    for band_energy in real(band_structure(kernel.dispersion, k).values)
        if band_energy < 0
            tr_ln_contribution += band_energy - kernel.temperature * log1p(exp(band_energy / kernel.temperature))
        else
            tr_ln_contribution += -kernel.temperature * log1p(exp(-band_energy / kernel.temperature))
        end
    end

    return tr_ln_contribution
end

_field_wavevector(::AuxiliaryField, ::AbstractKGrid{D}) where {D} = zero(SVector{D,Float64})
_field_wavevector(field::ChargeDensityWave{D}, ::AbstractKGrid{D}) where {D} = field.q
_field_wavevector(field::SpinDensityWave{D}, ::AbstractKGrid{D}) where {D} = field.q
_field_wavevector(field::FFLOPairing{D}, ::AbstractKGrid{D}) where {D} = field.q
_field_wavevector(field::PairDensityWave{D}, ::AbstractKGrid{D}) where {D} = field.q
_field_wavevector(field::StaticMeanField{D}, ::AbstractKGrid{D}) where {D} = field.q
_field_wavevector(field::DynamicalFluctuation{D}, ::AbstractKGrid{D}) where {D} = field.q

function _quadratic_action_term(
    phi::Real,
    field::AuxiliaryField,
    interaction::Interaction,
    kgrid::AbstractKGrid
)
    q_vec = _field_wavevector(field, kgrid)
    return Float64(phi)^2 / abs(V(q_vec, interaction))
end

_quadratic_action_term(::Tuple{}, ::AbstractVector{<:Real}, ::Interaction, ::AbstractKGrid, ::Int) = 0.0

function _quadratic_action_term(
    ::Tuple{},
    ::AbstractVector{<:Real},
    ::Tuple{},
    ::AbstractKGrid,
    ::Int
)
    return 0.0
end

function _quadratic_action_term(
    fields::Tuple{F,Vararg{AuxiliaryField}},
    phis::AbstractVector{<:Real},
    interactions::Tuple{I,Vararg{Interaction}},
    kgrid::AbstractKGrid,
    idx::Int
) where {F<:AuxiliaryField,I<:Interaction}
    current_term = _quadratic_action_term(phis[idx], first(fields), first(interactions), kgrid)
    return current_term + _quadratic_action_term(Base.tail(fields), phis, Base.tail(interactions), kgrid, idx + 1)
end

"""
    evaluate_action(phi, field, model, interaction, kgrid, ::ExactTrLn; T=1e-3)

Evaluates the exact Tr[ln] effective action at a specific order parameter `phi`.
- `field`: The symmetry breaking channel (e.g., ChargeDensityWave, SuperconductingPairing)
- `model`: The bare electronic dispersion
- `interaction`: The effective interaction potential (e.g., Constant, Bardeen-Pines)
- `kgrid`: The numerical integration grid
"""
function evaluate_action(
    phi::Float64,
    field::AuxiliaryField,
    model::ElectronicDispersion,
    interaction::Interaction,
    kgrid::AbstractKGrid,
    ::ExactTrLn;
    T::Float64=1e-3
)
    term1 = _quadratic_action_term(phi, field, interaction, kgrid)

    mf_disp = MeanFieldDispersion(model, field, phi)
    tr_ln_kernel = ExactTrLnContributionKernel(mf_disp, T)
    tr_ln_sum = Engine.integrate_grid(tr_ln_kernel, kgrid)

    return term1 + tr_ln_sum
end

# RPA action evaluation for a single scalar order parameter
function evaluate_action(
    phi::Float64,
    field::AuxiliaryField,
    model::ElectronicDispersion,
    interaction::Interaction,
    kgrid::AbstractKGrid,
    ::RPA;
    T::Float64=1e-3
)
    normal_model = normal_state_basis(model, field)
    chi0 = GeneralizedSusceptibility(normal_model, kgrid, field, T)

    q_vec = _field_wavevector(field, kgrid)
    V_total = V(q_vec, interaction)
    chi_val = chi0(q_vec)

    return (1.0 / abs(V_total) - real(chi_val)) * phi^2
end

function evaluate_action(
    phis::AbstractVector{<:Real},
    field::CompositeField,
    model::ElectronicDispersion,
    interaction::CompositeInteraction,
    kgrid::AbstractKGrid,
    ::ExactTrLn;
    T::Float64=1e-3
)
    length(field) == length(phis) || throw(DimensionMismatch("Number of fields must match number of phis."))
    length(field) == length(interaction) || throw(DimensionMismatch("Number of fields must match number of interactions."))

    term1 = _quadratic_action_term(field.fields, phis, interaction.interactions, kgrid, 1)
    mf_disp = MeanFieldDispersion(model, field, phis)
    tr_ln_kernel = ExactTrLnContributionKernel(mf_disp, T)
    tr_ln_sum = Engine.integrate_grid(tr_ln_kernel, kgrid)

    return term1 + tr_ln_sum
end

function evaluate_action(
    phis::AbstractVector{<:Real},
    field::CompositeField,
    model::ElectronicDispersion,
    interaction::CompositeInteraction,
    kgrid::AbstractKGrid,
    approx::ApproximationLevel;
    T::Float64=1e-3
)
    throw(ArgumentError("CompositeField multivariate action is currently implemented only for ExactTrLn(), got $(typeof(approx))."))
end

"""
    evaluate_action(phi_values::AbstractVector{<:Real}, field, model, interaction, kgrid, approx; T=1e-3)

Evaluates the effective action for a collection of `phi` values. 
Perfect for scanning the free energy landscape and plotting the "Mexican Hat" potential.
"""
function evaluate_action(
    phi_values::AbstractVector{<:Real},
    field::AuxiliaryField,
    model::ElectronicDispersion,
    interaction::Interaction,
    kgrid::AbstractKGrid,
    approx::ApproximationLevel;
    T::Float64=1e-3
)
    # 使用推导式遍历所有的 phi，并强制转换为 Float64 以匹配底层函数签名
    return [evaluate_action(Float64(phi), field, model, interaction, kgrid, approx; T=T) for phi in phi_values]
end
