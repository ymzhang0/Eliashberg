struct ExactTrLn <: ApproximationLevel end
struct RPA <: ApproximationLevel end

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
    # 1. 获取当前场的波矢
    q_vec = hasproperty(field, :q) ? field.q : zero(first(kgrid))

    # 2. 获取相互作用强度（完美解耦：这里甚至可以传入虚频进行扩展）
    V_total = V(q_vec, interaction)

    term1 = phi^2 / abs(V_total)

    # 3. 构造平均场重构的能带
    mf_disp = MeanFieldDispersion(model, field, phi)

    tr_ln_sum = 0.0
    for i in 1:length(kgrid)
        k = kgrid.points[i]
        w = kgrid.weights[i]

        lambdas = real(band_structure(mf_disp, k).values)

        for lam in lambdas
            if lam < 0
                tr_ln_sum += w * (lam - T * log1p(exp(lam / T)))
            else
                tr_ln_sum += w * (-T * log1p(exp(-lam / T)))
            end
        end
    end

    return term1 + tr_ln_sum
end

# RPA 版本的 action 评估同样清晰
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

    q_vec = hasproperty(field, :q) ? field.q : zero(first(kgrid))
    V_total = V(q_vec, interaction)
    chi_val = chi0(q_vec)

    return (1.0 / abs(V_total) - real(chi_val)) * phi^2
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