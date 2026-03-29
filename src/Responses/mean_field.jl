

# Generic constructor to ensure phi is Float64
function MeanFieldDispersion(bare::M, field::F, phi::Real) where {D,M<:ElectronicDispersion{D},F<:AuxiliaryField}
    return MeanFieldDispersion{D,M,F}(bare, field, Float64(phi))
end


function MeanFieldDispersion(bare::M, field::ChargeDensityWave{D}, phi::Float64) where {D,M<:ElectronicDispersion{D}}
    return MeanFieldDispersion{D,M,typeof(field)}(bare, field, phi)
end

function MeanFieldDispersion(bare::M, field::BCSReducedPairing, phi::Float64) where {D,M<:ElectronicDispersion{D}}
    return MeanFieldDispersion{D,M,BCSReducedPairing}(bare, field, phi)
end

function MeanFieldDispersion(bare::M, field::FFLOPairing{D}, phi::Float64) where {D,M<:ElectronicDispersion{D}}
    return MeanFieldDispersion{D,M,FFLOPairing{D}}(bare, field, phi)
end

function MeanFieldDispersion(bare::M, field::PairDensityWave{D}, phi::Float64) where {D,M<:ElectronicDispersion{D}}
    return MeanFieldDispersion{D,M,PairDensityWave{D}}(bare, field, phi)
end

# ----------------------------------------------------------------------------
# Matrix Elements (ε) - The Core Physics
# ----------------------------------------------------------------------------

# 1. Normal State Nambu (For standard BCS susceptibility)
function ε(k::SVector{D,Float64}, model::NormalNambuDispersion) where {D}
    ek = real(ε(k, model.bare)[1, 1])
    e_minus_k = real(ε(-k, model.bare)[1, 1])
    return Hermitian(@SMatrix [ek 0.0; 0.0 -e_minus_k])
end

# 2. Charge Density Wave (k coupled to k+q)
function ε(k::SVector{D,Float64}, model::MeanFieldDispersion{D,M,<:ChargeDensityWave{D}}) where {D,M}
    H11 = real(ε(k, model.bare_dispersion)[1, 1])
    H22 = real(ε(k + model.field.q, model.bare_dispersion)[1, 1])
    H12 = model.phi
    return Hermitian(@SMatrix [H11 H12; H12 H22])
end

# 3. Standard BCS Reduced Pairing (k coupled to -k)
function ε(k::SVector{D,Float64}, model::MeanFieldDispersion{D,M,BCSReducedPairing}) where {D,M}
    ek = real(ε(k, model.bare_dispersion)[1, 1])
    e_minus_k = real(ε(-k, model.bare_dispersion)[1, 1])
    Δ_k = model.phi * gap_form_factor(k, model.field)
    return Hermitian(@SMatrix [ek Δ_k; Δ_k -e_minus_k])
end

# 4. FFLO Pairing (Fulde-Ferrell: k coupled to -k+q)
function ε(k::SVector{D,Float64}, model::MeanFieldDispersion{D,M,<:FFLOPairing{D}}) where {D,M}
    # 粒子支：自旋向上 (能量减去 h)
    ek_up = real(ε(k, model.bare_dispersion)[1, 1]) - model.field.h

    # 空穴支：来自自旋向下的费米子 (-k+q)
    # 原本的空穴能量是 -ε(-k+q)。因为是自旋向下，本身能量要加上 h。
    # 所以空穴支的有效对角元是 -(ε(-k+q) + h) = -ε(-k+q) - h
    e_hole_down = real(ε(-k + model.field.q, model.bare_dispersion)[1, 1])
    minus_e_hole_down = -e_hole_down - model.field.h

    rel_k = k - model.field.q / 2.0
    Δ_k = model.phi * gap_form_factor(rel_k, model.field)

    return Hermitian(@SMatrix [ek_up Δ_k; Δ_k minus_e_hole_down])
end

# 5. Pair Density Wave (k coupled to BOTH -k+q AND -k-q)
# Requires a 3x3 extended Nambu spinor: (c_k, c_{-k+q}^\dagger, c_{-k-q}^\dagger)^T
function ε(k::SVector{D,Float64}, model::MeanFieldDispersion{D,M,<:PairDensityWave{D}}) where {D,M}
    ek = real(ε(k, model.bare_dispersion)[1, 1])
    e_hole_plus = real(ε(-k + model.field.q, model.bare_dispersion)[1, 1])
    e_hole_minus = real(ε(-k - model.field.q, model.bare_dispersion)[1, 1])

    rel_k_plus = k - model.field.q / 2.0
    rel_k_minus = k + model.field.q / 2.0

    # In a simple standing wave PDW, the amplitude splits evenly
    Δ_plus = (model.phi / 2.0) * gap_form_factor(rel_k_plus, model.field)
    Δ_minus = (model.phi / 2.0) * gap_form_factor(rel_k_minus, model.field)

    return Hermitian(@SMatrix [
        ek Δ_plus Δ_minus;
        Δ_plus -e_hole_plus 0.0;
        Δ_minus 0.0 -e_hole_minus
    ])
end

# ----------------------------------------------------------------------------
# Gap Form Factors
# ----------------------------------------------------------------------------

function gap_form_factor(k::SVector{2,Float64}, field::AuxiliaryField)
    if field.symmetry == :s_wave
        return 1.0
    elseif field.symmetry == :d_wave
        return cos(k[1]) - cos(k[2])
    elseif field.symmetry == :p_wave
        # Just an example of extensibility
        return sin(k[1]) + im * sin(k[2])
    else
        error("Unsupported pairing symmetry: $(field.symmetry)")
    end
end

gap_form_factor(k::SVector, field::AuxiliaryField) = 1.0