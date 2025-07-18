# Fermi-Dirac 分布
struct FermiDiracSmearing <: Smearing
    T::Float64     # temperature (in same unit as energy)
end

# Bose-Einstein 分布
struct BoseEinsteinSmearing <: Smearing
    T::Float64
end

# 高斯 smearing（常用于 DOS）
struct GaussianSmearing <: Smearing
    σ::Float64     # Gaussian width
end

function f(ε::Float64, model::FermiDiracSmearing)
    β = 1 / Constants.kB2Ha.val * model.T
    return 1.0 / (exp(β * ε) + 1)
end

function f(ε::Float64, model::BoseEinsteinSmearing)
    β = 1 / Constants.kB2Ha.val * model.T
    return 1.0 / (exp(β * ε) - 1)
end

#  δ(ε - ε₀) ≈ Gaussian
function f(ε::Float64, model::GaussianSmearing)
    return exp(-ε^2 / (2 * model.σ^2)) / (model.σ * sqrt(2π))
end

