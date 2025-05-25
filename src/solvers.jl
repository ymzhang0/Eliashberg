
function renormalized_dispersion(disp::ElectronicDispersion, Σ::SelfEnergy, k, ω)
    ε0 = disp(k)           # 
    Σval = Σ(k, ω)         # 
    return ε0 .+ Σval      # 
end


"""
    solve_bcs(kgrid, dispersion_model, interaction_model)

"""
function solve_bcs(
    kgrid::Vector{Float64},
    dispersion_model::ElectronicDispersion,
    interaction_model::Interaction,
)
    N = length(kgrid)
    H = zeros(N, N)
    for i in 1:N
        for j in 1:N
            H[i, j] += (i == j ? ε(kgrid[i], dispersion_model) : 0.0)
            H[i, j] += V(kgrid[i], kgrid[j], interaction_model, dispersion_model)
        end
    end
    eig = eigen(H)
    return eig.values, eig.vectors
end