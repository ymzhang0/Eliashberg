# scanners.jl
using StaticArrays

"""
    scan_instability_landscape(model::PhysicalModel, kgrid::KGrid{D}, qgrid::KGrid{D}; T=0.001, η=1e-3) where {D}

Evaluates the static susceptibility (ω=0) over an entire momentum grid `qgrid` to identify
the most unstable nesting wavevectors. 

Internal momentum integration is performed over `kgrid`.
Returns an array of susceptibility values corresponding to the points in `qgrid`.
"""
function scan_instability_landscape(model::PhysicalModel, kgrid::KGrid{D}, qgrid::KGrid{D}; T=0.001, η=1e-3) where {D}
    # Instantiate the susceptibility functor with a default density-like field
    chi_functor = GeneralizedSusceptibility(model, kgrid, ChargeDensityWave(zero(SVector{D,Float64})), T, η)
    
    # Pre-allocate the landscape array
    # We use real part as susceptibility is typically defined as the real response
    landscape = zeros(Float64, length(qgrid))
    
    @info "Scanning instability landscape over $(length(qgrid)) q-points..."
    
    for (i, q) in enumerate(qgrid.points)
        # Construct a DynamicalFluctuation with zero frequency for static instability analysis
        fluctuation = DynamicalFluctuation(q, 0.0)
        
        # Evaluate the susceptibility
        val = chi_functor(fluctuation)
        
        # For instability, we are interested in the magnitude/real part of the static response
        landscape[i] = real(val)
    end
    
    return landscape
end

"""
    scan_spectral_function(model::PhysicalModel, kgrid::KGrid{D}, qpath::KPath{D}, omegas::AbstractVector{Float64}; T=0.001, η=0.05) where {D}

Iterates over momentum `q` in `qpath` and frequency `ω` in `omegas` to compute the
dynamical spectral function A(q, ω) = Im[χ(q, ω)].

This version implements extreme algorithmic optimizations:
1. Multi-threading over the outermost `q` loop.
2. Global hoisting of `ε(k)` and `f(ε(k))` computations outside all loops.
3. Local hoisting of `ε(k+q)` and `f(ε(k+q))` inside the `q` loop but outside the `ω` loop.
4. Complete elimination of complex arithmetic in the innermost `k` loop using mathematical identities.
5. In-place memory writing and tight loop array bounds bypassing (`@inbounds`).
"""
function scan_spectral_function(model::PhysicalModel, kgrid::KGrid{D}, qpath::KPath{D}, omegas::AbstractVector{Float64}; T=0.001, η=0.05) where {D}
    nq = length(qpath)
    nω = length(omegas)
    Nk = length(kgrid)
    
    # The output matrix is pre-allocated. 
    # Thread safety: Each thread will uniquely write to a distinct row `i` of the matrix, 
    # completely avoiding data races and eliminating the need for atomic locks.
    spectral_matrix = zeros(Float64, nq, nω)
    
    @info "Scanning spectral function over $nq q-points and $nω frequencies with extreme optimization..."
    
    # --- 1. Global Hoisting ---
    # We precompute the band energies and Fermi-Dirac distributions for the 
    # entire unshifted grid `k` once. This completely eliminates redundant 
    # calls to `ε(k, model)` and temporary matrix allocations (Hermitian) across both loops.
    eks = Vector{Float64}(undef, Nk)
    fks = Vector{Float64}(undef, Nk)
    for k_idx in 1:Nk
        k = kgrid.points[k_idx]
        ek = real(ε(k, model)[1,1])
        eks[k_idx] = ek
        fks[k_idx] = 1.0 / (exp(ek / T) + 1.0)
    end
    
    # --- 2. Mandatory Multi-threading ---
    # Parallelizing the outermost loop over `q` ensures maximum work is distributed 
    # per thread while preserving independent memory writing paths.
    Threads.@threads for i in 1:nq
        q = qpath.points[i]
        
        # --- 3. q-dependent Local Hoisting ---
        # Inside the threaded `q` loop, pre-calculate the shifted energies `ε(k+q)`
        # and their Fermi distributions. 
        # Minimizing allocations: Instead of creating thousands of arrays inside the 
        # innermost loop, we only allocate these two 1D lightweight vectors per `q` iteration.
        ek_qs = Vector{Float64}(undef, Nk)
        fk_qs = Vector{Float64}(undef, Nk)
        
        for k_idx in 1:Nk
            k = kgrid.points[k_idx]
            
            # Evaluate shifted energy once for all frequencies
            ek_q = real(ε(k + q, model)[1,1])
            ek_qs[k_idx] = ek_q
            fk_qs[k_idx] = 1.0 / (exp(ek_q / T) + 1.0)
        end
        
        for j in 1:nω
            ω = omegas[j]
            sum_im = 0.0
            
            # --- 4 & 5. Innermost Loop Optimization & Annotations ---
            # Using `@inbounds` and `@simd` disables bounds checking and allows 
            # SIMD auto-vectorization of the highly predictable memory accesses.
            @inbounds @simd for k_idx in 1:Nk
                w = kgrid.weights[k_idx]
                
                # Fetch hoisted scalar values
                ek = eks[k_idx]
                ek_q = ek_qs[k_idx]
                fk = fks[k_idx]
                fk_q = fk_qs[k_idx]
                
                # Math Identity: We completely replace complex division and structs 
                # (which cause enormous overhead) with pure real arithmetic:
                # Im[1 / ((ε_{k+q} - ε_k) - ω - iη)] = η / (((ε_{k+q} - ε_k) - ω)^2 + η^2)
                diff = (ek_q - ek) - ω
                
                sum_im += w * (fk - fk_q) * η / (diff^2 + η^2)
            end
            
            # Safe write: Thread `i` exclusively owns row `i` in `spectral_matrix`.
            spectral_matrix[i, j] = sum_im
        end
    end
    
    return spectral_matrix
end
