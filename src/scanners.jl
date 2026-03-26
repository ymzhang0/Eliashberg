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
    # Instantiate the susceptibility functor
    chi_functor = LindhardSusceptibility(model, kgrid, T, η)
    
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

Returns a 2D Matrix of size (length(qpath), length(omegas)).
"""
function scan_spectral_function(model::PhysicalModel, kgrid::KGrid{D}, qpath::KPath{D}, omegas::AbstractVector{Float64}; T=0.001, η=0.05) where {D}
    # Instantiate the susceptibility functor with the desired broadening η
    chi_functor = LindhardSusceptibility(model, kgrid, T, η)
    
    nq = length(qpath)
    nω = length(omegas)
    spectral_matrix = zeros(Float64, nq, nω)
    
    @info "Scanning spectral function over $nq q-points and $nω frequencies..."
    
    for (i, q) in enumerate(qpath.points)
        for (j, ω) in enumerate(omegas)
            # Evaluate the dynamical susceptibility
            fluctuation = DynamicalFluctuation(q, ω)
            val = chi_functor(fluctuation)
            
            # Spectral function is proportional to the imaginary part
            # Standard convention for A(q, ω) = Im[χ(q, ω)]
            spectral_matrix[i, j] = imag(val)
        end
    end
    
    return spectral_matrix
end
