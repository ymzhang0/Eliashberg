using Pkg
Pkg.activate(".")
using Eliashberg
using StaticArrays

println("Testing KPath generation and visualization...")

try
    # Define high-symmetry points for a simple cubic lattice
    gamma = SVector{3, Float64}(0, 0, 0)
    x_point = SVector{3, Float64}(pi, 0, 0)
    m_point = SVector{3, Float64}(pi, pi, 0)
    
    nodes = [gamma, x_point, m_point, gamma]
    labels = ["Γ", "X", "M", "Γ"]
    
    kpath = generate_kpath(nodes, labels; n_pts_per_segment=10)
    println("KPath generated: length $(length(kpath.points))")
    
    model = TightBinding{3}(1.0, 0.0, 0.0) # 3D Tight Binding
    
    # Test visualization dispatch
    # Note: We can't actually see the plot in this environment, 
    # but we can check if it runs without error.
    fig = visualize_dispersion(model, kpath)
    println("Visualization figure created successfully.")
    
    println("SUCCESS: KPath generation and visualization are available and working.")
catch e
    println("FAILURE: KPath utilities missing or error: $e")
    rethrow(e)
end
