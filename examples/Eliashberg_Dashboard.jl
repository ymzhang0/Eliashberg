### A Pluto.jl Notebook ###
# v0.19.40

using Marker: @any

# ╔═╡ 00000000-0000-0000-0000-000000000001
begin
	import Pkg
	# Activate the local environment where Eliashberg.jl is located
	Pkg.activate(joinpath(Pkg.devdir(), "..", "..")) # Adjusting to the project root
    # Note: For a "ready-to-run" script, we assume Eliashberg is available in the load path.
    # In a real Pluto environment, the user would usually `using Eliashberg` after adding it.
end

# ╔═╡ 1e2a3b4c-5d6e-7f8a-9b0c-1d2e3f4g5h6i
begin
	using Eliashberg
	using PlutoUI
	using CairoMakie
	using StaticArrays
	using Optim
	using LinearAlgebra
end

# ╔═╡ 2a3b4c5d-6e7f-8a9b-0c1d-2e3f4g5h6i7j
md"""
# Eliashberg.jl: Reactive Physics Dashboard
### Visualizing Spontaneous Symmetry Breaking & Effective Action

Adjust the sliders below to explore how **Temperature** ($T$) and **Interaction Strength** ($V$) drive the system into a broken symmetry state (Superconductivity or CDW).
"""

# ╔═╡ 3b4c5d6e-7f8a-9b0c-1d2e-3f4g5h6i7j8k
begin
	md"**Temperature (T):** $(@bind T Slider(0.001:0.005:0.1, default=0.05, show_value=true))"
end

# ╔═╡ 4c5d6e7f-8a9b-0c1d-2e3f-4g5h6i7j8k9l
begin
	md"**Interaction Strength (V):** $(@bind V_in Slider(-5.0:0.1:1.0, default=-2.0, show_value=true))"
end

# ╔═╡ 5d6e7f8a-9b0c-1d2e-3f4g-5h6i7j8k9l0m
begin
	# 1. Setup Model
    # Using a 1D Tight Binding model for speed and clarity
	t_hop = 1.0
	mu = 0.5
	model = TightBinding{1}(t_hop, mu)
	field = SuperconductingPairing(:s_wave)
	
	# 2. Grid Setup
	k_pts = [SVector{1,Float64}(k) for k in range(-pi, pi, length=100)]
	weights = fill(2*pi/100, 100)
	grid = KGrid{1}(k_pts, weights)
	
	# 3. Interaction
	interaction = ConstantInteraction(V_in)
	
	# 4. Effective Action Object
	action = EffectiveAction(model, field, grid, interaction)
end

# ╔═╡ 6e7f8a9b-0c1d-2e3f-4g5h-6i7j8k9l0m1n
begin
	phi_vals = range(0.0, 0.5, length=50)
	
	f_exact = evaluate(action, collect(phi_vals), ExactTrLn(); T=T)
	f_rpa = evaluate(action, collect(phi_vals), RPA(); T=T)
	
	# Normalize to F(0) = 0 for better visual comparison
	f_exact .-= f_exact[1]
	f_rpa .-= f_rpa[1]
end

# ╔═╡ 7f8a9b0c-1d2e-3f4g-5h6i-7j8k9l0m1n2o
begin
	# Find the global minimum for "Exact" curve
	min_res = optimize(p -> evaluate(action, p, ExactTrLn(); T=T), 0.0, 0.5)
	phi_min = Optim.minimizer(min_res)
	f_min = Optim.minimum(min_res) - evaluate(action, 0.0, ExactTrLn(); T=T)
end

# ╔═╡ 8a9b0c1d-2e3f-4g5h-6i7j-8k9l0m1n2o3p
begin
	# Plotting with Makie
	set_theme!(theme_dark())
	
	fig = Figure(resolution = (800, 500))
	ax = Axis(fig[1, 1], 
		title = "Effective Action F(ϕ) - Quantum Phase Transition",
		xlabel = "Order Parameter ϕ (Gap Δ)",
		ylabel = "ΔF(ϕ)",
		backgroundcolor = :black
	)
	
	# Plot Curves
	lines!(ax, phi_vals, f_exact, color = :cyan, label = "Exact Tr[ln]", linewidth = 3)
	lines!(ax, phi_vals, f_rpa, color = :magenta, label = "RPA (Quadratic)", linestyle = :dash, linewidth = 2)
	
	# Ground State Tracking
	scatter!(ax, [phi_min], [f_min], color = :yellow, markersize = 12, label = "Ground State")
	
	# Symmetric vs Broken Symmetry Visuals
	if phi_min > 0.01
		text!(ax, phi_min + 0.02, f_min, 
			text = "Spontaneous Symmetry Breaking\n(Mexican Hat Potential)", 
			color = :yellow, align = (:left, :center)
		)
		vlines!(ax, [phi_min], color = (:yellow, 0.3), linestyle = :dot)
	else
		text!(ax, 0.05, 0.05, text = "Symmetric State (ϕ=0)", color = :white)
	end

	# Axis limits
	ylims!(ax, min(minimum(f_exact), minimum(f_rpa)) - 0.1, max(maximum(f_exact), maximum(f_rpa)) + 0.1)
	
	axislegend(ax, position = :rt)
	
	fig
end

# ╔═╡ 9b0c1d2e-3f4g-5h6i-7j8k-9l0m1n2o3p4q
md"""
### Physical Insights:
- **RPA Expansion**: The dashed line represents the harmonic approximation $(1/V - \chi_0)\phi^2$. It only captures the onset of instability when the curvature at $\phi=0$ becomes negative.
- **Exact Tr[ln]**: The solid line includes all quantum fluctuations. It provides the full potential landscape, revealing the non-linear stabilization of the order parameter.
- **Phase Transition**: When $V$ is sufficiently attractive ($V < 0$) and $T$ is low, the minimum shifts away from zero, indicating the system has reached a broken symmetry state.
"""

# ╔═╡ Cell order:
# ╟─2a3b4c5d-6e7f-8a9b-0c1d-2e3f4g5h6i7j
# ╟─3b4c5d6e-7f8a-9b0c-1d2e-3f4g5h6i7j8k
# ╟─4c5d6e7f-8a9b-0c1d-2e-3f4g5h6i7j8k9l
# ╠═8a9b0c1d-2e3f-4g5h-6i7j-8k9l0m1n2o3p
# ╟─9b0c1d2e-3f4g-5h6i-7j8k-9l0m1n2o3p4q
# ╟─00000000-0000-0000-0000-000000000001
# ╟─1e2a3b4c-5d6e-7f8a-9b0c-1d2e3f4g5h6i
# ╟─5d6e7f8a-9b0c-1d2e-3f4g-5h6i7j8k9l0m
# ╟─6e7f8a9b-0c1d-2e3f-4g5h-6i7j8k9l0m1n
# ╟─7f8a9b0c-1d2e-3f4g-5h6i-7j8k9l0m1n2o
