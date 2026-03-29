### A Pluto.jl Notebook ###
# v0.19.40

using Marker: @any

# ╔═╡ 00000000-0000-0000-0000-000000000001
begin
	import Pkg
    # Ensure necessary packages are available
	Pkg.activate(mktempdir())
	Pkg.add([
		Pkg.PackageSpec(name="WGLMakie"),
		Pkg.PackageSpec(name="PlutoUI"),
		Pkg.PackageSpec(name="StaticArrays"),
		Pkg.PackageSpec(name="LinearAlgebra"),
		Pkg.PackageSpec(name="Statistics")
	])
end

# ╔═╡ 11111111-1111-1111-1111-111111111111
begin
	using WGLMakie
	using PlutoUI
	using StaticArrays
	using LinearAlgebra
	using Statistics
end

# ╔═╡ 22222222-2222-2222-2222-222222222222
md"""
# Ginzburg-Landau Physics Explorer
### Amplitude, Phase, and Topological Defects

This notebook contains two interactive tools to explore the Ginzburg-Landau (GL) effective action and its dynamics.
"""

# ╔═╡ 33333333-3333-3333-3333-333333333333
md"""
## Task 1: 1D Ginzburg-Landau Action Simulator
Explores how amplitude wiggles (Higgs mode) and phase twists (Goldstone/Supercurrent) increase the free energy.
"""

# ╔═╡ 44444444-4444-4444-4444-444444444444
begin
	md"**Base Amplitude $R_0$:** $(@bind R0 Slider(0.1:0.1:2.0, default=1.0, show_value=true))"
end

# ╔═╡ 55555555-5555-5555-5555-555555555555
begin
	md"**Amplitude Wiggle $\Delta R$:** $(@bind dR Slider(0.0:0.05:1.0, default=0.2, show_value=true))"
end

# ╔═╡ 66666666-6666-6666-6666-666666666666
begin
	md"**Phase Twist $\nabla\theta$ (Supercurrent):** $(@bind grad_theta Slider(0.0:0.1:5.0, default=1.0, show_value=true))"
end

# ╔═╡ 77777777-7777-7777-7777-777777777777
begin
	# 1D Simulation Constants
	alpha = -1.0
	beta = 0.5
	gamma = 1.0
	L = 2π
	Nx = 200
	x_range = range(0, L, length=Nx)
	dx = step(x_range)
	
	# Field Construction
	# ψ(x) = (R_0 + ΔR*sin(x)) * exp(i * grad_theta * x)
	R_x = @. R0 + dR * sin(x_range)
	theta_x = @. grad_theta * x_range
	psi_x = @. R_x * exp(im * theta_x)
	
	# Numerical Gradients
	dR_dx = [0.0; diff(R_x) / dx]
	dtheta_dx = [0.0; fill(grad_theta, Nx-1)]
	
	# Action Components
	S_pot = sum(@. (alpha * R_x^2 + beta * R_x^4)) * dx
	S_grad_R = sum(@. gamma * dR_dx^2) * dx
	S_grad_theta = sum(@. gamma * R_x^2 * dtheta_dx^2) * dx
	
	action_vals = [S_pot, S_grad_R, S_grad_theta]
	labels = ["Potential", "Amp Gradient", "Phase Gradient"]
end

# ╔═╡ 88888888-8888-8888-8888-888888888888
begin
	# Visualization for Task 1
	fig1 = Figure(resolution = (800, 400))
	
	ax1 = Axis(fig1[1, 1], title="Field Profile", xlabel="x", ylabel="Value")
	lines!(ax1, x_range, R_x, label="Amplitude R(x)", color=:blue, linewidth=3)
	lines!(ax1, x_range, @.(theta_x % (2π)), label="Phase θ(x) mod 2π", color=:red, linestyle=:dash)
	axislegend(ax1)
	
	ax2 = Axis(fig1[1, 2], title="Action Components", ylabel="Action Value", xticks = (1:3, labels))
	barplot!(ax2, 1:3, action_vals, color=[:green, :blue, :red])
	
	fig1
end

# ╔═╡ 99999999-9999-9999-9999-999999999999
md"""
---
## Task 2: 2D Complex Field Sandbox (TDGL Dynamics)
Interactive 2D relaxation of a complex scalar field. Use the buttons to perturb the system and watch it evolve towards the Ginzburg-Landau ground state.
"""

# ╔═╡ aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa
begin 
	# TDGL Parameters
	const N2 = 30
	const dt = 0.05
	const α2 = -1.0
	const β2 = 0.5
	const γ2 = 1.0
	
	# Observables for Interactivity
	psi_obs = Observable(ones(ComplexF64, N2, N2))
	relaxation_running = Observable(false)
end

# ╔═╡ bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb
begin
	# 2D Field Buttons
	md"""
	$(@bind btn_noise Button("Add Noise"))
	$(@bind btn_current Button("Impose Supercurrent"))
	$(@bind btn_vortex Button("Add Vortex"))
	$(@bind btn_relax Toggle(false, label="Relaxing..."))
	"""
end

# ╔═╡ cccccccc-cccc-cccc-cccc-cccccccccccc
begin
	# Handle Button Triggers
	btn_noise
	let 
		new_psi = psi_obs[] .+ 0.2 .* (randn(ComplexF64, N2, N2))
		psi_obs[] = new_psi
	end
end

# ╔═╡ dddddddd-dddd-dddd-dddd-dddddddddddd
begin
	btn_current
	let
		curr_psi = psi_obs[]
		for i in 1:N2, j in 1:N2
			curr_psi[i, j] *= exp(im * 0.3 * i)
		end
		psi_obs[] = curr_psi
	end
end

# ╔═╡ eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee
begin
	btn_vortex
	let
		curr_psi = psi_obs[]
		center = N2 / 2
		for i in 1:N2, j in 1:N2
			dx = i - center
			dy = j - center
			r = sqrt(dx^2 + dy^2)
			phi = atan(dy, dx)
			curr_psi[i, j] = (1.0 - exp(-r/2.0)) * exp(im * phi)
		end
		psi_obs[] = curr_psi
	end
end

# ╔═╡ ffffffff-ffff-ffff-ffff-ffffffffffff
begin
	# TDGL Relaxation Step
	function tdgl_step!(psi)
		laplacian = zeros(ComplexF64, N2, N2)
		for i in 1:N2, j in 1:N2
			# Periodic BC or fallback
			im1 = i == 1 ? N2 : i - 1
			ip1 = i == N2 ? 1 : i + 1
			jm1 = j == 1 ? N2 : j - 1
			jp1 = j == N2 ? 1 : j + 1
			
			laplacian[i, j] = psi[ip1, j] + psi[im1, j] + psi[i, jp1] + psi[i, jm1] - 4*psi[i,j]
		end
		
		# Update Rule: ∂t ψ = -(αψ + 2β|ψ|^2ψ - γ∇^2ψ)
		@. psi -= dt * (α2 * psi + 2 * β2 * abs2(psi) * psi - γ2 * laplacian)
	end

	# Continuous Relaxation Loop
	if btn_relax
		while btn_relax
			curr = copy(psi_obs[])
			tdgl_step!(curr)
			psi_obs[] = curr
			sleep(0.01) # Yield to UI
		end
	end
end

# ╔═╡ 10101010-1010-1010-1010-101010101010
begin
	# Calculate Actions for Gauges
	total_action = lift(psi_obs) do p
		sum(@. α2 * abs2(p) + β2 * abs2(p)^2)
	end
	
	md"**Current Potential Action:** $(total_action)"
end

# ╔═╡ 20202020-2020-2020-2020-202020202020
begin
	# 2D Visualization
	X = 1:N2
	Y = 1:N2
	
	# Derived Observables for Arrow plot
	pts = [Point2f(x, y) for x in X for y in Y]
	dirs = lift(psi_obs) do p
		[Vec2f(real(v), imag(v)) for v in p]
	end
	
	colors = lift(psi_obs) do p
		[angle(v) for v in p]
	end
	
	fig2 = Figure(resolution = (600, 600))
	ax2d = Axis(fig2[1, 1], title="2D Complex Field (Quiver Plot)", aspect=DataAspect())
	
	arrows!(ax2d, pts, dirs, 
		arrowsize = 7, lengthscale = 0.8,
		color = colors, colormap = :hsv, colorrange = (-pi, pi)
	)
	
	fig2
end

# ╔═╡ Cell order:
# ╟─22222222-2222-2222-2222-222222222222
# ╟─33333333-3333-3333-3333-333333333333
# ╟─44444444-4444-4444-4444-444444444444
# ╟─55555555-5555-5555-5555-555555555555
# ╟─66666666-6666-6666-6666-666666666666
# ╠═88888888-8888-8888-8888-888888888888
# ╟─99999999-9999-9999-9999-999999999999
# ╟─bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb
# ╠═20202020-2020-2020-2020-202020202020
# ╟─10101010-1010-1010-1010-101010101010
# ╟─00000000-0000-0000-0000-000000000001
# ╟─11111111-1111-1111-1111-111111111111
# ╟─77777777-7777-7777-7777-777777777777
# ╟─aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa
# ╟─cccccccc-cccc-cccc-cccc-cccccccccccc
# ╟─dddddddd-dddd-dddd-dddd-dddddddddddd
# ╟─eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee
# ╟─ffffffff-ffff-ffff-ffff-ffffffffffff
