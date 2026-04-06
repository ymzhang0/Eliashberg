
"""
    compute_dispersion_surface_data(disp::Dispersion, kgrid::KGrid{2})

Compute a two-dimensional scalar field over a rectangular parameter grid.
"""
function compute_dispersion_surface_data(disp::Dispersion, kgrid::KGrid{2})
    kxs = unique(sort([k[1] for k in kgrid.points]))
    kys = unique(sort([k[2] for k in kgrid.points]))
    ix = Dict(kx => idx for (idx, kx) in enumerate(kxs))
    iy = Dict(ky => idx for (idx, ky) in enumerate(kys))
    energy_matrix = zeros(Float64, length(kxs), length(kys))

    for k in kgrid.points
        values = real(band_structure(disp, k).values)
        energy_matrix[ix[k[1]], iy[k[2]]] = values[1]
    end

    return DispersionSurfaceData(kxs, kys, energy_matrix)
end

"""
    compute_band_data(disp::ElectronicDispersion, kpath::KPath{D}) where {D}

Compute band values along a path in parameter space. The returned dense matrix
has one column per band and one row per path sample.
"""
function compute_band_data(disp::ElectronicDispersion, kpath::KPath{D}) where {D}
    bands = [real(band_structure(disp, k).values) for k in kpath.points]
    num_bands = length(bands[1])
    band_matrix = fill(NaN, length(kpath.points), num_bands)

    for band_idx in 1:num_bands
        band_matrix[:, band_idx] = [values[band_idx] for values in bands]
    end

    return BandStructureData(
        kpath=kpath,
        bands=band_matrix,
        num_bands=num_bands
    )
end

"""
    compute_fermi_surface_volume(disp::ElectronicDispersion; n_pts=100)

Compute a dense scalar field over a cubic sampling box for isosurface rendering.
"""
function compute_fermi_surface_volume(disp::ElectronicDispersion; n_pts::Integer=100)
    kxs = range(-π, π, length=n_pts)
    kys = range(-π, π, length=n_pts)
    kzs = range(-π, π, length=n_pts)
    energy_volume = zeros(Float32, length(kxs), length(kys), length(kzs))

    for (i, kx) in enumerate(kxs)
        for (j, ky) in enumerate(kys)
            for (k, kz) in enumerate(kzs)
                values = real(band_structure(disp, SVector{3,Float64}(kx, ky, kz)).values)
                energy_volume[i, j, k] = Float32(values[1])
            end
        end
    end

    return FermiSurfaceData(kxs, kys, kzs, energy_volume)
end

"""
    compute_landscape_line_data(qgrid::KGrid{1}, landscape_vector::AbstractVector{<:Real})

Align a one-dimensional sampled field with a sorted parameter axis for plotting.
"""
function compute_landscape_line_data(qgrid::KGrid{1}, landscape_vector::AbstractVector{<:Real})
    length(qgrid) == length(landscape_vector) || throw(DimensionMismatch("Landscape vector length must match the grid length."))
    qs = [q[1] for q in qgrid.points]
    perm = sortperm(qs)
    return LandscapeLineData(qs[perm], landscape_vector[perm])
end

"""
    compute_landscape_axes(qgrid::KGrid{2})

Extract the rectangular parameter axes corresponding to a two-dimensional scan.
"""
function compute_landscape_axes(qgrid::KGrid{2})
    qxs = unique(sort([q[1] for q in qgrid.points]))
    qys = unique(sort([q[2] for q in qgrid.points]))
    return (; qxs, qys)
end

"""
    compute_landscape_surface_data(qgrid::KGrid{2}, landscape_matrix::AbstractMatrix{<:Real})

Wrap a two-dimensional instability landscape together with its rectangular axes
into a typed object suitable for generic plotting.
"""
function compute_landscape_surface_data(qgrid::KGrid{2}, landscape_matrix::AbstractMatrix{<:Real})
    axes = compute_landscape_axes(qgrid)
    return LandscapeSurfaceData(axes.qxs, axes.qys, landscape_matrix)
end

"""
    compute_phase_transition_data(phis, Ts, field, model, interaction, kgrid; approx=ExactTrLn(), phi_guess=0.2)

Map temperature samples to free-energy curves and reduce each point through the
effective-action solver. Returns pure arrays suitable for plotting.
"""
function compute_phase_transition_data(
    phis::AbstractVector{<:Real},
    Ts::AbstractVector{<:Real},
    field::AuxiliaryField,
    model::ElectronicDispersion,
    interaction::Interaction,
    kgrid::AbstractKGrid;
    approx::ApproximationLevel=ExactTrLn(),
    phi_guess::Real=0.2
)
    free_energy = zeros(Float64, length(phis), length(Ts))
    condensation_energy = zeros(Float64, length(phis), length(Ts))
    order_parameters = zeros(Float64, length(Ts))
    current_guess = Float64(phi_guess)

    for (idx, T) in enumerate(Ts)
        energy_curve = Float64.(evaluate_action(phis, field, model, interaction, kgrid, approx; T=T))
        free_energy[:, idx] = energy_curve
        condensation_energy[:, idx] = energy_curve .- energy_curve[1]

        phi_gs = solve_ground_state(field, model, interaction, kgrid, approx; phi_guess=current_guess, T=T)
        phi_gs = phi_gs < 1e-4 ? 0.0 : phi_gs
        order_parameters[idx] = phi_gs
        current_guess = phi_gs > 0.0 ? phi_gs : max(Float64(phi_guess), 0.05)
    end

    return PhaseDiagramData(
        phis=Float64.(phis),
        Ts=Float64.(Ts),
        free_energy=free_energy,
        condensation_energy=condensation_energy,
        order_parameters=order_parameters
    )
end

"""
    compute_renormalized_band_data(Ts, field, model, interaction, kgrid, kpath; approx=ExactTrLn(), phi_guess=0.5)

Compute the mean-field gap and the corresponding renormalized bands for each
temperature sample along a path in parameter space.
"""
function compute_renormalized_band_data(
    Ts::AbstractVector{<:Real},
    field::AuxiliaryField,
    model::ElectronicDispersion,
    interaction::Interaction,
    kgrid::AbstractKGrid,
    kpath::KPath;
    approx::ApproximationLevel=ExactTrLn(),
    phi_guess::Real=0.5
)
    gaps = zeros(Float64, length(Ts))
    current_guess = Float64(phi_guess)

    # 1. 核心映射：对每一个温度 T，计算并返回一个 (N_kpoints, N_bands) 的矩阵
    band_matrices = map(enumerate(Ts)) do (idx, T)
        # 求解基态
        phi_gs = solve_ground_state(field, model, interaction, kgrid, approx; phi_guess=current_guess, T=T)
        phi_gs = phi_gs < 1e-4 ? 0.0 : phi_gs
        gaps[idx] = phi_gs
        current_guess = phi_gs > 0.0 ? phi_gs : max(Float64(phi_guess), 0.05)

        # 构造真实的重整化色散
        renormalized_dispersion = MeanFieldDispersion(model, field, phi_gs)
        
        # 计算该温度下路径上的所有本征值
        # bands_k 是一个 Vector{SVector{N, Float64}}，N 会根据 field 自动变成 1, 2, 或 3
        bands_k = [real(band_structure(renormalized_dispersion, k).values) for k in kpath.points]
        
        # 将 Vector{SVector} 沿着第一维度堆叠成 Matrix
        return stack(bands_k, dims=1) 
    end

    # 2. 自动升维：band_matrices 是一个包含多个 Matrix 的 Vector
    # 再次使用 stack，直接将其变成一个 (N_kpoints, N_bands, N_Ts) 的 3D 数组！
    renormalized_bands = stack(band_matrices)

    # 3. 计算 Bare Bands (用于参考)
    bare_bands_k = [real(band_structure(model, k).values) for k in kpath.points]
    bare_bands = stack(bare_bands_k, dims=1)

    return RenormalizedBandData(
        kpath=kpath,
        bare_bands=Float64.(bare_bands),
        renormalized_bands=Float64.(renormalized_bands),
        gaps=gaps,
        temperatures=Float64.(Ts)
    )
end

"""
    compute_zeeman_pairing_data(T_val, h_val, q_vals, model, interaction, kgrid; approx=ExactTrLn(), phi_guess=0.4)

Map a one-dimensional external parameter axis to optimized order parameters and
condensation energies for FFLO-style scans.
"""
function compute_zeeman_pairing_data(
    T_val::Real,
    h_val::Real,
    q_vals::AbstractVector{<:Real},
    model::ElectronicDispersion,
    interaction::Interaction,
    kgrid::AbstractKGrid;
    approx::ApproximationLevel=ExactTrLn(),
    phi_guess::Real=0.4
)
    dim = length(first(kgrid.points))
    minimal_energy = zeros(Float64, length(q_vals))
    optimal_gaps = zeros(Float64, length(q_vals))
    current_guess = Float64(phi_guess)

    for (idx, q) in enumerate(q_vals)
        q_vector = SVector{dim,Float64}(ntuple(i -> i == 1 ? Float64(q) : 0.0, dim))
        fflo_field = FFLOPairing(q_vector, h_val)
        phi_gs = solve_ground_state(fflo_field, model, interaction, kgrid, approx; phi_guess=current_guess, T=T_val)
        phi_gs = phi_gs < 1e-4 ? 0.0 : phi_gs

        optimal_gaps[idx] = phi_gs
        minimal_energy[idx] = evaluate_action(phi_gs, fflo_field, model, interaction, kgrid, approx; T=T_val)
        current_guess = phi_gs > 0.05 ? phi_gs : max(Float64(phi_guess), 0.05)
    end

    zero_q = zero(SVector{dim,Float64})
    normal_energy = evaluate_action(0.0, FFLOPairing(zero_q, h_val), model, interaction, kgrid, approx; T=T_val)
    condensation_energy = minimal_energy .- normal_energy
    minimum_index = argmin(condensation_energy)

    return ZeemanPairingData(
        q_vals,
        condensation_energy,
        optimal_gaps,
        q_vals[minimum_index],
        minimum_index
    )
end

"""
    compute_collective_mode_spectral_data(T_val, field, model, interaction, kgrid, qpath; omega_max_factor=5.0, n_omegas=100, eta=0.02, approx=ExactTrLn(), phi_guess=0.4, kwargs...)

Compute the dynamical spectral map of a mean-field state and return only pure
array data for downstream visualization.
"""
function compute_collective_mode_spectral_data(
    T_val::Real,
    field::AuxiliaryField,
    model::ElectronicDispersion,
    interaction::Interaction,
    kgrid::AbstractKGrid,
    qpath::KPath;
    omega_max_factor::Real=5.0,
    n_omegas::Integer=100,
    eta::Real=0.02,
    approx::ApproximationLevel=ExactTrLn(),
    phi_guess::Real=0.4,
    bootstrap_workers::Bool=false,
    n_workers::Integer=max(0, Threads.nthreads() - 1),
    project::Union{Nothing,AbstractString}=Base.active_project(),
    restrict::Bool=true
)
    phi_gs = solve_ground_state(field, model, interaction, kgrid, approx; phi_guess=Float64(phi_guess), T=T_val)
    phi_gs = phi_gs < 1e-4 ? 0.0 : phi_gs

    bdg_dispersion = MeanFieldDispersion(model, field, phi_gs)
    omega_max = phi_gs > 0.0 ? omega_max_factor * phi_gs : 2.0
    omegas = collect(range(0.0, omega_max, length=n_omegas))
    spectral_matrix = scan_rpa_spectral_function_hpc(
        bdg_dispersion,
        interaction,
        field,
        kgrid,
        qpath.points,
        omegas;
        T=T_val,
        η=eta,
        bootstrap_workers=bootstrap_workers,
        n_workers=n_workers,
        project=project,
        restrict=restrict
    )

    pair_breaking_edge = phi_gs > 0.0 ? 2.0 * phi_gs : nothing
    return SpectralMapData(
        qpath=qpath,
        omegas=omegas,
        spectral_matrix=spectral_matrix,
        gap=phi_gs,
        pair_breaking_edge=pair_breaking_edge,
        temperature=Float64(T_val)
    )
end
