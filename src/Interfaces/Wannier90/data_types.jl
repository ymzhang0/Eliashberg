export Wannier90BandComparison

Base.@kwdef struct Wannier90BandComparison{D}
    reference::BandStructureData{D}
    model::BandStructureData{D}
    shifted_model::BandStructureData{D}
    kpoints_fractional::Vector{SVector{3, Float64}}
    energy_shift::Float64
    rms_error::Float64
    max_error::Float64
    difference::Matrix{Float64}

    function Wannier90BandComparison{D}(
        reference::BandStructureData{D},
        model::BandStructureData{D},
        shifted_model::BandStructureData{D},
        kpoints_fractional::AbstractVector{<:SVector{3, <:Real}},
        energy_shift::Real,
        rms_error::Real,
        max_error::Real,
        difference::AbstractMatrix{<:Real},
    ) where {D}
        size(reference.bands) == size(model.bands) == size(shifted_model.bands) ||
            throw(DimensionMismatch("Compared band matrices must all have the same shape."))
        size(reference.bands) == size(difference) ||
            throw(DimensionMismatch("Difference matrix shape must match the compared band matrices."))
        length(reference.kpath) == length(kpoints_fractional) ||
            throw(DimensionMismatch("The fractional k-point list must match the number of path samples."))

        return new{D}(
            reference,
            model,
            shifted_model,
            [SVector{3, Float64}(point) for point in kpoints_fractional],
            Float64(energy_shift),
            Float64(rms_error),
            Float64(max_error),
            Float64.(difference),
        )
    end
end
Wannier90BandComparison(
    reference::BandStructureData{D},
    model::BandStructureData{D},
    shifted_model::BandStructureData{D},
    kpoints_fractional::AbstractVector{<:SVector{3, <:Real}},
    energy_shift::Real,
    rms_error::Real,
    max_error::Real,
    difference::AbstractMatrix{<:Real},
) where {D} = Wannier90BandComparison{D}(reference, model, shifted_model, kpoints_fractional, energy_shift, rms_error, max_error, difference)
