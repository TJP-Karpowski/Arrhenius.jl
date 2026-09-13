"""
    mix_trans(gas::A, P, T, X, mean_MW) where {A <: Arrhenius.Solution}

Compute mixture-averaged transport properties for an ideal-gas mixture.

Inputs:
- `P`: pressure in Pa.
- `T`: temperature in K.
- `X`: species mole fractions ordered like `gas.species_names`.
- `mean_MW`: mixture molecular weight in kg/kmol, equivalent to g/mol.

Returns `(η_mix, λ_mix, Dkm)`:
- `η_mix`: mixture viscosity in Pa*s.
- `λ_mix`: mixture thermal conductivity in W/(m*K).
- `Dkm`: mixture-averaged species diffusion coefficients in m^2/s.

The pure-species viscosity, pure-species thermal conductivity, and binary
diffusion coefficients are evaluated from the log-temperature polynomial
coefficients stored in `gas.trans`. Those coefficients are generated from the
YAML species transport data when `CreateSolution` is called.

The mixture viscosity uses the Wilke mixture rule, the conductivity uses the
common mixture-averaged arithmetic/harmonic mean approximation, and `Dkm` uses
the mixture-averaged diffusion expression for mass-averaged diffusive fluxes
driven by mole-fraction gradients.

References:
- R. J. Kee, M. E. Coltrin, and P. Glarborg, *Chemically Reacting Flow*,
  Wiley, 2003, mixture-averaged transport formulas.
- P. D. Neufeld, A. R. Janzen, and R. A. Aziz, "Empirical Equations to
  Calculate 16 of the Transport Collision Integrals Ω(l,s)* for the
  Lennard-Jones (12-6) Potential", *Journal of Chemical Physics*, 1972,
  collision-integral correlations used when constructing `gas.trans`.
"""
function mix_trans(gas::A, P, T, X, mean_MW) where {A <: Arrhenius.Solution}

    logT = log(T)
    trans_T = [logT^6, logT^5, logT^4, logT^3, logT^2, logT, 1.0]
    
    ## species_viscosities_poly

    η = trans_T' * gas.trans.species_viscosities_poly

    Wk_over_Wj = gas.MW * (1.0 ./ gas.MW')

    Wj_over_Wk = 1 ./ Wk_over_Wj

    ηk_over_ηj = η' * (1.0 ./ η)

    # ηk_over_ηj can go slightly negative -- η is a fitted log(T) polynomial,
    # not a value with a guaranteed-positive functional form, and can dip
    # below zero just outside its well-fit range (confirmed directly: a
    # BFER flame Jacobian assembly, which probes T at ForwardDiff.Dual-
    # perturbed values around each cell's real temperature, hit
    # `sqrt(-0.05...)` here). Physically, viscosity is never negative, so
    # flooring with `positive_rate_argument` (same convention used for the
    # same reason throughout ReorderedKinetics.jl) treats a spurious
    # negative fit value as vanishingly small rather than crashing.
    Φ = @. 1.0 / sqrt(8.0) / sqrt(1.0 + Wk_over_Wj) *
        (1.0 + sqrt(positive_rate_argument(ηk_over_ηj)) * (Wj_over_Wk)^(0.25))^2

    η_mix = sum((X .* η') ./ (Φ * X))

    ## thermal_conductivity_func
    λ = trans_T' * gas.trans.thermal_conductivity_poly

    λ_mix = (sum(X .* λ') + 1.0 / sum(X ./ λ')) / 2.0

    ## binary_diff_coeffs_func

    X_eps = clamp.(X, 1.e-12, Inf)
    D = reshape(trans_T' * gas.trans.binary_diff_coeffs_poly, 
                gas.n_species, gas.n_species)
    for i in 1:gas.n_species
        D[i, i] = Inf
    end

    XjWj = X_eps' * gas.MW .- X_eps .* gas.MW

    XjDjk = sum(X_eps .* (1.0 ./ D'), dims=1) .- (X_eps ./ diag(D))'

    Dkm = XjWj ./ XjDjk' ./ mean_MW / P * one_atm

    return η_mix, λ_mix, Dkm
end
export mix_trans

function viscosity(gas::A, T, X) where {A <: Arrhenius.Solution}

    logT = log(T)
    trans_T = [logT^6, logT^5, logT^4, logT^3, logT^2, logT, 1.0]
    
    ## species_viscosities_poly

    η = trans_T' * gas.trans.species_viscosities_poly

    Wk_over_Wj = gas.MW * (1.0 ./ gas.MW')

    Wj_over_Wk = 1 ./ Wk_over_Wj

    ηk_over_ηj = η' * (1.0 ./ η)

    # ηk_over_ηj can go slightly negative -- η is a fitted log(T) polynomial,
    # not a value with a guaranteed-positive functional form, and can dip
    # below zero just outside its well-fit range (confirmed directly: a
    # BFER flame Jacobian assembly, which probes T at ForwardDiff.Dual-
    # perturbed values around each cell's real temperature, hit
    # `sqrt(-0.05...)` here). Physically, viscosity is never negative, so
    # flooring with `positive_rate_argument` (same convention used for the
    # same reason throughout ReorderedKinetics.jl) treats a spurious
    # negative fit value as vanishingly small rather than crashing.
    Φ = @. 1.0 / sqrt(8.0) / sqrt(1.0 + Wk_over_Wj) *
        (1.0 + sqrt(positive_rate_argument(ηk_over_ηj)) * (Wj_over_Wk)^(0.25))^2

    return sum((X .* η') ./ (Φ * X))
end

function conductivity(gas::A, T, X) where {A <: Arrhenius.Solution}
    logT = log(T)
    trans_T = [logT^6, logT^5, logT^4, logT^3, logT^2, logT, 1.0]
    ## thermal_conductivity_func
    λ = trans_T' * gas.trans.thermal_conductivity_poly

    return (sum(X .* λ') + 1.0 / sum(X ./ λ')) / 2.0
end
export conductivity , viscosity
