using Arrhenius
using LinearAlgebra
using Test

const MECH_DIR = joinpath(@__DIR__, "..", "mechanism")

function mechanism_path(name)
    return joinpath(MECH_DIR, name)
end

function mass_fractions(gas, composition)
    X = zeros(gas.n_species)
    for (species, amount) in composition
        idx = species_index(gas, species)
        @test idx !== nothing
        X[idx] = amount
    end
    X ./= sum(X)
    return X2Y(gas, X, dot(X, gas.MW))
end

@testset "activation energy units" begin
    @test Arrhenius.parse_activation_energy_cal_per_mol(10.0) == 10.0
    @test Arrhenius.parse_activation_energy_cal_per_mol("4.184 J/mol") ≈ 1.0
    @test Arrhenius.parse_activation_energy_cal_per_mol("4.184 kJ/mol") ≈ 1000.0
    @test Arrhenius.parse_activation_energy_cal_per_mol("1.5 kcal/mol") ≈ 1500.0
end

@testset "parser equation edge cases" begin
    species = ["O2", "H2O", "CH4", "CO", "CO2", "M"]
    yaml = Dict(
        "reactions" => Any[
            Dict(
                "equation" => "CH4 + 1.5 O2 => CO + 2 H2O",
                "rate-constant" => Dict("A" => 2.0e12, "b" => 0.0, "Ea" => "3.5e4 cal/mol"),
                "orders" => Dict("CH4" => 0.9, "O2" => 1.1),
            ),
            Dict(
                "equation" => "CO + 5.00E-01 O2 <=> CO2",
                "rate-constant" => Dict("A" => 6.324555e7, "b" => 0.0, "Ea" => "1.2e4 cal/mol"),
            ),
            Dict(
                "equation" => "2 CO (+ M) <=> CO2 (+M)",
                "type" => "falloff",
                "high-P-rate-constant" => Dict("A" => 1.0e12, "b" => 0.0, "Ea" => 0.0),
                "low-P-rate-constant" => Dict("A" => 1.0e18, "b" => -1.0, "Ea" => 100.0),
                "Troe" => Dict("A" => 0.5, "T1" => 1.0e30, "T3" => 1.0e-30),
                "efficiencies" => Dict("CO" => 1.5, "CO2" => 2.0),
            ),
            Dict(
                "equation" => "CO + O2 <=> CO2 + O2",
                "duplicate" => true,
                "rate-constant" => Dict("A" => 1.0, "b" => 0.0, "Ea" => 0.0),
            ),
            Dict(
                "equation" => "CO + O2 <=> CO2 + O2",
                "duplicate" => true,
                "rate-constant" => Dict("A" => 2.0, "b" => 0.0, "Ea" => 0.0),
            ),
        ],
    )

    kin = Arrhenius.build_kinetic_data_from_yaml(yaml, species)
    idx = Dict(s => i for (i, s) in enumerate(species))

    @test kin.reactant_stoich_coeffs[idx["CH4"], 1] == 1.0
    @test kin.reactant_stoich_coeffs[idx["O2"], 1] == 1.5
    @test kin.product_stoich_coeffs[idx["CO"], 1] == 1.0
    @test kin.product_stoich_coeffs[idx["H2O"], 1] == 2.0
    @test kin.reactant_orders[idx["CH4"], 1] == 0.9
    @test kin.reactant_orders[idx["O2"], 1] == 1.1

    @test kin.reactant_stoich_coeffs[idx["O2"], 2] == 0.5
    @test kin.is_reversible[2]

    @test kin.reactant_stoich_coeffs[idx["CO"], 3] == 2.0
    @test kin.product_stoich_coeffs[idx["CO2"], 3] == 1.0
    @test kin.Arrhenius_0[1, :] ≈ [1.0e18, -1.0, 100.0]
    @test kin.Troe_[1, :] ≈ [0.5, 1.0e30, 1.0e30, 1.0e-30]
    @test kin.efficiencies_coeffs_full[idx["CO"], 3] == 1.5
    @test kin.efficiencies_coeffs_full[idx["CO2"], 3] == 2.0

    @test kin.Arrhenius_coeffs[4, 1] == 1.0
    @test kin.Arrhenius_coeffs[5, 1] == 2.0
end

@testset "parser species aliases" begin
    yaml = Dict(
        "species" => Any[
            Dict("name" => "N2", "composition" => Dict("N" => 2.0)),
            Dict("name" => "HE", "composition" => Dict("He" => 1.0)),
        ],
        "reactions" => Any[
            Dict(
                "equation" => "He + N2 <=> He + N2",
                "rate-constant" => Dict("A" => 1.0, "b" => 0.0, "Ea" => "1.0 K"),
            ),
        ],
    )
    species = ["N2", "HE"]
    MW = Arrhenius.build_MW_from_yaml(yaml, [" N2", "HE"])
    kin = Arrhenius.build_kinetic_data_from_yaml(yaml, species)

    @test MW[1] ≈ 2 * Arrhenius.ELEMENT_MW["N"]
    @test MW[2] ≈ Arrhenius.ELEMENT_MW["He"]
    @test kin.reactant_stoich_coeffs[2, 1] == 1.0
    @test kin.product_stoich_coeffs[2, 1] == 1.0
end

@testset "YAML mechanism construction" begin
    cases = [
        ("2S_CH4_CM2.yaml", 6, 2),
        ("He_Ne.yaml", 2, 1),
        ("h2o2.yaml", 9, 28),
        ("h2_li_2004.yaml", 9, 21),
        ("gri30.yaml", 53, 325),
    ]

    for (file, n_species, n_reactions) in cases
        gas = CreateSolution(mechanism_path(file))
        @test gas.n_species == n_species
        @test gas.n_reactions == n_reactions
        @test length(gas.MW) == n_species
        @test all(isfinite, gas.MW)
        @test all(>(0), gas.MW)
        @test gas.kinetics isa Arrhenius.ReorderedKinetics
    end
end

@testset "phase selection" begin
    mech = mechanism_path("2S_CH4_CM2.yaml")

    default_gas = CreateSolution(mech)
    @test default_gas.species_names == ["O2", "H2O", "CH4", "CO", "CO2", "N2"]

    indexed_gas = CreateSolution(mech; phase_index = 4)
    named_gas = CreateSolution(mech; phase_name = "CH4_CM2_avbp")
    positional_gas = CreateSolution(mech; phase = "CH4_CM2_avbp")

    @test indexed_gas.species_names == ["N2", "O2", "H2O", "CH4", "CO", "CO2"]
    @test named_gas.species_names == indexed_gas.species_names
    @test positional_gas.species_names == indexed_gas.species_names
    @test indexed_gas.ele_matrix == named_gas.ele_matrix
    @test indexed_gas.thermo.nasa_low == named_gas.thermo.nasa_low
end

@testset "mechanism support matrix" begin
    supported = [
        "2S_CH4_CM2.yaml",
        "He_Ne.yaml",
        "JP10skeletal.yaml",
        "LiDryer.yaml",
        "ch4_bfer.yaml",
        "gri30.yaml",
        "h2_12s.yaml",
        "h2_elte.yaml",
        "h2_li_2004.yaml",
        "h2o2.yaml",
        "nc7_ver3.1_mech.yaml",
    ]

    for file in supported
        gas = CreateSolution(mechanism_path(file))
        @test gas.n_species > 0
        @test gas.n_reactions > 0
    end

    @test_throws UndefVarError CreateSolution(mechanism_path("nDodecane_Reitz.yaml"))
end

@testset "nonallocating reactor RHS" begin
    cases = [
        (
            mechanism = "h2o2.yaml",
            composition = ("H2" => 2.0, "O2" => 1.0, "AR" => 3.76),
            temperature = 1000.0,
            pressure = one_atm,
        ),
        (
            mechanism = "gri30.yaml",
            composition = ("CH4" => 0.5, "O2" => 1.0, "N2" => 3.76),
            temperature = 980.0,
            pressure = 15.0 * one_atm,
        ),
    ]

    for case in cases
        gas = CreateSolution(mechanism_path(case.mechanism))
        state = isobaricReactor(gas)
        Y0 = mass_fractions(gas, case.composition)
        u = vcat(Y0, case.temperature)
        du = similar(u)

        state(du, u, (case.pressure,), 0.0)
        allocations = @allocated state(du, u, (case.pressure,), 0.0)

        @test allocations == 0
        @test all(isfinite, du)
    end
end

@testset "PackedKinetics parity" begin
    cases = [
        (
            mechanism = "h2o2.yaml",
            composition = ("H2" => 2.0, "O2" => 1.0, "AR" => 3.76),
            temperature = 1000.0,
            pressure = one_atm,
        ),
        (
            mechanism = "gri30.yaml",
            composition = ("CH4" => 0.5, "O2" => 1.0, "N2" => 3.76),
            temperature = 980.0,
            pressure = 15.0 * one_atm,
        ),
    ]

    for case in cases
        gas = CreateSolution(mechanism_path(case.mechanism))
        state = isobaricReactor(gas)
        Y0 = mass_fractions(gas, case.composition)
        updateReactor!(state, Y0, case.temperature, case.pressure)

        wdot_reordered = zeros(gas.n_species)
        wdot_packed = zeros(gas.n_species)
        kf_reordered = zeros(gas.n_reactions)
        kr_reordered = zeros(gas.n_reactions)
        kf_packed = zeros(gas.n_reactions)
        kr_packed = zeros(gas.n_reactions)
        packed = PackedKinetics(gas.kinetics)

        wdot_func!(
            wdot_reordered,
            kf_reordered,
            kr_reordered,
            state.C,
            state.g_RT,
            gas.kinetics,
            state.T,
        )
        wdot_func!(
            wdot_packed,
            kf_packed,
            kr_packed,
            state.C,
            state.g_RT,
            packed,
            state.T,
        )

        @test wdot_packed ≈ wdot_reordered rtol = 1.0e-12 atol = 1.0e-20
    end
end

@testset "YAML transport parser" begin
    include("transport/transport.jl")
    @test transport_test()
end
