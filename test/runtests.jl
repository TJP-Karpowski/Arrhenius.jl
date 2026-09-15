using Arrhenius
using ADTypes
using LinearAlgebra
using OrdinaryDiffEq
using OrdinaryDiffEqRosenbrock
using OrdinaryDiffEqSDIRK
using SparseArrays
using Sundials
using Test
using TOML

const MECH_DIR = joinpath(@__DIR__, "..", "mechanism")
const CANTERA_MECH_DIR = joinpath(@__DIR__, "mechanisms", "cantera")

function mechanism_path(name)
    return joinpath(MECH_DIR, name)
end

function cantera_mechanism_path(name)
    return joinpath(CANTERA_MECH_DIR, name)
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

function reactor_state_vector(gas, composition, temperature)
    return vcat(mass_fractions(gas, composition), temperature)
end

function ignition_problem(case; jacobian = :none)
    gas = CreateSolution(mechanism_path(case.mechanism))
    state = isobaricReactor(gas)
    u0 = reactor_state_vector(gas, case.composition, case.temperature)
    tspan = (0.0, case.t_end)

    if jacobian == :generated
        jac = GeneratedAnalyticalIsobaricJacobianWorkspace(gas)
        f = ODEFunction(state; jac = jac)
        return gas, ODEProblem(f, u0, tspan, (case.pressure,))
    elseif jacobian != :none
        error("Unknown Jacobian variant: $jacobian")
    end

    return gas, ODEProblem(state, u0, tspan, (case.pressure,))
end

function successful(sol)
    return string(sol.retcode) == "Success"
end

function invalid_reactor_state(u, p, t)
    return u[end] <= 0 || any(x -> x < -1.0e-14, @view(u[1:end-1]))
end

function major_species_indices(initial_u, reference_u, n_species; threshold = 1.0e-6)
    return [
        i for i in 1:n_species
        if abs(initial_u[i]) >= threshold || abs(reference_u[i]) >= threshold
    ]
end

function final_state_errors(sol, reference_u, species_indices)
    T_err = abs(sol.u[end][end] - reference_u[end])
    Y_err = maximum(abs(sol.u[end][i] - reference_u[i]) for i in species_indices)
    return T_err, Y_err
end

function validate_final_state(sol, reference_u, species_indices; T_atol = 1.0e-3, Y_atol = 1.0e-6)
    @test successful(sol)
    T_err, Y_err = final_state_errors(sol, reference_u, species_indices)
    @test T_err <= T_atol
    @test Y_err <= Y_atol
end

@testset "Boivin 12-step and three-step QSS mechanisms" begin
    mechanism = mechanism_path("h2_12s.yaml")
    skeletal = CreateSolution(mechanism)
    reduced = CreateSolution(mechanism; phase = "boivin-qss")
    four_step = CreateSolution(mechanism; phase = "boivin-four-step")
    h_only = CreateSolution(mechanism; phase = "boivin-h-only")
    @test skeletal.n_species == 9
    @test skeletal.n_reactions == 12
    @test reduced.species_names == ["H2", "H", "O2", "H2O", "HO2", "N2"]
    @test reduced.n_reactions == 3
    @test four_step.species_names ==
        ["H2", "H", "O2", "H2O", "HO2", "H2O2", "N2"]
    @test four_step.n_reactions == 4
    @test h_only.species_names == ["H2", "H", "O2", "H2O", "N2"]
    @test h_only.n_reactions == 2

    # Independent reference values produced by the authors' downloaded
    # h2_3s_rates_PCI_nocorr Fortran routine at this state.
    C = [0.5, 1.0e-3, 0.2, 0.1, 1.0e-4, 1.0]
    forward = zeros(3)
    reverse = zeros(3)
    algebraic = zeros(2)
    boivin_qss_rates!(forward, reverse, algebraic, C, 1200.0)
    @test forward ≈ [4.4599717579635922e4, 1.4111918850699319e6,
                     1.4005501602293199e6] rtol = 5.0e-9
    @test reverse ≈ [8.5171075865739908e1, 4.3159078509856435e1,
                     7.7265685007271113e3] rtol = 5.0e-9
    @test all(>=(0), algebraic)

    reactor = BoivinQSSReactor(reduced)
    Y = mass_fractions(reduced,
        ("H2" => 2.0, "O2" => 1.0, "N2" => 3.76))
    u = vcat(Y, 1200.0)
    du = similar(u)
    reactor(du, u, (one_atm,), 0.0)
    @test all(isfinite, du)
    @test abs(sum(@view du[1:reduced.n_species])) <= 1.0e-12
    expected_dTdt = -dot(reactor.thermo.h_RT .* Arrhenius.R .* u[end] ./
        reduced.MW, @view(du[1:reduced.n_species])) / reactor.thermo.cp_mass
    @test du[end] ≈ expected_dTdt rtol = 2.0e-14
    @test qss_species(BoivinQSSModel()) == ("O", "OH", "H2O2")

    C4 = [0.5, 1.0e-3, 0.2, 0.1, 1.0e-4, 2.0e-5, 1.0]
    forward4 = zeros(4); reverse4 = zeros(4); algebraic4 = zeros(2)
    boivin_four_step_rates!(forward4, reverse4, algebraic4, C4, 1200.0)
    @test all(isfinite, forward4)
    @test all(>=(0), forward4)
    @test all(>=(0), reverse4)
    @test all(>=(0), algebraic4)

    reactor4 = BoivinFourStepReactor(four_step)
    Y4 = mass_fractions(four_step,
        ("H2" => 2.0, "O2" => 1.0, "N2" => 3.76))
    u4 = vcat(Y4, 1200.0); du4 = similar(u4)
    reactor4(du4, u4, (one_atm,), 0.0)
    @test all(isfinite, du4)
    @test abs(sum(@view du4[1:four_step.n_species])) <= 1.0e-12
    @test qss_species(BoivinFourStepModel()) == ("O", "OH")

    reactor2 = BoivinHOnlyReactor(h_only)
    Y2 = mass_fractions(h_only,
        ("H2" => 2.0, "O2" => 1.0, "N2" => 3.76))
    u2 = vcat(Y2, 1200.0); du2 = similar(u2)
    reactor2(du2, u2, (one_atm,), 0.0)
    @test all(isfinite, du2)
    @test abs(sum(@view du2[1:h_only.n_species])) <= 1.0e-12
    @test abs(reactor2.forward[3] - reactor2.reverse[3]) <=
        1.0e-10 * max(reactor2.forward[3], reactor2.reverse[3], 1.0)
    @test all(>=(0), values(qss_concentrations(reactor2)))
    @test qss_species(BoivinHOnlyModel()) == ("O", "OH", "HO2", "H2O2")
end

@testset "EM2C virtual H2 mechanism" begin
    gas = CreateSolution(mechanism_path("v_h2_air.yaml"))
    @test gas.species_names == ["H2", "O2", "N2", "H2O", "P1", "I1"]
    @test gas.n_reactions == 4
    @test gas.kinetics.arrhenius_coeffs[:, 3] ≈
        [30736.12457721262, 61.96794995484235,
         678.578929995648, 41.64856002363403] rtol = 1.0e-14

    # Independent Cantera 3.2 kinetics reference at 1200 K and one atmosphere.
    # Only the invalid virtual-species transport geometry metadata was omitted
    # when loading the authors' file in Cantera; kinetics were unchanged.
    Y = [0.0285, 0.226, 0.7455, 0.0, 0.0, 0.0]
    reactor = isobaricReactor(gas)
    u = vcat(Y, 1200.0); du = similar(u)
    reactor(du, u, (one_atm,), 0.0)
    @test du[1:gas.n_species] ≈
        [-64487.912205067994, -511776.8389726593, 0.0,
         229169.08647405007, 215964.3184261991, 131131.34627747835] rtol = 2.0e-14
    @test abs(sum(@view du[1:gas.n_species])) <= 2.0e-9
end

@testset "activation energy units" begin
    @test Arrhenius.parse_activation_energy_cal_per_mol(10.0) == 10.0
    @test Arrhenius.parse_activation_energy_cal_per_mol("4.184 J/mol") ≈ 1.0
    @test Arrhenius.parse_activation_energy_cal_per_mol("4.184 kJ/mol") ≈ 1000.0
    @test Arrhenius.parse_activation_energy_cal_per_mol("1.5 kcal/mol") ≈ 1500.0
    @test Arrhenius.parse_activation_energy_cal_per_mol(4184.0, "J/kmol") ≈ 1.0
end

@testset "mechanism unit conversion" begin
    species = ["A", "B", "C"]
    yaml = Dict(
        "units" => Dict("length" => "cm", "time" => "s", "quantity" => "mol", "activation-energy" => "cal/mol"),
        "reactions" => Any[
            Dict(
                "equation" => "A + B => C",
                "rate-constant" => Dict("A" => 1.0e6, "b" => 0.0, "Ea" => 0.0),
            ),
            Dict(
                "equation" => "A + B + M => C + M",
                "type" => "three-body",
                "rate-constant" => Dict("A" => 1.0e12, "b" => 0.0, "Ea" => 0.0),
            ),
            Dict(
                "equation" => "A + B (+M) => C (+M)",
                "type" => "falloff",
                "high-P-rate-constant" => Dict("A" => 1.0e9, "b" => 0.0, "Ea" => 0.0),
                "low-P-rate-constant" => Dict("A" => 1.0e15, "b" => 0.0, "Ea" => 0.0),
            ),
        ],
    )

    kin = Arrhenius.build_kinetic_data_from_yaml(yaml, species)
    @test kin.Arrhenius_coeffs[1, 1] ≈ 1.0e3
    @test kin.Arrhenius_coeffs[2, 1] ≈ 1.0e6
    @test kin.Arrhenius_coeffs[3, 1] ≈ 1.0e6
    @test kin.Arrhenius_0[1, 1] ≈ 1.0e9
    @test Arrhenius.ELEMENT_MW["H"] == 1.008
    @test Arrhenius.ELEMENT_MW["N"] == 14.007
    @test Arrhenius.ELEMENT_MW["O"] == 15.999
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
                "equation" => "CO + O2 (+O2) <=> CO2 (+O2)",
                "type" => "falloff",
                "high-P-rate-constant" => Dict("A" => 1.0e12, "b" => 0.0, "Ea" => 0.0),
                "low-P-rate-constant" => Dict("A" => 1.0e18, "b" => -1.0, "Ea" => 100.0),
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
    @test kin.reactant_stoich_coeffs[idx["CO"], 4] == 1.0
    @test kin.reactant_stoich_coeffs[idx["O2"], 4] == 1.0
    @test kin.product_stoich_coeffs[idx["CO2"], 4] == 1.0
    @test kin.product_stoich_coeffs[idx["O2"], 4] == 0.0
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

    @test Arrhenius.parenthesized_collider(
        "H + O2 (+M) <=> HO2 (+M)") == "M"
    @test Arrhenius.parenthesized_collider(
        "H + O2 (+N2) <=> HO2 (+N2)") == "N2"
    @test Arrhenius.parenthesized_collider(
        "H + O2 <=> O + OH") === nothing
    @test kin.efficiencies_coeffs_full[idx["O2"], 4] == 1.0
    @test sum(kin.efficiencies_coeffs_full[:, 4]) == 1.0

    @test kin.Arrhenius_coeffs[5, 1] == 1.0
    @test kin.Arrhenius_coeffs[6, 1] == 2.0
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

@testset "pressure-dependent Arrhenius parsing" begin
    yaml = Dict(
        "reactions" => Any[
            Dict(
                "equation" => "H2 => 2 H",
                "type" => "pressure-dependent-Arrhenius",
                "rate-constants" => Any[
                    Dict("P" => "0.1 atm", "A" => 1.0e10, "b" => 0.0, "Ea" => 0.0),
                    Dict("P" => "10 atm", "A" => 1.0e12, "b" => 0.0, "Ea" => 0.0),
                ],
            ),
            Dict(
                "equation" => "H + H => H2",
                "type" => "linear-Burke",
                "colliders" => Any[
                    Dict(
                        "name" => "M",
                        "type" => "pressure-dependent-Arrhenius",
                        "rate-constants" => Any[
                            Dict("P" => "0.1 atm", "A" => 2.0e9, "b" => 0.0, "Ea" => 0.0),
                            Dict("P" => "10 atm", "A" => 2.0e11, "b" => 0.0, "Ea" => 0.0),
                        ],
                    ),
                    Dict("name" => "H", "efficiency" => Dict("A" => 0.5, "b" => 0.0, "Ea" => 0.0)),
                ],
            ),
        ],
    )

    kin = Arrhenius.build_kinetic_data_from_yaml(yaml, ["H2", "H"])
    @test kin.index_pressure_dependent == [1, 2]
    @test kin.pressure_dependent_pressures ≈ [0.1, 10.0, 0.1, 10.0]
    @test kin.pressure_dependent_arrhenius[:, 1] ≈ [1.0e10, 1.0e12, 2.0e9, 2.0e11]
    @test kin.pressure_dependent_is_linear_burke == [false, true]
    @test kin.linear_burke_efficiencies[2, 2, 1] == 0.5

    reactions = Arrhenius.ReorderedKinetics(
        kin.Arrhenius_coeffs,
        kin.reactant_orders,
        spzeros(2, 0),
        spzeros(2, 0),
        sparse((kin.product_stoich_coeffs - kin.reactant_stoich_coeffs)'),
        spzeros(0, 2),
        zeros(0),
        spzeros(2, 0),
        spzeros(2, 0),
        zeros(0, 3),
        zeros(0, 4),
        kin.pressure_dependent_pressures,
        kin.pressure_dependent_arrhenius,
        kin.pressure_dependent_ptr,
        kin.index_pressure_dependent,
        kin.pressure_dependent_is_linear_burke,
        kin.linear_burke_efficiencies,
        3,
        3,
        3,
        3,
    )
    kf = zeros(2)
    Arrhenius.arrheniusLaw!(kf, reactions.arrhenius_coeffs, log(1000.0), 4184.0 / R / 1000.0)
    C = [0.0, one_atm / R / 1000.0]
    Arrhenius.pressure_dependent_rates!(kf, reactions, C, 1000.0, log(1000.0), 4184.0 / R / 1000.0)
    @test kf[1] ≈ 1.0e11
    @test kf[2] ≈ exp((1 - log(5.0) / log(100.0)) * log(2.0e9) + log(5.0) / log(100.0) * log(2.0e11))

    wdot = zeros(2)
    kf = zeros(2)
    kr = zeros(2)
    g_RT = zeros(2)
    J = zeros(2, 2)
    production_rate_jacobian_C!(J, kf, kr, C, g_RT, reactions, 1000.0)
    wdot_func!(wdot, kf, kr, C, g_RT, reactions, 1000.0)

    schedule = Arrhenius.ProductionJacobianSchedule(reactions)
    kernel = Arrhenius.GeneratedProductionJacobianKernel(reactions, schedule)
    generated_wdot = zeros(2)
    fused_wdot = zeros(2)
    generated_J = zeros(2, 2)
    fused_J = zeros(2, 2)
    scheduled_dwdot_dT = zeros(2)
    generated_dwdot_dT = zeros(2)
    Arrhenius.wdot_func!(generated_wdot, zeros(2), zeros(2), C, g_RT, reactions, kernel, 1000.0)
    Arrhenius.production_rate_jacobian_C!(generated_J, zeros(2), zeros(2), C, g_RT, reactions, kernel, 1000.0)
    Arrhenius.production_rate_jacobian_C_and_wdot!(
        fused_J,
        fused_wdot,
        zeros(2),
        zeros(2),
        C,
        g_RT,
        reactions,
        kernel,
        1000.0,
    )
    Arrhenius.production_rate_temperature_derivative!(
        scheduled_dwdot_dT,
        zeros(2),
        zeros(2),
        C,
        g_RT,
        reactions,
        schedule,
        1000.0,
        zeros(2),
    )
    Arrhenius.production_rate_temperature_derivative!(
        generated_dwdot_dT,
        zeros(2),
        zeros(2),
        C,
        g_RT,
        reactions,
        kernel,
        1000.0,
        zeros(2),
    )
    @test generated_wdot ≈ wdot
    @test generated_J ≈ J
    @test fused_wdot ≈ wdot
    @test fused_J ≈ J
    @test generated_dwdot_dT ≈ scheduled_dwdot_dT

    for j in eachindex(C)
        h = 1.0e-6 * max(abs(C[j]), 1.0)
        C_plus = copy(C)
        C_minus = copy(C)
        C_plus[j] += h
        C_minus[j] -= h
        wdot_plus = similar(wdot)
        wdot_minus = similar(wdot)
        wdot_func!(wdot_plus, zeros(2), zeros(2), C_plus, g_RT, reactions, 1000.0)
        wdot_func!(wdot_minus, zeros(2), zeros(2), C_minus, g_RT, reactions, 1000.0)
        @test J[:, j] ≈ (wdot_plus - wdot_minus) ./ (2h) rtol = 1.0e-5 atol = 1.0e-4
    end
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

@testset "Cantera mechanism fixtures" begin
    cases = [
        (
            file = "air.yaml",
            phase = 1,
            n_species = 8,
            n_reactions = 8,
            composition = ("O2" => 0.21, "N2" => 0.78, "AR" => 0.01),
            temperature = 1200.0,
            pressure = one_atm,
        ),
        (
            file = "gri30_highT.yaml",
            phase = 1,
            n_species = 53,
            n_reactions = 325,
            composition = ("CH4" => 1.0, "O2" => 2.0, "N2" => 7.52),
            temperature = 1800.0,
            pressure = one_atm,
        ),
        (
            file = "SiF4_NH3_mec.yaml",
            phase = "gas",
            n_species = 17,
            n_reactions = 33,
            composition = ("NH3" => 4.0, "SIF4" => 3.0, "H2" => 1.0),
            temperature = 1200.0,
            pressure = one_atm,
        ),
        (
            file = "nDodecane_Reitz.yaml",
            phase = "nDodecane_IG",
            n_species = 100,
            n_reactions = 553,
            composition = ("c12h26" => 1.0, "o2" => 18.5, "n2" => 69.56),
            temperature = 900.0,
            pressure = one_atm,
        ),
    ]

    for case in cases
        gas = CreateSolution(cantera_mechanism_path(case.file); phase = case.phase)
        @test gas.n_species == case.n_species
        @test gas.n_reactions == case.n_reactions
        @test all(isfinite, gas.MW)
        @test all(>(0), gas.MW)
        @test gas.kinetics isa Arrhenius.ReorderedKinetics

        u = reactor_state_vector(gas, case.composition, case.temperature)
        state = isobaricReactor(gas)
        du = similar(u)
        state(du, u, (case.pressure,), 0.0)
        allocations = @allocated state(du, u, (case.pressure,), 0.0)
        @test allocations == 0
        @test all(isfinite, du)

        if case.file == "air.yaml"
            J = zeros(length(u), length(u))
            jac = GeneratedAnalyticalIsobaricJacobianWorkspace(gas)
            jacobian!(J, jac, u, (case.pressure,), 0.0)
            @test all(isfinite, J)
        end
    end

    @testset "Float32 batch and analytical Jacobian" begin
        gas = CreateSolution(mechanism_path("h2o2.yaml"))
        Y64 = reshape(mass_fractions(gas,
            ("H2" => 2.0, "O2" => 1.0, "AR" => 3.76)), :, 1)
        T64 = [1100.0]
        p64 = [one_atm]
        reference_Y = similar(Y64)
        reference_T = similar(T64)
        batch64 = BatchedIdealGasReactor(gas, 1)
        batched_isobaric_rhs!(reference_Y, reference_T, batch64, Y64, T64, p64)

        Y32 = Float32.(Y64)
        T32 = Float32.(T64)
        p32 = Float32.(p64)
        result_Y = similar(Y32)
        result_T = similar(T32)
        batch32 = BatchedIdealGasReactor(gas, 1; real_type=Float32)
        batched_isobaric_rhs!(result_Y, result_T, batch32, Y32, T32, p32)
        @test Float64.(result_Y) ≈ reference_Y rtol=2.0e-4 atol=1.0e-7
        @test Float64.(result_T) ≈ reference_T rtol=2.0e-4 atol=1.0e-4

        multipliers = ones(Float32, gas.n_reactions, 1)
        result_Y_with_multipliers = similar(Y32)
        result_T_with_multipliers = similar(T32)
        batched_isobaric_rhs!(result_Y_with_multipliers,
            result_T_with_multipliers, batch32, Y32, T32, p32, multipliers)
        @test result_Y_with_multipliers == result_Y
        @test result_T_with_multipliers == result_T

        jacobian = GeneratedAnalyticalIsobaricJacobianWorkspace(gas, Float32)
        J = zeros(Float32, gas.n_species + 1, gas.n_species + 1)
        u = vcat(Y32[:, 1], T32[1])
        Arrhenius.jacobian!(J, jacobian, u, (p32[1],), 0.0f0)
        @test all(isfinite, J)
    end
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
        generated_state = GeneratedIsobaricReactor(gas)
        Y0 = mass_fractions(gas, case.composition)
        u = vcat(Y0, case.temperature)
        du = similar(u)
        du_generated = similar(u)

        state(du, u, (case.pressure,), 0.0)
        generated_state(du_generated, u, (case.pressure,), 0.0)
        allocations = @allocated state(du, u, (case.pressure,), 0.0)
        generated_allocations = @allocated generated_state(du_generated, u, (case.pressure,), 0.0)

        @test allocations == 0
        @test generated_allocations == 0
        @test all(isfinite, du)
        @test du_generated ≈ du rtol = 1.0e-12 atol = 1.0e-12
    end
end

@testset "batched isobaric reactor RHS" begin
    cases = [
        (
            mechanism = "h2o2.yaml",
            compositions = [
                ("H2" => 2.0, "O2" => 1.0, "AR" => 3.76),
                ("H2" => 1.0, "O2" => 1.0, "AR" => 2.0),
                ("H2" => 0.5, "O2" => 1.0, "AR" => 1.0),
            ],
            temperatures = [950.0, 1100.0, 1300.0],
            pressures = [one_atm, 2.0 * one_atm, 0.5 * one_atm],
        ),
        (
            mechanism = "gri30.yaml",
            compositions = [
                ("CH4" => 1.0, "O2" => 2.0, "N2" => 7.52),
                ("CH4" => 0.5, "O2" => 1.0, "N2" => 3.76),
            ],
            temperatures = [1200.0, 1500.0],
            pressures = [one_atm, 10.0 * one_atm],
        ),
    ]

    for case in cases
        gas = CreateSolution(mechanism_path(case.mechanism))
        n_reactors = length(case.temperatures)
        Y = zeros(gas.n_species, n_reactors)
        for reactor in 1:n_reactors
            Y[:, reactor] .= mass_fractions(gas, case.compositions[reactor])
        end
        T = copy(case.temperatures)
        p = copy(case.pressures)
        dYdt = similar(Y)
        dTdt = similar(T)
        batch = BatchedIdealGasReactor(gas, n_reactors)

        batched_isobaric_rhs!(dYdt, dTdt, batch, Y, T, p)

        state = isobaricReactor(gas)
        du = zeros(gas.n_species + 1)
        for reactor in 1:n_reactors
            u = vcat(Y[:, reactor], T[reactor])
            state(du, u, (p[reactor],), 0.0)
            @test dYdt[:, reactor] ≈ du[1:gas.n_species] rtol = 1.0e-12 atol = 1.0e-12
            @test dTdt[reactor] ≈ du[end] rtol = 1.0e-12 atol = 1.0e-12
        end
    end
end

@testset "thermodynamic gas state" begin
    gas = CreateSolution(mechanism_path("h2o2.yaml"))
    state = GasState(gas)
    set_TPX!(state, 1000.0, one_atm, "H2:2,O2:1,AR:3.76")

    @test temperature(state) == 1000.0
    @test pressure(state) == one_atm
    @test sum(Arrhenius.mole_fractions(state)) ≈ 1.0
    @test sum(Arrhenius.mass_fractions(state)) ≈ 1.0
    @test density(state) ≈ one_atm / R / temperature(state) * mean_molecular_weight(state)

    X = Arrhenius.mole_fractions(state)
    Y = Arrhenius.mass_fractions(state)
    @test X ≈ Y2X(gas, Y)
    @test Y ≈ X2Y(gas, X)
    @test concentrations(state) ≈ X .* pressure(state) ./ (R * temperature(state))

    @test species_cp_mole(state) ≈ cal_cp(gas, temperature(state), pressure(state), X)
    @test cp_mole(state) ≈ cal_cp_mean(gas, temperature(state), pressure(state), X)
    @test cp_mass(state) ≈ cal_cpmass_mean(gas, temperature(state), pressure(state), X)

    @test species_enthalpies_mole(state) ≈ cal_h(gas, temperature(state), pressure(state), X)
    @test enthalpy_mole(state) ≈ cal_h_mean(gas, temperature(state), pressure(state), X)
    @test enthalpy_mass(state) ≈ cal_hmass_mean(gas, temperature(state), pressure(state), X)

    @test species_entropies_mole(state) ≈ cal_s(gas, temperature(state), pressure(state), X)
    @test entropy_mole(state) ≈ cal_s_mean(gas, temperature(state), pressure(state), X)
    @test entropy_mass(state) ≈ cal_smass_mean(gas, temperature(state), pressure(state), X)

    @test species_gibbs_mole(state) ≈ cal_g(gas, temperature(state), pressure(state), X)
    @test gibbs_mole(state) ≈ cal_g_mean(gas, temperature(state), pressure(state), X)
    @test gibbs_mass(state) ≈ cal_gmass_mean(gas, temperature(state), pressure(state), X)

    set_TPY!(state, 1200.0, 2 * one_atm, Dict("H2" => 0.1, "O2" => 0.2, "AR" => 0.7))
    @test temperature(state) == 1200.0
    @test pressure(state) == 2 * one_atm
    @test sum(Arrhenius.mass_fractions(state)) ≈ 1.0
    @test Arrhenius.mass_fractions(state)[species_index(gas, "AR")] ≈ 0.7
    @test Arrhenius.mole_fractions(state) ≈ Y2X(gas, Arrhenius.mass_fractions(state))

    set_TP!(state, 900.0, one_atm)
    @test temperature(state) == 900.0
    @test pressure(state) == one_atm
    @test sum(Arrhenius.mole_fractions(state)) ≈ 1.0

    wdot = net_production_rates(state)
    @test length(wdot) == gas.n_species
    @test all(isfinite, wdot)
    reactor = isobaricReactor(gas)
    du = zeros(gas.n_species + 1)
    reactor(du, vcat(Arrhenius.mass_fractions(state), temperature(state)),
        (pressure(state),), 0.0)
    @test wdot ≈ du[1:gas.n_species] .* density(state) ./ gas.MW

    @test_throws ErrorException set_TPX!(state, 1000.0, one_atm, "NOT_A_SPECIES:1")
    @test_throws ErrorException set_TPX!(state, -1.0, one_atm, X)
end

@testset "documented examples" begin
    thermo_result = include(joinpath(@__DIR__, "..", "examples", "thermo_equilibrium.jl"))
    @test thermo_result.initial.T == 1000.0
    @test thermo_result.initial.rho > 0
    @test thermo_result.tp.T ≈ 1000.0
    @test thermo_result.hp.T > thermo_result.initial.T
    @test sum(thermo_result.tp.X) ≈ 1.0
    @test sum(thermo_result.hp.X) ≈ 1.0

    flame_result = include(joinpath(@__DIR__, "..", "examples", "adiabatic_flame_temperature.jl"))
    @test size(flame_result.flame_temperatures) == (length(flame_result.phis), 3)
    @test all(isfinite, flame_result.flame_temperatures)
    @test all(>(flame_result.phis[1]), flame_result.phis[2:end])
    @test all(>(1000.0), flame_result.flame_temperatures)

    batched_result = include(joinpath(@__DIR__, "..", "examples", "batched_reactors.jl"))
    @test size(batched_result.cpu.Y, 2) == 4
    @test all(isfinite, batched_result.cpu.dYdt)
    @test all(isfinite, batched_result.cpu.dTdt)
    if batched_result.metal !== nothing
        @test batched_result.metal.max_Y_error < 1.0f-4
        @test batched_result.metal.max_T_error < 1.0f-2
    end
end

@testset "TP equilibrium against stored Cantera references" begin
    references = TOML.parsefile(joinpath(@__DIR__, "equilibrium", "cantera_equilibrium_references.toml"))
    grid_cases = [case for case in references["case"] if haskey(case, "phi") && get(case, "mode", "TP") == "TP"]
    expected_grid = Set(
        (fuel, phi, pressure_bar)
        for fuel in ("H2", "CH4"), phi in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0),
            pressure_bar in (0.1, 1.0, 10.0, 100.0)
    )
    @test Set((case["fuel"], case["phi"], case["pressure_bar"]) for case in grid_cases) == expected_grid

    for case in references["case"]
        get(case, "mode", "TP") == "TP" || continue
        gas = CreateSolution(mechanism_path(case["mechanism"]))
        state = GasState(gas)
        set_TPX!(state, case["temperature"], case["pressure"], case["composition"])
        initial_element_moles = gas.ele_matrix * mole_fractions(state)
        initial_element_ratios = initial_element_moles ./ sum(initial_element_moles)

        result = equilibrate!(state, :TP)
        @test result.converged
        @test result.residual_norm <= 1.0e-9
        @test temperature(state) ≈ case["temperature"]
        @test pressure(state) ≈ case["pressure"]
        equilibrium_element_moles = gas.ele_matrix * mole_fractions(state)
        @test equilibrium_element_moles ./ sum(equilibrium_element_moles) ≈ initial_element_ratios rtol = 1.0e-10 atol = 1.0e-12

        reference_X = zeros(gas.n_species)
        for (species, value) in zip(case["species"], case["mole_fractions"])
            idx = species_index(gas, species)
            @test idx !== nothing
            reference_X[idx] = value
        end
        @test mole_fractions(state) ≈ reference_X rtol = 1.0e-8 atol = 1.0e-12

        copied = equilibrate(GasState(gas; T = case["temperature"], p = case["pressure"], X = case["composition"]), :TP)
        @test mole_fractions(copied) ≈ reference_X rtol = 1.0e-8 atol = 1.0e-12

        Xeq = equilibrium_composition(gas; T = case["temperature"], p = case["pressure"], X = case["composition"])
        @test Xeq ≈ reference_X rtol = 1.0e-8 atol = 1.0e-12

        workspace_state = GasState(gas; T = case["temperature"], p = case["pressure"], X = case["composition"])
        workspace = EquilibriumWorkspace(gas)
        workspace_result = equilibrate!(workspace_state, workspace, :TP)
        @test workspace_result.converged
        @test mole_fractions(workspace_state) ≈ reference_X rtol = 1.0e-8 atol = 1.0e-12
    end
end

@testset "GRI30 methane-air TP equilibrium" begin
    gas = CreateSolution(mechanism_path("gri30.yaml"))
    cases = [
        (T = 1800.0, CO2 = 9.412931539688627e-2, H2O = 1.8941570277327557e-1, CO = 8.596494147255291e-4),
        (T = 2200.0, CO2 = 8.637492517990907e-2, H2O = 1.8417253463078193e-1, CO = 8.053208769362313e-3),
        (T = 2500.0, CO2 = 6.929969401607715e-2, H2O = 1.70791483839875e-1, CO = 2.371577688163727e-2),
        (T = 3000.0, CO2 = 2.8677154729835206e-2, H2O = 1.1259687053058755e-1, CO = 5.8471144175313496e-2),
    ]

    for case in cases
        state = GasState(gas)
        set_TPX!(state, case.T, one_atm, "CH4:1,O2:2,N2:7.52")
        initial_element_moles = gas.ele_matrix * mole_fractions(state)
        initial_element_ratios = initial_element_moles ./ sum(initial_element_moles)

        result = equilibrate!(state, :TP)
        @test result.converged
        @test result.residual_norm <= 1.0e-8
        @test temperature(state) ≈ case.T
        @test pressure(state) ≈ one_atm
        @test sum(mole_fractions(state)) ≈ 1.0
        @test all(>=(0), mole_fractions(state))

        equilibrium_element_moles = gas.ele_matrix * mole_fractions(state)
        @test equilibrium_element_moles ./ sum(equilibrium_element_moles) ≈ initial_element_ratios rtol = 1.0e-10 atol = 1.0e-12

        X = mole_fractions(state)
        @test X[species_index(gas, "CO2")] ≈ case.CO2 rtol = 1.0e-6 atol = 1.0e-10
        @test X[species_index(gas, "H2O")] ≈ case.H2O rtol = 1.0e-6 atol = 1.0e-10
        @test X[species_index(gas, "CO")] ≈ case.CO rtol = 1.0e-6 atol = 1.0e-10
    end
end

@testset "GRI30 methane-air HP equilibrium" begin
    gas = CreateSolution(mechanism_path("gri30.yaml"))
    state = GasState(gas)
    set_TPX!(state, 1000.0, one_atm, "CH4:1,O2:2,N2:7.52")
    initial_hmass = enthalpy_mass(state)
    initial_element_moles = gas.ele_matrix * mole_fractions(state)
    initial_element_ratios = initial_element_moles ./ sum(initial_element_moles)
    workspace = EquilibriumWorkspace(gas)

    result = equilibrate!(state, workspace, :HP)
    @test result.converged
    @test result.residual_norm <= 1.0e-8
    @test temperature(state) ≈ 2541.146449846565 rtol = 1.0e-8 atol = 1.0e-5
    @test pressure(state) ≈ one_atm
    @test enthalpy_mass(state) ≈ initial_hmass rtol = 1.0e-8 atol = 1.0e-4
    equilibrium_element_moles = gas.ele_matrix * mole_fractions(state)
    @test equilibrium_element_moles ./ sum(equilibrium_element_moles) ≈ initial_element_ratios rtol = 1.0e-10 atol = 1.0e-12

    X = mole_fractions(state)
    @test X[species_index(gas, "CO2")] ≈ 6.613902028385384e-2 rtol = 1.0e-6 atol = 1.0e-10
    @test X[species_index(gas, "H2O")] ≈ 1.6788921989328606e-1 rtol = 1.0e-6 atol = 1.0e-10
    @test X[species_index(gas, "CO")] ≈ 2.65779163694953e-2 rtol = 1.0e-6 atol = 1.0e-10
    @test X[species_index(gas, "H2")] ≈ 1.0700482066086672e-2 rtol = 1.0e-6 atol = 1.0e-10
    @test X[species_index(gas, "NO")] ≈ 5.760776528100096e-3 rtol = 1.0e-6 atol = 1.0e-10
end

@testset "HP equilibrium against stored Cantera references" begin
    references = TOML.parsefile(joinpath(@__DIR__, "equilibrium", "cantera_equilibrium_references.toml"))
    grid_cases = [case for case in references["case"] if haskey(case, "phi") && get(case, "mode", "TP") == "HP"]
    expected_grid = Set(
        (fuel, phi, pressure_bar)
        for fuel in ("H2", "CH4"), phi in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0),
            pressure_bar in (0.1, 1.0, 10.0, 100.0)
    )
    @test Set((case["fuel"], case["phi"], case["pressure_bar"]) for case in grid_cases) == expected_grid

    for case in references["case"]
        get(case, "mode", "TP") == "HP" || continue
        gas = CreateSolution(mechanism_path(case["mechanism"]))
        state = GasState(gas)
        set_TPX!(state, case["temperature"], case["pressure"], case["composition"])
        initial_hmass = enthalpy_mass(state)
        initial_element_moles = gas.ele_matrix * mole_fractions(state)
        initial_element_ratios = initial_element_moles ./ sum(initial_element_moles)

        result = equilibrate!(state, :HP)
        @test result.converged
        @test result.residual_norm <= 1.0e-8
        @test temperature(state) ≈ case["equilibrium_temperature"] rtol = 1.0e-8 atol = 1.0e-4
        @test pressure(state) ≈ case["pressure"]
        @test enthalpy_mass(state) ≈ initial_hmass rtol = 1.0e-8 atol = 1.0e-4
        equilibrium_element_moles = gas.ele_matrix * mole_fractions(state)
        @test equilibrium_element_moles ./ sum(equilibrium_element_moles) ≈ initial_element_ratios rtol = 1.0e-10 atol = 1.0e-12

        reference_X = zeros(gas.n_species)
        for (species, value) in zip(case["species"], case["mole_fractions"])
            idx = species_index(gas, species)
            @test idx !== nothing
            reference_X[idx] = value
        end
        @test mole_fractions(state) ≈ reference_X rtol = 1.0e-8 atol = 1.0e-12

        copied = equilibrate(GasState(gas; T = case["temperature"], p = case["pressure"], X = case["composition"]), :HP)
        @test temperature(copied) ≈ case["equilibrium_temperature"] rtol = 1.0e-8 atol = 1.0e-4
        @test mole_fractions(copied) ≈ reference_X rtol = 1.0e-8 atol = 1.0e-12

        workspace_state = GasState(gas; T = case["temperature"], p = case["pressure"], X = case["composition"])
        workspace = EquilibriumWorkspace(gas)
        workspace_result = equilibrate!(workspace_state, workspace, :HP)
        @test workspace_result.converged
        @test temperature(workspace_state) ≈ case["equilibrium_temperature"] rtol = 1.0e-8 atol = 1.0e-4
        @test mole_fractions(workspace_state) ≈ reference_X rtol = 1.0e-8 atol = 1.0e-12
    end
end

@testset "ignition solver support matrix" begin
    cases = Dict(
        "h2o2" => (
            mechanism = "h2o2.yaml",
            composition = ("H2" => 2.0, "O2" => 1.0),
            temperature = 1000.0,
            pressure = one_atm,
            t_end = 1.0e-3,
        ),
        "h2_li_2004" => (
            mechanism = "h2_li_2004.yaml",
            composition = ("H2" => 2.0, "O2" => 1.0, "N2" => 3.76),
            temperature = 1000.0,
            pressure = one_atm,
            t_end = 1.0e-3,
        ),
    )

    references = Dict{String, NamedTuple}()
    for (name, case) in cases
        gas, prob = ignition_problem(case; jacobian = :none)
        reference_algorithm, reference_kwargs = name == "h2_li_2004" ?
            (Rosenbrock23(autodiff=AutoFiniteDiff()), NamedTuple()) :
            (CVODE_BDF(), (alg_hints = [:stiff],))
        sol = solve(prob, reference_algorithm;
            reltol = 1.0e-9,
            abstol = 1.0e-12,
            save_everystep = false,
            reference_kwargs...,
        )
        @test successful(sol)
        references[name] = (
            u = sol.u[end],
            major_species = major_species_indices(prob.u0, sol.u[end], gas.n_species),
        )
    end

    expected_supported = [
        ("h2o2", Rodas5P(autodiff=AutoFiniteDiff()), (isoutofdomain = invalid_reactor_state,), 1.0e-3),
        ("h2o2", Rodas4P(autodiff=AutoFiniteDiff()), (isoutofdomain = invalid_reactor_state,), 1.0e-3),
        ("h2o2", CVODE_BDF(), (alg_hints = [:stiff],), 1.0e-3),
        ("h2_li_2004", Rodas5P(autodiff=AutoFiniteDiff()), (dtmax = 1.0e-6, isoutofdomain = invalid_reactor_state), 1.0e-3),
        ("h2_li_2004", Rodas4P(autodiff=AutoFiniteDiff()), (dtmax = 1.0e-6, isoutofdomain = invalid_reactor_state), 1.0e-3),
        ("h2_li_2004", KenCarp4(autodiff=AutoFiniteDiff()), (dtmax = 1.0e-6, isoutofdomain = invalid_reactor_state), 1.0e-3),
        ("h2_li_2004", TRBDF2(autodiff=AutoFiniteDiff()), (dtmax = 1.0e-6, isoutofdomain = invalid_reactor_state), 1.0e-3),
    ]

    for (name, algorithm, solve_kwargs, T_atol) in expected_supported
        _, prob = ignition_problem(cases[name]; jacobian = :generated)
        sol = solve(prob, algorithm;
            reltol = 1.0e-6,
            abstol = 1.0e-9,
            save_everystep = false,
            solve_kwargs...,
        )
        validate_final_state(sol, references[name].u, references[name].major_species; T_atol = T_atol)
    end

    _, li_rodas4p_prob = ignition_problem(cases["h2_li_2004"]; jacobian = :generated)
    li_rodas4p_sol = solve(li_rodas4p_prob, Rodas4P(autodiff=AutoFiniteDiff());
        reltol = 1.0e-6,
        abstol = 1.0e-9,
        save_everystep = false,
    )
    @test successful(li_rodas4p_sol)
    li_T_err, li_Y_err = final_state_errors(
        li_rodas4p_sol,
        references["h2_li_2004"].u,
        references["h2_li_2004"].major_species,
    )
    @test li_T_err > 1.0e-3 || li_Y_err > 1.0e-7

    _, li_cvode_prob = ignition_problem(cases["h2_li_2004"]; jacobian = :generated)
    li_cvode_sol = solve(li_cvode_prob, CVODE_BDF();
        reltol = 1.0e-6,
        abstol = 1.0e-9,
        save_everystep = false,
        alg_hints = [:stiff],
    )
    @test !successful(li_cvode_sol)
end

include("ignition/stagni_2025.jl")

@testset "isobaric reactor Jacobian" begin
    gas = CreateSolution(mechanism_path("h2o2.yaml"))
    X0 = fill(1 / gas.n_species, gas.n_species)
    Y0 = X2Y(gas, X0, dot(X0, gas.MW))
    u = vcat(Y0, 1100.0)
    p = (one_atm,)
    t = 0.0

    jac = IsobaricJacobianWorkspace(gas)
    J = zeros(length(u), length(u))
    jacobian!(J, jac, u, p, t)

    @test all(isfinite, J)

    state = isobaricReactor(gas)
    du_plus = similar(u)
    du_minus = similar(u)
    fd_col = similar(u)
    u_plus = copy(u)
    u_minus = copy(u)

    for j in eachindex(u)
        h = 1.0e-6 * max(abs(u[j]), 1.0)
        u_plus .= u
        u_minus .= u
        u_plus[j] += h
        u_minus[j] -= h
        state(du_plus, u_plus, p, t)
        state(du_minus, u_minus, p, t)
        @. fd_col = (du_plus - du_minus) / (2 * h)
        @test J[:, j] ≈ fd_col rtol = 1.0e-4 atol = 1.0e-7
    end
end

@testset "analytical production-rate concentration Jacobian" begin
    gas = CreateSolution(mechanism_path("2S_CH4_CM2.yaml"))
    state = isobaricReactor(gas)
    X0 = fill(1 / gas.n_species, gas.n_species)
    Y0 = X2Y(gas, X0, dot(X0, gas.MW))
    updateReactor!(state, Y0, 1200.0, one_atm)

    J = zeros(gas.n_species, gas.n_species)
    production_rate_jacobian_C!(
        J,
        zeros(gas.n_reactions),
        zeros(gas.n_reactions),
        state.C,
        state.g_RT,
        gas.kinetics,
        state.T,
    )

    @test all(isfinite, J)

    wdot_plus = zeros(gas.n_species)
    wdot_minus = zeros(gas.n_species)
    fd_col = zeros(gas.n_species)
    C_plus = copy(state.C)
    C_minus = copy(state.C)
    kf = zeros(gas.n_reactions)
    kr = zeros(gas.n_reactions)

    for j in eachindex(state.C)
        h = 1.0e-6 * max(abs(state.C[j]), 1.0)
        C_plus .= state.C
        C_minus .= state.C
        C_plus[j] += h
        C_minus[j] -= h
        wdot_func!(wdot_plus, kf, kr, C_plus, state.g_RT, gas.kinetics, state.T)
        wdot_func!(wdot_minus, kf, kr, C_minus, state.g_RT, gas.kinetics, state.T)
        @. fd_col = (wdot_plus - wdot_minus) / (2 * h)
        @test J[:, j] ≈ fd_col rtol = 1.0e-6 atol = 1.0e-8
    end

    reactions = Arrhenius.ReorderedKinetics(
        [2.0e3 0.0 0.0],
        sparse([1], [1], [1.0], 3, 1),
        sparse([2], [1], [1.0], 3, 1),
        sparse([1], [1], [1.0], 3, 1),
        sparse([1, 1], [1, 2], [-1.0, 1.0], 1, 3),
        sparse([1, 1], [1, 2], [-1.0, 1.0], 1, 3),
        [0.0],
        sparse([1, 2, 3], [1, 1, 1], [1.0, 2.0, 0.5], 3, 1),
        spzeros(3, 0),
        zeros(0, 3),
        zeros(0, 4),
        1,
        1,
        2,
        2,
    )
    C = [0.2, 0.1, 0.3]
    g_RT = zeros(3)
    T = 1000.0
    J_tb = zeros(3, 3)
    production_rate_jacobian_C!(
        J_tb,
        zeros(1),
        zeros(1),
        C,
        g_RT,
        reactions,
        T,
    )

    @test all(isfinite, J_tb)

    wdot_plus = zeros(3)
    wdot_minus = zeros(3)
    fd_col = zeros(3)
    C_plus = copy(C)
    C_minus = copy(C)
    kf = zeros(1)
    kr = zeros(1)

    for j in eachindex(C)
        h = 1.0e-6 * max(abs(C[j]), 1.0)
        C_plus .= C
        C_minus .= C
        C_plus[j] += h
        C_minus[j] -= h
        wdot_func!(wdot_plus, kf, kr, C_plus, g_RT, reactions, T)
        wdot_func!(wdot_minus, kf, kr, C_minus, g_RT, reactions, T)
        @. fd_col = (wdot_plus - wdot_minus) / (2 * h)
        @test J_tb[:, j] ≈ fd_col rtol = 1.0e-8 atol = 1.0e-8
    end

    gas = CreateSolution(mechanism_path("h2o2.yaml"))
    state = isobaricReactor(gas)
    X0 = fill(1 / gas.n_species, gas.n_species)
    Y0 = X2Y(gas, X0, dot(X0, gas.MW))
    updateReactor!(state, Y0, 1100.0, one_atm)

    J_h2o2 = zeros(gas.n_species, gas.n_species)
    production_rate_jacobian_C!(
        J_h2o2,
        zeros(gas.n_reactions),
        zeros(gas.n_reactions),
        state.C,
        state.g_RT,
        gas.kinetics,
        state.T,
    )

    @test all(isfinite, J_h2o2)

    generated_workspace = GeneratedAnalyticalIsobaricJacobianWorkspace(gas)
    wdot_reference = zeros(gas.n_species)
    wdot_generated = zeros(gas.n_species)
    wdot_func!(
        wdot_reference,
        zeros(gas.n_reactions),
        zeros(gas.n_reactions),
        state.C,
        state.g_RT,
        gas.kinetics,
        state.T,
    )
    wdot_func!(
        wdot_generated,
        zeros(gas.n_reactions),
        zeros(gas.n_reactions),
        state.C,
        state.g_RT,
        gas.kinetics,
        generated_workspace.kernel,
        state.T,
    )
    @test wdot_generated ≈ wdot_reference rtol = 1.0e-12 atol = 1.0e-12

    J_generated = zeros(gas.n_species, gas.n_species)
    production_rate_jacobian_C!(
        J_generated,
        zeros(gas.n_reactions),
        zeros(gas.n_reactions),
        state.C,
        state.g_RT,
        gas.kinetics,
        generated_workspace.kernel,
        state.T,
    )
    @test J_generated ≈ J_h2o2 rtol = 1.0e-12 atol = 1.0e-12

    wdot_fused = zeros(gas.n_species)
    J_fused = zeros(gas.n_species, gas.n_species)
    Arrhenius.production_rate_jacobian_C_and_wdot!(
        J_fused,
        wdot_fused,
        zeros(gas.n_reactions),
        zeros(gas.n_reactions),
        state.C,
        state.g_RT,
        gas.kinetics,
        generated_workspace.kernel,
        state.T,
    )
    @test wdot_fused ≈ wdot_reference rtol = 1.0e-12 atol = 1.0e-12
    @test J_fused ≈ J_h2o2 rtol = 1.0e-12 atol = 1.0e-12

    wdot_plus = zeros(gas.n_species)
    wdot_minus = zeros(gas.n_species)
    fd_col = zeros(gas.n_species)
    C_plus = copy(state.C)
    C_minus = copy(state.C)
    kf = zeros(gas.n_reactions)
    kr = zeros(gas.n_reactions)

    for j in eachindex(state.C)
        h = 1.0e-6 * max(abs(state.C[j]), 1.0)
        C_plus .= state.C
        C_minus .= state.C
        C_plus[j] += h
        C_minus[j] -= h
        wdot_func!(wdot_plus, kf, kr, C_plus, state.g_RT, gas.kinetics, state.T)
        wdot_func!(wdot_minus, kf, kr, C_minus, state.g_RT, gas.kinetics, state.T)
        @. fd_col = (wdot_plus - wdot_minus) / (2 * h)
        @test J_h2o2[:, j] ≈ fd_col rtol = 1.0e-6 atol = 1.0e-8
    end
end

@testset "analytical production-rate temperature derivative" begin
    for mechanism in ("2S_CH4_CM2.yaml", "h2o2.yaml")
        gas = CreateSolution(mechanism_path(mechanism))
        state = isobaricReactor(gas)
        X0 = fill(1 / gas.n_species, gas.n_species)
        Y0 = X2Y(gas, X0, dot(X0, gas.MW))
        updateReactor!(state, Y0, 1100.0, one_atm)

        dwdot_dT = zeros(gas.n_species)
        production_rate_temperature_derivative!(
            dwdot_dT,
            zeros(gas.n_reactions),
            zeros(gas.n_reactions),
            state.C,
            state.g_RT,
            gas.kinetics,
            state.T,
        )

        @test all(isfinite, dwdot_dT)

        wdot_plus = zeros(gas.n_species)
        wdot_minus = zeros(gas.n_species)
        kf = zeros(gas.n_reactions)
        kr = zeros(gas.n_reactions)
        h = 1.0e-6 * state.T
        wdot_func!(wdot_plus, kf, kr, state.C, state.g_RT, gas.kinetics, state.T + h)
        wdot_func!(wdot_minus, kf, kr, state.C, state.g_RT, gas.kinetics, state.T - h)
        fd = (wdot_plus - wdot_minus) ./ (2 * h)

        @test dwdot_dT ≈ fd rtol = 1.0e-5 atol = 1.0e-7

        generated_workspace = GeneratedAnalyticalIsobaricJacobianWorkspace(gas)
        dwdot_dT_generated = zeros(gas.n_species)
        dg_RT_dT = zeros(gas.n_species)
        Arrhenius.g_RT_temperature_derivative!(dg_RT_dT, state)
        production_rate_temperature_derivative!(
            dwdot_dT_generated,
            zeros(gas.n_reactions),
            zeros(gas.n_reactions),
            state.C,
            state.g_RT,
            gas.kinetics,
            generated_workspace.kernel,
            state.T,
            dg_RT_dT,
        )
        production_rate_temperature_derivative!(
            dwdot_dT,
            zeros(gas.n_reactions),
            zeros(gas.n_reactions),
            state.C,
            state.g_RT,
            gas.kinetics,
            state.T,
            dg_RT_dT,
        )
        @test dwdot_dT_generated ≈ dwdot_dT rtol = 1.0e-12 atol = 1.0e-12
    end
end

@testset "analytical isobaric species Jacobian" begin
    gas = CreateSolution(mechanism_path("h2o2.yaml"))
    state = isobaricReactor(gas)
    X0 = fill(1 / gas.n_species, gas.n_species)
    Y0 = X2Y(gas, X0, dot(X0, gas.MW))
    u = vcat(Y0, 1100.0)
    p = (one_atm,)
    t = 0.0

    J = zeros(gas.n_species, gas.n_species + 1)
    isobaric_species_jacobian!(
        J,
        state,
        zeros(gas.n_species, gas.n_species),
        zeros(gas.n_species),
        zeros(gas.n_species),
        zeros(gas.n_species),
        u,
        p,
        t,
    )

    @test all(isfinite, J)

    rhs = isobaricReactor(gas)
    du_plus = similar(u)
    du_minus = similar(u)
    u_plus = copy(u)
    u_minus = copy(u)
    fd_col = zeros(gas.n_species)

    for j in eachindex(u)
        h = 1.0e-6 * max(abs(u[j]), 1.0)
        u_plus .= u
        u_minus .= u
        u_plus[j] += h
        u_minus[j] -= h
        rhs(du_plus, u_plus, p, t)
        rhs(du_minus, u_minus, p, t)
        @. fd_col = (du_plus[1:gas.n_species] - du_minus[1:gas.n_species]) / (2 * h)
        @test J[:, j] ≈ fd_col rtol = 1.0e-5 atol = 1.0e-7
    end
end

@testset "analytical isobaric reactor Jacobian" begin
    gas = CreateSolution(mechanism_path("h2o2.yaml"))
    state = isobaricReactor(gas)
    X0 = fill(1 / gas.n_species, gas.n_species)
    Y0 = X2Y(gas, X0, dot(X0, gas.MW))
    u = vcat(Y0, 1100.0)
    p = (one_atm,)
    t = 0.0

    J = zeros(gas.n_species + 1, gas.n_species + 1)
    isobaric_reactor_jacobian!(
        J,
        state,
        zeros(gas.n_species, gas.n_species),
        zeros(gas.n_species),
        zeros(gas.n_species),
        zeros(gas.n_species),
        zeros(gas.n_species),
        u,
        p,
        t,
    )

    @test all(isfinite, J)

    scheduled_workspace = AnalyticalIsobaricJacobianWorkspace(gas)
    J_scheduled = similar(J)
    jacobian!(J_scheduled, scheduled_workspace, u, p, t)
    @test J_scheduled ≈ J rtol = 1.0e-12 atol = 1.0e-12

    generated_workspace = GeneratedAnalyticalIsobaricJacobianWorkspace(gas)
    J_generated = similar(J)
    jacobian!(J_generated, generated_workspace, u, p, t)
    @test J_generated ≈ J_scheduled rtol = 1.0e-12 atol = 1.0e-12

    J_sparse = isobaric_jacobian_prototype(gas)
    @test size(J_sparse) == size(J)
    @test nnz(J_sparse) == length(J)
    jacobian!(J_sparse, scheduled_workspace, u, p, t)
    @test Matrix(J_sparse) ≈ J rtol = 1.0e-12 atol = 1.0e-12
    jacobian!(J_sparse, generated_workspace, u, p, t)
    @test Matrix(J_sparse) ≈ J_scheduled rtol = 1.0e-12 atol = 1.0e-12

    rhs = isobaricReactor(gas)
    du_plus = similar(u)
    du_minus = similar(u)
    u_plus = copy(u)
    u_minus = copy(u)
    fd_col = similar(u)

    for j in eachindex(u)
        h = 1.0e-6 * max(abs(u[j]), 1.0)
        u_plus .= u
        u_minus .= u
        u_plus[j] += h
        u_minus[j] -= h
        rhs(du_plus, u_plus, p, t)
        rhs(du_minus, u_minus, p, t)
        @. fd_col = (du_plus - du_minus) / (2 * h)
        @test J[:, j] ≈ fd_col rtol = 1.0e-5 atol = 1.0e-7
    end

    sparse_Y0 = mass_fractions(gas, ("H2" => 2.0, "O2" => 1.0))
    sparse_u = vcat(sparse_Y0, 1000.0)
    isobaric_reactor_jacobian!(
        J,
        state,
        zeros(gas.n_species, gas.n_species),
        zeros(gas.n_species),
        zeros(gas.n_species),
        zeros(gas.n_species),
        zeros(gas.n_species),
        sparse_u,
        p,
        t,
    )
    @test all(isfinite, J)
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

include("nasa_intervals.jl")
