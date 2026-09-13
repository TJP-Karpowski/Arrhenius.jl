using Arrhenius

const MECH_DIR = joinpath(@__DIR__, "..", "mechanism")

function thermo_equilibrium_example()
    gas = CreateSolution(joinpath(MECH_DIR, "h2o2.yaml"))
    state = GasState(gas)

    set_TPX!(state, 1000.0, one_atm, "H2:2,O2:1,AR:3.76")

    initial = (;
        T = temperature(state),
        p = pressure(state),
        rho = density(state),
        mean_mw = mean_molecular_weight(state),
        cp_mass = cp_mass(state),
        h_mass = enthalpy_mass(state),
        s_mass = entropy_mass(state),
        wdot_norm = maximum(abs, net_production_rates(state)),
    )

    tp_state = deepcopy(state)
    tp_result = equilibrate!(tp_state, :TP)

    hp_state = deepcopy(state)
    hp_result = equilibrate!(hp_state, :HP)

    return (;
        initial,
        tp = (result = tp_result, T = temperature(tp_state), X = mole_fractions(tp_state)),
        hp = (result = hp_result, T = temperature(hp_state), X = mole_fractions(hp_state)),
    )
end

result = thermo_equilibrium_example()

println("Initial density: $(result.initial.rho) kg/m^3")
println("Initial cp_mass: $(result.initial.cp_mass) J/(kg K)")
println("TP equilibrium T: $(result.tp.T) K")
println("HP equilibrium T: $(result.hp.T) K")

result
