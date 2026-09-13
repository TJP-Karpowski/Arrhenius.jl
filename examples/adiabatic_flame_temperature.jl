using Arrhenius
using Printf

const MECH_DIR = joinpath(@__DIR__, "..", "mechanism")

"""
Compute methane-air adiabatic flame temperatures over equivalence ratio.

The reactants enter at `inlet_temperature`. Pressures are specified in bar and
converted to Pa. One equilibrium workspace is reused because every state uses
the same GRI-Mech 3.0 `Solution`.
"""
function adiabatic_flame_temperature_sweep(;
    phis = collect(0.5:0.1:2.0),
    pressures_bar = [1.0, 10.0, 30.0],
    inlet_temperature = 300.0,
)
    gas = CreateSolution(joinpath(MECH_DIR, "gri30.yaml"))
    state = GasState(gas)
    workspace = EquilibriumWorkspace(gas)
    flame_temperatures = zeros(length(phis), length(pressures_bar))

    for (pressure_index, pressure_bar) in pairs(pressures_bar)
        pressure = pressure_bar * 1.0e5
        for (phi_index, phi) in pairs(phis)
            # CH4 + 2 O2 is stoichiometric, so scaling CH4 by phi gives the
            # requested equivalence ratio while retaining air's N2/O2 ratio.
            composition = "CH4:$(phi),O2:2,N2:7.52"
            set_TPX!(state, inlet_temperature, pressure, composition)
            result = equilibrate!(state, workspace, :HP)
            result.converged || error("HP equilibrium failed at phi=$phi, p=$(pressure_bar) bar")
            flame_temperatures[phi_index, pressure_index] = temperature(state)
        end
    end

    return (; phis, pressures_bar, flame_temperatures)
end

result = adiabatic_flame_temperature_sweep()

header = ["phi"; ["$(pressure) bar" for pressure in result.pressures_bar]]
println(join(rpad.(header, 14), ""))
for (i, phi) in pairs(result.phis)
    @printf("%-14.2f", phi)
    for temperature in result.flame_temperatures[i, :]
        @printf("%-14.1f", temperature)
    end
    println()
end

result
