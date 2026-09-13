# Ideal-gas Equilibrium

Arrhenius supports single-phase ideal-gas equilibrium at fixed temperature and
pressure (`:TP`) and fixed enthalpy and pressure (`:HP`). Both modes conserve
the active elemental abundances and return an `EquilibriumResult` containing
the convergence status, iteration count, and final residual norm.

## TP equilibrium

```julia
using Arrhenius

gas = CreateSolution("mechanism/gri30.yaml")
state = GasState(gas;
    T = 2500.0,
    p = 10.0e5,
    X = "CH4:1.5,O2:2,N2:7.52",
)

result = equilibrate!(state, :TP)
Xeq = mole_fractions(state)
```

The state is updated in place while its temperature and pressure remain fixed.
Use `equilibrate(state, :TP)` when the input state must remain unchanged, or
`equilibrium_composition(gas; T, p, X, mode=:TP)` when only mole fractions are
needed.

## HP equilibrium

```julia
state = GasState(gas;
    T = 1000.0,
    p = one_atm,
    X = "CH4:1,O2:2,N2:7.52",
)

initial_h = enthalpy_mass(state)
result = equilibrate!(state, :HP)

Tad = temperature(state)
@assert isapprox(enthalpy_mass(state), initial_h; rtol=1e-8)
```

HP uses nested TP solves and a bracketed temperature solve. Pressure and total
specific enthalpy remain fixed.

## Repeated solves

Each call can allocate its own temporary arrays, which is simplest for one-off
work. When solving many states for the same mechanism, an
`EquilibriumWorkspace` lets those arrays be reused:

```julia
workspace = EquilibriumWorkspace(gas)
equilibrate!(state, workspace, :TP)
equilibrate!(state, workspace, :HP)
```

A workspace belongs to the `Solution` used to construct it and cannot be shared
with another mechanism object. It contains scratch buffers, not an independent
thermodynamic state. Results still live in the `GasState` passed to
`equilibrate!`.

### Example: adiabatic flame temperature over phi and pressure

For methane-air, `CH4:phi,O2:2,N2:7.52` varies the fuel amount relative to the
stoichiometric reaction `CH4 + 2 O2`. This example evaluates reactants entering
at 300 K:

```julia
phis = collect(0.5:0.1:2.0)
pressures_bar = [1.0, 10.0, 30.0]
Tad = zeros(length(phis), length(pressures_bar))

state = GasState(gas)
workspace = EquilibriumWorkspace(gas)

for (j, pressure_bar) in pairs(pressures_bar)
    for (i, phi) in pairs(phis)
        set_TPX!(state, 300.0, pressure_bar * 1e5,
            "CH4:$(phi),O2:2,N2:7.52")
        result = equilibrate!(state, workspace, :HP)
        result.converged || error("equilibrium did not converge")
        Tad[i, j] = temperature(state)
    end
end
```

The complete runnable version is
`examples/adiabatic_flame_temperature.jl`. This is a sequential parameter
sweep. `BatchedIdealGasReactor` serves a different purpose: it evaluates
reaction ODE right-hand sides for many states and does not compute equilibrium.

## Validation and limitations

Stored Cantera 3.2 references exercise full equilibrium compositions for:

- H2-air with `h2_li_2004.yaml`
- CH4-air with `gri30.yaml`
- equivalence ratios 0.5, 0.75, 1.0, 1.25, 1.5, and 2.0
- pressures 0.1, 1, 10, and 100 bar
- TP at 2500 K and HP from an initial 1000 K

The suite checks convergence, elemental conservation, full species mole
fractions, fixed pressure, and HP enthalpy and temperature. Additional compact
hydrogen cases use `h2o2.yaml`.

Multiphase, surface, fixed-volume, `:UV`, and `:SV` equilibrium are not
implemented. Inputs must describe a supported ideal-gas phase.
