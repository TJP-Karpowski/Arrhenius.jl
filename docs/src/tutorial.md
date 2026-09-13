# Tutorials

These tutorials use the current state and reactor APIs. Paths assume Julia is
started from the repository root.

## Thermodynamic state workflow

Create a mechanism and initialize methane-air directly in mole amounts:

```julia
using Arrhenius

gas = CreateSolution("mechanism/gri30.yaml")
state = GasState(gas;
    T = 1000.0,
    p = 15one_atm,
    X = "CH4:1,O2:2,N2:7.52",
)

println("density = ", density(state), " kg/m^3")
println("cp = ", cp_mass(state), " J/(kg K)")
println("CH4 production rate = ",
    net_production_rates(state)[species_index(gas, "CH4")])
```

All state setters refresh cached mole fractions, mass fractions,
concentrations, density, and thermodynamic properties together.

## Adiabatic flame-temperature sweep

An adiabatic flame temperature is the HP equilibrium temperature of the inlet
mixture. The runnable
`examples/adiabatic_flame_temperature.jl` example evaluates methane-air over
`phi = 0.5:0.1:2.0` at 1, 10, and 30 bar.

```julia
gas = CreateSolution("mechanism/gri30.yaml")
state = GasState(gas)
workspace = EquilibriumWorkspace(gas)

phis = collect(0.5:0.1:2.0)
pressures_bar = [1.0, 10.0, 30.0]
Tad = zeros(length(phis), length(pressures_bar))

for (j, pressure_bar) in pairs(pressures_bar)
    for (i, phi) in pairs(phis)
        set_TPX!(state, 300.0, pressure_bar * 1e5,
            "CH4:$(phi),O2:2,N2:7.52")
        equilibrate!(state, workspace, :HP)
        Tad[i, j] = temperature(state)
    end
end
```

The workspace is scratch storage for repeated equilibrium calculations with
one mechanism. It does not represent another gas and does not change the
mathematical result. This loop is sequential; it is unrelated to the batched
reactor RHS API.

## Constant-pressure ignition

For time integration, build the initial mass fractions with `GasState`, then
use the package reactor RHS. `OrdinaryDiffEq` and `Sundials` are test/example
dependencies rather than core Arrhenius dependencies.

```julia
using OrdinaryDiffEq
using Sundials

gas = CreateSolution("mechanism/gri30.yaml")
initial = GasState(gas;
    T = 980.0,
    p = 15one_atm,
    X = "CH4:1,O2:2,N2:7.52",
)

u0 = vcat(mass_fractions(initial), temperature(initial))
rhs = isobaricReactor(gas)
problem = ODEProblem(rhs, u0, (0.0, 0.1), (pressure(initial),))
solution = solve(problem, CVODE_BDF();
    reltol = 1e-6,
    abstol = 1e-9,
    save_everystep = false,
)

final_temperature = solution.u[end][end]
```

The reactor state vector is `[Y; T]`, and the parameter tuple contains the
constant pressure. Check `solution.retcode` before using a result. Solver and
mechanism combinations should follow [Solver and Mechanism Support](@ref).

## Analytical Jacobian

Stiff native OrdinaryDiffEq solvers can use a generated analytical Jacobian:

```julia
using ADTypes

jacobian = GeneratedAnalyticalIsobaricJacobianWorkspace(gas)
ode_function = ODEFunction(rhs; jac = jacobian)
problem = ODEProblem(ode_function, u0, (0.0, 0.1), (pressure(initial),))

solution = solve(problem, Rodas5P(autodiff=AutoFiniteDiff());
    reltol = 1e-6,
    abstol = 1e-9,
    isoutofdomain = (u, p, t) -> u[end] <= 0 || any(x -> x < -1e-14, @view(u[1:end-1])),
)
```

The generated workspace selects a mechanism-specialized implementation where
appropriate and falls back to the schedule-based analytical kernel for large
pressure-dependent mechanisms.

## Batched reactor right-hand sides

`BatchedIdealGasReactor` evaluates derivatives for many independent reactor
states. It does not integrate them and is not needed for equilibrium sweeps.
Use it when a workload already stores species as
`(n_species, n_reactors)` arrays:

```julia
batch = BatchedIdealGasReactor(gas, n_reactors)
batched_isobaric_rhs!(dYdt, dTdt, batch, Y, T, p)
```

See [Examples](@ref) for the complete CPU example and experimental Metal path.

## Low-level customization

The conversion, thermodynamic, and reaction-rate array functions remain
available for custom numerical kernels. They require manual consistency of
temperature, pressure, composition, density, and units. Start with `GasState`
and `isobaricReactor`; descend to low-level functions only when an application
needs direct buffer control.
