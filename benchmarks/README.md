# Benchmarks

This directory stores lightweight performance baselines for the current CPU
chemistry path. Run these before structural cleanup and after each major
refactor so performance regressions are visible.

```sh
julia --project=. benchmarks/baseline.jl
```

The script writes timestamped CSV and Markdown summaries under
`benchmarks/results/`.

Useful knobs:

```sh
ARRHENIUS_BASELINE_SAMPLES=30 ARRHENIUS_BASELINE_EVALS=200 julia --project=. benchmarks/baseline.jl
```

The default benchmark cases are:

- `CreateSolution`: YAML parsing and mechanism construction.
- `updateReactor!`: state update, thermo polynomial selection, mixture state.
- `wdot_func!`: reordered chemistry source term only.
- `isobaricReactor_rhs`: full nonallocating isobaric reactor RHS.

Transport coefficient fitting is included in `CreateSolution`, so construction
time also covers YAML transport parsing and polynomial generation.

## Ignition Integration

The full ODE-loop benchmark uses `OrdinaryDiffEq`, so it has a separate
benchmark environment to keep solver dependencies out of the core package
load path.

```sh
julia --project=benchmarks benchmarks/ignition.jl
```

Useful knob:

```sh
ARRHENIUS_IGNITION_SAMPLES=10 julia --project=benchmarks benchmarks/ignition.jl
```

This benchmark measures complete isobaric ignition solves with
`Rosenbrock23(autodiff=false)`, `reltol=1e-6`, `abstol=1e-9`, and
`save_everystep=false`. Treat these results as end-to-end integration
baselines, not pure chemistry RHS baselines.

## Ignition vs Cantera

The Cantera comparison benchmark measures the same autoignition cases through
several Arrhenius solver/Jacobian variants plus Cantera's
`IdealGasConstPressureReactor`/CVODE path. Cantera is optional and is invoked
through Python, so no Python bridge package is needed in the Julia environment.

```sh
julia --project=benchmarks benchmarks/ignition_cantera_compare.jl
```

Useful knobs:

```sh
ARRHENIUS_IGNITION_COMPARE_SAMPLES=5 \
ARRHENIUS_IGNITION_COMPARE_CASES=h2o2_hydrogen_air,gri30_methane_air \
ARRHENIUS_CANTERA_PYTHON=/path/to/python-with-cantera \
julia --project=benchmarks benchmarks/ignition_cantera_compare.jl
```

To run only the Arrhenius rows:

```sh
ARRHENIUS_COMPARE_CANTERA=false julia --project=benchmarks benchmarks/ignition_cantera_compare.jl
```

The script writes `ignition_cantera_compare_*.csv` and `.md` under
`benchmarks/results/`. Cantera allocations are not measured from Julia and are
reported as `n/a`. The validation column compares each row's final state against
the high-tolerance Arrhenius reference for that case, so a Cantera timing row can
be present even when final-state agreement is marked as failed.
The Li hydrogen case applies `max_time_step=1e-6` to Cantera, matching the
Arrhenius step cap used for that stiff induction case.

## Equilibrium

The equilibrium benchmark measures direct `GasState` equilibrium solves for
small hydrogen mechanisms and GRI30 methane-air TP equilibrium.

```sh
julia --project=. benchmarks/equilibrium.jl
```

Useful knob:

```sh
ARRHENIUS_EQUILIBRIUM_SAMPLES=20 julia --project=. benchmarks/equilibrium.jl
```

The output includes default and workspace-backed timings, allocations,
nonlinear iterations, residual norm, and final temperature for each case.

## Batched Metal Smoke

The optional Metal/KernelAbstractions smoke script checks the experimental
batched RHS GPU path in the benchmark environment.

```sh
julia --project=benchmarks benchmarks/batched_metal_smoke.jl
```

Current status:

- Metal requires `Float32`; `Float64` device arrays are rejected by Metal.jl.
- The simple Arrhenius/reversible kernel is validated against the CPU batched
  RHS on `2S_CH4_CM2.yaml`.
- h2o2 is validated through `NoPdepPackedBatchedMechanism`, which packs
  thermodynamic and kinetic tables into real and integer device buffers so the
  third-body/falloff/Troe kernel stays under Apple GPU argument-buffer limits.
- Pressure-dependent mechanisms still need the same packed-buffer treatment.
