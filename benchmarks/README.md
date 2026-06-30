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
