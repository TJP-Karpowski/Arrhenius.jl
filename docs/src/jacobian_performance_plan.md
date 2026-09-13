# Jacobian and Performance Roadmap

This note is a restart handoff for the next Arrhenius performance phase.

## Current Baseline

- Core parser no longer depends on NPZ.
- Construction-time allocations are not critical; RHS and integration-time allocations are.
- `PackedKinetics` is available and benchmarked against `ReorderedKinetics`.
- Ignition benchmarks compare `Rosenbrock23(autodiff=false)` with and without a supplied ForwardDiff Jacobian.
- The supplied ForwardDiff Jacobian improves solve time in the current ignition benchmark, but allocates heavily.

## Immediate Jacobian Plan

1. Add a package-level isobaric reactor Jacobian API:
   - `jacobian!(J, workspace, u, p, t)`
   - reusable workspace object
   - SciML-compatible callback signature
2. Start with a ForwardDiff-backed implementation because it gives a fast correctness reference.
3. Add tests that compare Jacobian columns against finite-difference directional derivatives.
4. Wire benchmarks to the package API instead of benchmark-local helper code.

## Analytical Jacobian Progress

- `production_rate_jacobian_C!` computes `d(wdot) / dC` for Arrhenius reactions.
- Covered terms:
  - irreversible mass-action reactions
  - reversible mass-action reactions at fixed thermodynamic state
  - third-body efficiencies through `M_eff = sum(eff_i * C_i)`
  - Lindemann falloff pressure-reduction derivatives
  - Troe falloff correction derivatives
  - production-rate temperature derivatives at fixed concentration
  - full isobaric reactor dense Jacobian assembly for `[Y..., T]`
  - packed derivative schedule used by `AnalyticalIsobaricJacobianWorkspace`
- Current tests compare against finite differences of `wdot_func!` for:
  - `2S_CH4_CM2.yaml` simple irreversible/reversible reactions
  - a synthetic reversible third-body reaction
  - `h2o2.yaml` reactions including third-body, falloff, and Troe terms
  - full h2o2 isobaric reactor species and temperature rows
  - scheduled analytical workspace against the reference analytical assembly
  - generated analytical workspace against the scheduled analytical workspace
  - generated source-term and temperature-derivative kernels against the generic implementations
- Next unsupported terms:
  - GPU-compatible generated/scheduled kernel storage
  - generated reactor assembly contractions

## Current Benchmark Signal

- Scheduled analytical Jacobian is low-allocation and solver-compatible.
- The packed derivative schedule removed the worst dense sparse-lookup overhead:
  - previous gri30 analytical median was about 9.9 s
  - scheduled gri30 analytical median is about 0.79 s in the latest five-sample run
- ForwardDiff remains faster in wall time for gri30, but allocates heavily.
- Sparse `jac_prototype` support is available through `isobaric_jacobian_prototype(gas)`.
- Sparse-prototype analytical fill reduces gri30 allocations versus dense analytical, but is slower in wall time because the full isobaric reactor Jacobian is structurally dense.
- The dense isobaric assembly now uses the diagonal plus rank-one concentration transform, reducing assembly from `O(N^3)` to `O(N^2)`.
- The derivative schedule was flattened into pointer ranges plus contiguous species/order/coefficient buffers. This reduces nested small-array traversal and is closer to the representation needed for generated kernels and GPU buffers.
- Direct callback benchmarks after flattening show the current gri30 analytical dense Jacobian costs about 411 us/call versus about 617 us/call for ForwardDiff, while both reduce the same solver work.
- The latest gri30 ignition benchmark shows analytical dense at about 494 ms versus ForwardDiff at about 533 ms and no supplied Jacobian at about 863 ms.
- For small hydrogen mechanisms, analytical dense remains dominated by fixed reactor assembly costs. The Li hydrogen ignition case benefits end-to-end, but h2o2 is not consistently faster than ForwardDiff/no-jac in this benchmark.
- Solver stats show supplied Jacobians reduce gri30 RHS calls from about 49k to about 3.3k, while `njacs`, `nsolve`, `nw`, and accept/reject counts are essentially identical across supplied Jacobian variants.
- `GeneratedAnalyticalIsobaricJacobianWorkspace` now emits mechanism-specialized reaction loops for source-term evaluation, `d(wdot)/dC`, and `d(wdot)/dT`.
- Direct callback benchmarks after generating all three chemistry loops show a large gain:
  - h2o2 generated dense: about 3.3 us versus analytical dense about 6.5 us
  - h2_li_2004 generated dense: about 2.8 us versus analytical dense about 5.2 us
  - gri30 generated dense: about 56 us versus analytical dense about 215 us and ForwardDiff about 262 us
- Generated source evaluation alone is not yet faster for gri30: about 9.1 us versus generic `wdot_func!` about 8.5 us. The current win comes mainly from generated derivative kernels.
- End-to-end `Rosenbrock23` ignition also improves:
  - h2o2: generated about 16 ms versus analytical about 21 ms
  - h2_li_2004: generated about 6 ms versus analytical about 8 ms
  - gri30: generated about 107 ms versus analytical about 248 ms, ForwardDiff about 265 ms, and no supplied Jacobian about 464 ms
- Solver stats remain nearly identical across supplied Jacobian variants for gri30, so the improvement is from cheaper Jacobian/source callback work rather than changed time stepping.
- A solver matrix now compares the Rosenbrock23 Jacobian variants plus generated-Jacobian stiff solvers, including CVODE_BDF from Sundials.
- Latest five-sample solver-matrix signal:
  - h2o2: CVODE_BDF generated is about 1.3 ms, Rodas4P generated is about 4.5 ms, Rosenbrock23 generated is about 13 ms.
  - h2_li_2004: native generated solvers report very small solve times, but CVODE_BDF returns `ConvergenceFailure`; this case needs a final-state correctness check before interpreting timings.
  - gri30: CVODE_BDF generated is about 18 ms versus Rosenbrock23 generated about 97 ms.
- Several native stiff solvers fail during warmup on h2o2/gri30 because trial stages produce negative concentrations or invalid falloff pressure ratios, causing `log`/`log10` domain errors in the RHS. This is a solver-robustness issue to fix before treating the solver matrix as final.
- Generated-kernel construction adds noticeable compile latency. Keep it opt-in until compile caching and code-size behavior are better understood.
- Next performance work should target RHS domain robustness/positivity handling, final-state validation in solver benchmarks, generated reactor assembly contractions, and GPU-compatible storage/code generation.

## Next Performance Steps

1. Make the ForwardDiff path less allocation-heavy by reusing typed reactor and output work arrays.
2. Add Jacobian sparsity construction from stoichiometry and pass `jac_prototype` to SciML.
3. Implement analytical derivatives for source terms:
   - Arrhenius rates with respect to temperature
   - concentration/order derivatives
   - reverse-rate equilibrium derivatives
   - third-body, falloff, and Troe corrections
4. Generate mechanism-specialized RHS/Jacobian kernels from the flattened schedule:
   - generated `d(wdot)/dC`, `d(wdot)/dT`, and source evaluation are implemented
   - next generate or specialize dense reactor assembly contractions
   - compare against the generic schedule on h2_li_2004 and gri30
   - keep the generic path as the correctness/reference backend
5. Move the generated/scheduled kernels onto GPU-compatible packed arrays once the CPU data layout is stable.
6. Benchmark stiff solvers with supplied Jacobians:
   - `Rosenbrock23`
   - `Rodas5P`
   - `Rodas4P`
   - `KenCarp4`
   - `TRBDF2`
   - `CVODE_BDF`
7. Add benchmark correctness checks:
   - retcode must be `Success` for a timing to be considered valid
   - final temperature and major species should match a trusted reference within tolerance
   - failed solver rows should remain visible in the markdown/CSV output
8. Defer GPU execution until RHS and Jacobian data layout are stable. The likely GPU target is many independent reactor/cell integrations, not one reactor split across GPU threads.

## Longer-Term ADER/Time Polynomial Path

- Use the Jacobian API as the first derivative provider for local time-polynomial source integration.
- Add higher derivative support incrementally after the first Jacobian path is tested.
- Keep ADER experiments behind benchmarks or experimental APIs until correctness and stability are clear.
