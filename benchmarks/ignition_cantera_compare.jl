#!/usr/bin/env julia

using Pkg
Pkg.instantiate()

using ADTypes
using Arrhenius
using Dates
using LinearAlgebra
using OrdinaryDiffEq
using OrdinaryDiffEqRosenbrock
using OrdinaryDiffEqSDIRK
using Printf
using Statistics
using Sundials

const RELTOL = 1.0e-6
const ABSTOL = 1.0e-9
const REFERENCE_RELTOL = 1.0e-9
const REFERENCE_ABSTOL = 1.0e-12
const FINAL_T_ATOL = 1.0e-3
const FINAL_Y_ATOL = 1.0e-6
const MAJOR_SPECIES_THRESHOLD = 1.0e-6

const CASES = [
    (
        name = "h2o2_hydrogen_air",
        mechanism = joinpath(@__DIR__, "..", "mechanism", "h2o2.yaml"),
        temperature = 1000.0,
        pressure = Arrhenius.one_atm,
        t_end = 1.0e-3,
        composition = ("H2" => 2.0, "O2" => 1.0),
        composition_string = "H2:2,O2:1",
    ),
    (
        name = "h2_li_2004_hydrogen_air",
        mechanism = joinpath(@__DIR__, "..", "mechanism", "h2_li_2004.yaml"),
        temperature = 1000.0,
        pressure = Arrhenius.one_atm,
        t_end = 1.0e-3,
        composition = ("H2" => 2.0, "O2" => 1.0, "N2" => 3.76),
        composition_string = "H2:2,O2:1,N2:3.76",
        solve_kwargs = (dtmax = 1.0e-6,),
        cantera_max_step = 1.0e-6,
    ),
    (
        name = "gri30_methane_air",
        mechanism = joinpath(@__DIR__, "..", "mechanism", "gri30.yaml"),
        temperature = 1500.0,
        pressure = 10.0 * Arrhenius.one_atm,
        t_end = 2.0e-3,
        composition = ("CH4" => 1.0, "O2" => 2.0, "N2" => 7.52),
        composition_string = "CH4:1,O2:2,N2:7.52",
    ),
]

function invalid_reactor_state(u, p, t)
    return u[end] <= 0 || any(x -> x < -1.0e-14, @view(u[1:end-1]))
end

const ARRHENIUS_METHODS = [
    (
        implementation = "Arrhenius",
        method = "Rosenbrock23(no jac)",
        jacobian = :none,
        algorithm = () -> Rosenbrock23(autodiff=AutoFiniteDiff()),
        solve_kwargs = (isoutofdomain = invalid_reactor_state,),
    ),
    (
        implementation = "Arrhenius",
        method = "Rosenbrock23(jac=Analytical)",
        jacobian = :analytical,
        algorithm = () -> Rosenbrock23(autodiff=AutoFiniteDiff()),
        solve_kwargs = (isoutofdomain = invalid_reactor_state,),
    ),
    (
        implementation = "Arrhenius",
        method = "Rodas4P(jac=Generated)",
        jacobian = :generated,
        algorithm = () -> Rodas4P(autodiff=AutoFiniteDiff()),
        solve_kwargs = (isoutofdomain = invalid_reactor_state,),
    ),
    (
        implementation = "Arrhenius",
        method = "CVODE_BDF(jac=Generated)",
        jacobian = :generated,
        algorithm = () -> CVODE_BDF(),
        solve_kwargs = (alg_hints = [:stiff],),
    ),
    (
        implementation = "Arrhenius",
        method = "CVODE_BDF(jac=Generated,rhs=Generated)",
        jacobian = :generated_rhs,
        algorithm = () -> CVODE_BDF(),
        solve_kwargs = (alg_hints = [:stiff],),
    ),
]

function case_solve_kwargs(case)
    return :solve_kwargs in propertynames(case) ? case.solve_kwargs : NamedTuple()
end

function cantera_max_step(case)
    return :cantera_max_step in propertynames(case) ? case.cantera_max_step : 0.0
end

function mass_fractions(gas, composition)
    X = zeros(gas.n_species)
    for (species, amount) in composition
        idx = species_index(gas, species)
        idx === nothing && error("Species '$species' is not present in mechanism")
        X[idx] = amount
    end
    X ./= sum(X)
    return X2Y(gas, X, dot(X, gas.MW))
end

function sample!(f; samples::Int)
    times_ns = Vector{Int64}(undef, samples)
    allocations = Vector{Int64}(undef, samples)
    for i in 1:samples
        GC.gc(false)
        allocations[i] = @allocated f()
        GC.gc(false)
        t0 = time_ns()
        f()
        times_ns[i] = time_ns() - t0
    end
    return (
        min_ns = minimum(times_ns),
        median_ns = median(times_ns),
        mean_ns = mean(times_ns),
        max_ns = maximum(times_ns),
        alloc_bytes = median(allocations),
    )
end

function stat_value(stats, name::Symbol)
    return name in propertynames(stats) ? getproperty(stats, name) : 0
end

function build_problem(case; jacobian = :none)
    gas = CreateSolution(case.mechanism)
    state = isobaricReactor(gas)
    u0 = vcat(mass_fractions(gas, case.composition), case.temperature)
    tspan = (0.0, case.t_end)

    if jacobian == :analytical
        jac = AnalyticalIsobaricJacobianWorkspace(gas)
        return gas, ODEProblem(ODEFunction(state; jac = jac), u0, tspan, (case.pressure,))
    elseif jacobian == :generated
        jac = GeneratedAnalyticalIsobaricJacobianWorkspace(gas)
        return gas, ODEProblem(ODEFunction(state; jac = jac), u0, tspan, (case.pressure,))
    elseif jacobian == :generated_rhs
        state = GeneratedIsobaricReactor(gas)
        jac = GeneratedAnalyticalIsobaricJacobianWorkspace(gas)
        return gas, ODEProblem(ODEFunction(state; jac = jac), u0, tspan, (case.pressure,))
    elseif jacobian != :none
        error("Unknown Jacobian variant: $jacobian")
    end

    return gas, ODEProblem(state, u0, tspan, (case.pressure,))
end

function is_successful(sol)
    return string(sol.retcode) == "Success"
end

function build_reference(case)
    gas, prob = build_problem(case; jacobian = :none)
    algorithm, solve_kwargs = case.name == "h2_li_2004_hydrogen_air" ?
        (Rosenbrock23(autodiff=AutoFiniteDiff()), NamedTuple()) :
        (CVODE_BDF(), (alg_hints = [:stiff],))
    sol = solve(prob, algorithm;
        reltol = REFERENCE_RELTOL,
        abstol = REFERENCE_ABSTOL,
        save_everystep = false,
        solve_kwargs...,
    )
    !is_successful(sol) && error("Reference solve failed for $(case.name): $(sol.retcode)")
    major_species = [
        i for i in 1:gas.n_species
        if abs(prob.u0[i]) >= MAJOR_SPECIES_THRESHOLD || abs(sol.u[end][i]) >= MAJOR_SPECIES_THRESHOLD
    ]
    return (u = sol.u[end], major_species = major_species)
end

function final_state_validation(final_u, reference)
    final_T_abs_err = abs(final_u[end] - reference.u[end])
    final_Y_major_linf_err = maximum(abs(final_u[i] - reference.u[i]) for i in reference.major_species)
    validation = final_T_abs_err <= FINAL_T_ATOL && final_Y_major_linf_err <= FINAL_Y_ATOL ? "pass" : "fail"
    return (
        validation = validation,
        final_T_abs_err = final_T_abs_err,
        final_Y_major_linf_err = final_Y_major_linf_err,
    )
end

function failed_row(case, gas, implementation, method, retcode, message)
    return (
        case = case.name,
        mechanism = basename(case.mechanism),
        species = gas === nothing ? 0 : gas.n_species,
        reactions = gas === nothing ? 0 : gas.n_reactions,
        implementation = implementation,
        method = method,
        t_end = case.t_end,
        reltol = RELTOL,
        abstol = ABSTOL,
        retcode = retcode,
        validation = "failed",
        final_T_abs_err = -1.0,
        final_Y_major_linf_err = -1.0,
        nf = 0,
        njacs = 0,
        nsteps = 0,
        min_ns = -1,
        median_ns = -1,
        mean_ns = -1,
        max_ns = -1,
        alloc_bytes = -1,
        notes = message,
    )
end

function compact_error(err)
    msg = replace(sprint(showerror, err), '\n' => ' ')
    length(msg) > 220 && (msg = first(msg, 220) * "...")
    return "$(nameof(typeof(err))): $msg"
end

function benchmark_arrhenius(case, spec, reference; samples)
    gas, prob = build_problem(case; jacobian = spec.jacobian)
    solve_kwargs = merge(case_solve_kwargs(case), spec.solve_kwargs)
    warmup_sol = try
        solve(prob, spec.algorithm();
            reltol = RELTOL,
            abstol = ABSTOL,
            save_everystep = false,
            solve_kwargs...,
        )
    catch err
        return failed_row(case, gas, spec.implementation, spec.method, "Error", compact_error(err))
    end

    if !is_successful(warmup_sol)
        return failed_row(case, gas, spec.implementation, spec.method, string(warmup_sol.retcode), "warmup retcode")
    end

    validation = final_state_validation(warmup_sol.u[end], reference)
    timing = validation.validation == "pass" ? sample!(samples=samples) do
        sol = solve(prob, spec.algorithm();
            reltol = RELTOL,
            abstol = ABSTOL,
            save_everystep = false,
            solve_kwargs...,
        )
        !is_successful(sol) && error("timed solve returned $(sol.retcode)")
        return sol
    end : (min_ns = -1, median_ns = -1, mean_ns = -1, max_ns = -1, alloc_bytes = -1)

    stats = warmup_sol.stats
    return (
        case = case.name,
        mechanism = basename(case.mechanism),
        species = gas.n_species,
        reactions = gas.n_reactions,
        implementation = spec.implementation,
        method = spec.method,
        t_end = case.t_end,
        reltol = RELTOL,
        abstol = ABSTOL,
        retcode = string(warmup_sol.retcode),
        validation...,
        nf = stat_value(stats, :nf),
        njacs = stat_value(stats, :njacs),
        nsteps = stat_value(stats, :naccept) + stat_value(stats, :nreject),
        timing...,
        notes = "",
    )
end

function parse_key_value_output(text)
    values = Dict{String,String}()
    for line in split(chomp(text), '\n')
        isempty(line) && continue
        key, value = split(line, "="; limit = 2)
        values[key] = value
    end
    return values
end

function run_cantera(case; samples)
    python = get(ENV, "ARRHENIUS_CANTERA_PYTHON", "python3")
    runner = joinpath(@__DIR__, "cantera_ignition_runner.py")
    cmd = `$(python) $(runner) --mechanism $(case.mechanism) --temperature $(case.temperature) --pressure $(case.pressure) --composition $(case.composition_string) --t-end $(case.t_end) --rtol $(RELTOL) --atol $(ABSTOL) --samples $(samples) --max-step $(cantera_max_step(case))`
    return readchomp(cmd)
end

function benchmark_cantera(case, reference; samples)
    gas = try
        CreateSolution(case.mechanism)
    catch err
        nothing
    end

    output = try
        run_cantera(case; samples = samples)
    catch err
        return failed_row(case, gas, "Cantera", "IdealGasConstPressureReactor/CVODE", "skipped", compact_error(err))
    end

    values = parse_key_value_output(output)
    if get(values, "status", "error") != "ok"
        return failed_row(case, gas, "Cantera", "IdealGasConstPressureReactor/CVODE", "Error", output)
    end

    final_Y = parse.(Float64, split(values["final_mass_fractions"], ';'))
    final_u = vcat(final_Y, parse(Float64, values["final_temperature"]))
    validation = final_state_validation(final_u, reference)
    return (
        case = case.name,
        mechanism = basename(case.mechanism),
        species = parse(Int, values["species"]),
        reactions = parse(Int, values["reactions"]),
        implementation = "Cantera",
        method = "IdealGasConstPressureReactor/CVODE",
        t_end = case.t_end,
        reltol = RELTOL,
        abstol = ABSTOL,
        retcode = values["retcode"],
        validation...,
        nf = parse(Int, values["rhs_evals"]),
        njacs = parse(Int, values["jac_evals"]),
        nsteps = parse(Int, values["steps"]),
        min_ns = round(Int, parse(Float64, values["min_ns"])),
        median_ns = round(Int, parse(Float64, values["median_ns"])),
        mean_ns = round(Int, parse(Float64, values["mean_ns"])),
        max_ns = round(Int, parse(Float64, values["max_ns"])),
        alloc_bytes = -1,
        notes = "Cantera $(values["cantera_version"])",
    )
end

function write_csv(path, rows)
    open(path, "w") do io
        println(io, "timestamp,julia_version,case,mechanism,species,reactions,implementation,method,t_end,reltol,abstol,retcode,validation,final_T_abs_err,final_Y_major_linf_err,nf,njacs,nsteps,min_ns,median_ns,mean_ns,max_ns,alloc_bytes,notes")
        timestamp = Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS")
        for row in rows
            println(io, join((
                timestamp,
                VERSION,
                row.case,
                row.mechanism,
                row.species,
                row.reactions,
                row.implementation,
                row.method,
                row.t_end,
                row.reltol,
                row.abstol,
                row.retcode,
                row.validation,
                row.final_T_abs_err,
                row.final_Y_major_linf_err,
                row.nf,
                row.njacs,
                row.nsteps,
                round(Int, row.min_ns),
                round(Int, row.median_ns),
                round(Int, row.mean_ns),
                round(Int, row.max_ns),
                round(Int, row.alloc_bytes),
                replace(row.notes, "," => ";"),
            ), ","))
        end
    end
end

function write_markdown(path, rows; samples, python)
    open(path, "w") do io
        println(io, "# Arrhenius vs Cantera Ignition Benchmark")
        println(io)
        println(io, "- Timestamp: $(Dates.format(now(), dateformat"yyyy-mm-dd HH:MM:SS"))")
        println(io, "- Julia: $(VERSION)")
        println(io, "- Cantera Python: `$python`")
        println(io, "- Samples: $samples")
        println(io, "- Tolerances: `reltol=$RELTOL`, `abstol=$ABSTOL`")
        println(io, "- Reference: high-tolerance Arrhenius solve, `reltol=$REFERENCE_RELTOL`, `abstol=$REFERENCE_ABSTOL`")
        println(io, "- Validation: final `T` absolute error <= $FINAL_T_ATOL and final major-species mass-fraction Linf error <= $FINAL_Y_ATOL")
        println(io, "- Cantera allocations are not measured from Julia and are recorded as `n/a`.")
        println(io)
        println(io, "| Case | Implementation | Method | Species | Reactions | Median solve | Alloc | Validation |")
        println(io, "|---|---|---|---:|---:|---:|---:|---|")
        for row in rows
            time_text = row.median_ns < 0 ? "failed" : @sprintf("%.3f ms", row.median_ns / 1.0e6)
            alloc_text = row.alloc_bytes < 0 ? "n/a" : string(round(Int, row.alloc_bytes), " B")
            @printf(io, "| `%s` | `%s` | `%s` | %d | %d | %s | %s | `%s` |\n",
                row.case,
                row.implementation,
                row.method,
                row.species,
                row.reactions,
                time_text,
                alloc_text,
                row.validation,
            )
        end
        println(io)
        println(io, "| Case | Implementation | Method | retcode | final T err | final major Y err | nf/rhs | njacs | nsteps | Notes |")
        println(io, "|---|---|---|---|---:|---:|---:|---:|---:|---|")
        for row in rows
            T_err_text = row.final_T_abs_err < 0 ? "failed" : @sprintf("%.3e", row.final_T_abs_err)
            Y_err_text = row.final_Y_major_linf_err < 0 ? "failed" : @sprintf("%.3e", row.final_Y_major_linf_err)
            @printf(io, "| `%s` | `%s` | `%s` | `%s` | %s | %s | %d | %d | %d | %s |\n",
                row.case,
                row.implementation,
                row.method,
                row.retcode,
                T_err_text,
                Y_err_text,
                row.nf,
                row.njacs,
                row.nsteps,
                row.notes,
            )
        end
    end
end

function main()
    samples = parse(Int, get(ENV, "ARRHENIUS_IGNITION_COMPARE_SAMPLES", "3"))
    include_cantera = parse(Bool, lowercase(get(ENV, "ARRHENIUS_COMPARE_CANTERA", "true")))
    python = get(ENV, "ARRHENIUS_CANTERA_PYTHON", "python3")
    case_filter = split(get(ENV, "ARRHENIUS_IGNITION_COMPARE_CASES", ""), ','; keepempty = false)
    cases = isempty(case_filter) ? CASES : [case for case in CASES if case.name in case_filter]
    isempty(cases) && error("No benchmark cases selected by ARRHENIUS_IGNITION_COMPARE_CASES")

    rows = NamedTuple[]
    for case in cases
        println("Benchmarking $(case.name)")
        reference = build_reference(case)
        for spec in ARRHENIUS_METHODS
            push!(rows, benchmark_arrhenius(case, spec, reference; samples = samples))
        end
        if include_cantera
            push!(rows, benchmark_cantera(case, reference; samples = samples))
        end
    end

    mkpath(joinpath(@__DIR__, "results"))
    stamp = Dates.format(now(), dateformat"yyyymmdd_HHMMSS")
    csv_path = joinpath(@__DIR__, "results", "ignition_cantera_compare_$stamp.csv")
    md_path = joinpath(@__DIR__, "results", "ignition_cantera_compare_$stamp.md")
    write_csv(csv_path, rows)
    write_markdown(md_path, rows; samples = samples, python = python)
    println("Wrote $csv_path")
    println("Wrote $md_path")
end

main()
