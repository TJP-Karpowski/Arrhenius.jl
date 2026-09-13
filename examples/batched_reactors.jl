using Arrhenius
using LinearAlgebra

const MECH_DIR = joinpath(@__DIR__, "..", "mechanism")

const HAS_METAL_BACKEND = try
    @eval using KernelAbstractions
    @eval using Metal
    true
catch err
    err isa ArgumentError || err isa LoadError || rethrow()
    false
end

function mass_fractions_from_moles(gas, composition, ::Type{T} = Float64) where {T}
    X = zeros(T, gas.n_species)
    for (species, amount) in composition
        idx = species_index(gas, species)
        idx === nothing && error("Species '$species' is not present in this mechanism")
        X[idx] = T(amount)
    end
    X ./= sum(X)
    return T.(X2Y(gas, Float64.(X), dot(Float64.(X), gas.MW)))
end

function batched_cpu_example(; real_type = Float64)
    gas = CreateSolution(joinpath(MECH_DIR, "h2o2.yaml"))
    compositions = [
        ("H2" => 2.0, "O2" => 1.0, "AR" => 3.76),
        ("H2" => 1.0, "O2" => 1.0, "AR" => 2.0),
        ("H2" => 0.5, "O2" => 1.0, "AR" => 1.0),
        ("H2" => 1.5, "O2" => 0.5, "AR" => 2.0),
    ]
    n_reactors = length(compositions)

    Y = zeros(real_type, gas.n_species, n_reactors)
    for reactor in 1:n_reactors
        Y[:, reactor] .= mass_fractions_from_moles(gas, compositions[reactor], real_type)
    end
    T = real_type[950, 1100, 1300, 1000]
    p = real_type[one_atm, 2one_atm, 0.5one_atm, one_atm]

    batch = BatchedIdealGasReactor(gas, n_reactors; real_type)
    dYdt = similar(Y)
    dTdt = similar(T)
    batched_isobaric_rhs!(dYdt, dTdt, batch, Y, T, p)

    return (; gas, Y, T, p, dYdt, dTdt)
end

function batched_metal_example()
    if !HAS_METAL_BACKEND
        @info "Metal/KernelAbstractions are not available; skipping GPU example"
        return nothing
    end

    cpu = batched_cpu_example(; real_type = Float32)
    n_reactors = size(cpu.Y, 2)

    Y_d = Metal.MtlArray(cpu.Y)
    T_d = Metal.MtlArray(cpu.T)
    p_d = Metal.MtlArray(cpu.p)
    dYdt_d = similar(Y_d)
    dTdt_d = similar(T_d)

    packed = NoPdepPackedBatchedMechanism(cpu.gas, n_reactors; array = Metal.MtlArray, real_type = Float32)
    batched_isobaric_rhs!(dYdt_d, dTdt_d, packed, Y_d, T_d, p_d, Metal.MetalBackend())

    return (;
        dYdt = Array(dYdt_d),
        dTdt = Array(dTdt_d),
        max_Y_error = maximum(abs.(Array(dYdt_d) .- cpu.dYdt)),
        max_T_error = maximum(abs.(Array(dTdt_d) .- cpu.dTdt)),
    )
end

cpu = batched_cpu_example()
metal = batched_metal_example()

println("CPU batched reactors: $(size(cpu.Y, 2))")
println("Maximum |dY/dt|: $(maximum(abs, cpu.dYdt))")
println("Maximum |dT/dt|: $(maximum(abs, cpu.dTdt))")
if metal !== nothing
    println("Metal max dYdt error vs CPU: $(metal.max_Y_error)")
    println("Metal max dTdt error vs CPU: $(metal.max_T_error)")
end

(; cpu, metal)
