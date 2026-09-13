module Arrhenius
    using YAML
    using LinearAlgebra
    using SparseArrays
    #using Symbolics
    #using TimerOutputs
    #using LoopVectorization 
    #using IfElse
    
    include("Constants.jl")
    include("DataStructure.jl")
    #include("Reactions.jl")
    include("ReorderedKinetics.jl")
    include("PackedKinetics.jl")
    include("parser.jl")
    include("Solution.jl")
    include("Magic.jl")
    include("Thermo.jl")
    include("State.jl")
    
    include("StandardKinetics.jl")
    include("Transport.jl")
    include("ODE_functions.jl")
end
