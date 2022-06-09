module Arrhenius
    using YAML
    using NPZ
    using LinearAlgebra
    using SparseArrays
    #using TimerOutputs
    #using LoopVectorization 
    #using IfElse
    
    include("Constants.jl")
    include("DataStructure.jl")
    include("Reactions.jl")
    include("Solution.jl")
    include("Magic.jl")
    include("Thermo.jl")
    
    include("Kinetics.jl")
    include("Transport.jl")
end
