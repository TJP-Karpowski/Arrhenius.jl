"""
    const R = 8314.46261815324 #J/(kmol*K)

Gas constant
"""
const R = 8314.46261815324 #J/(kmol*K)
export R


"""
    const one_atm = 101325.0 #pascal

One atm
"""
const one_atm = 101325.0 #pascal
export one_atm


# Molar masses in g/mol (same numbers as kg/kmol). Keep these aligned with
# Cantera's default atomic weights so mole-to-mass conversion parity holds.
const ELEMENT_MW = Dict{String,Float64}(
    "H"  => 1.008,
    "He" => 4.002602,
    "C"  => 12.011,
    "N"  => 14.007,
    "O"  => 15.999,
    "F"  => 18.9984032,
    "Ne" => 20.18,
    "Ar" => 39.95,
    "Cl" => 35.45,
    "S"  => 32.06,
    "Si" => 28.085,
    # add more here if your mechanism needs them
)
