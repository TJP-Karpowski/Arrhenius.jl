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


# molar masses in g/mol (same numbers as kg/kmol)
const ELEMENT_MW = Dict{String,Float64}(
    "H"  => 1.00794,
    "He" => 4.002602,
    "C"  => 12.0107,
    "N"  => 14.0067,
    "O"  => 15.9994,
    "F"  => 18.9984032,
    "Ne" => 20.1797,
    "Ar" => 39.948,
    "Cl" => 35.453,
    "S"  => 32.065,
    "Si" => 28.0855,
    # add more here if your mechanism needs them
)

