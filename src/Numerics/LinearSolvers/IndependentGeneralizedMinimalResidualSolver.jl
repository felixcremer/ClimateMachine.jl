module IndGenMinResSolver

export IndGenMinRes

using ..LinearSolvers
const LS = LinearSolvers

# struct
struct IndGenMinRes{FT} <: LS.AbstractIterativeLinearSolver
    # minimum
    rtol::FT
    atol::FT
    # Add more structure if necessary
end

# constructor
function IndGenMinRes(args...)
    # body of constructor
    return IndGenMinRes(contructor_args...)
end

# initialize function (1)
function LS.initialize!(linearoperator!, Q, Qrhs, solver::IndGenMinRes, args...)
    # body of initialize function in abstract iterative solver
    return Bool, Int
end

# iteration function (2)
function LS.doiteration!(linearoperator!, Q, Qrhs, solver::IndGenMinRes, threshold, args...)
    # body of iteration
    return Bool, Int, Float
end

end # end of module
