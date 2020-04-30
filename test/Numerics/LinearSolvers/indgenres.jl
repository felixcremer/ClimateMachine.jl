using MPI
using Test
using LinearAlgebra
using Random
using GPUifyLoops, StaticArrays
using CLIMA
using CLIMA.LinearSolvers
using CLIMA.IndGenMinResSolver
using CLIMA.MPIStateArrays
using CUDAapi
using Random
Random.seed!(1235)
#=
let
    CLIMA.init()
    mpicomm = MPI.COMM_WORLD
    ArrayType = CLIMA.array_type()
    device = ArrayType == Array ? CPU() : CUDA()
    n = 100
    T = Float64
    A = rand(n, n)
    scale = 2.0
    ϵ = 0.1
    # Matrix 1
    A = A' * A .* ϵ + scale * I

    # Matrix 2
    # A = Diagonal(collect(1:n) * 1.0)
    positive_definite = minimum(eigvals(A)) > eps(1.0)
    @test positive_definite

    b = ones(n) * 1.0
    mulbyA!(y, x) = (y .= A * x)

    tol = sqrt(eps(T))
    method(b, tol) = ConjugateGradient(b, max_iter = n)
    linearsolver = method(b, tol)

    x = ones(n) * 1.0
    x0 = copy(x)
    iters = linearsolve!(mulbyA!, linearsolver, x, b; max_iters = Inf)
    exact = A \ b
    x0 = copy(x)
end
=#
