module IndGenMinResSolver

export IndGenMinRes

using ..LinearSolvers
const LS = LinearSolvers
using Adapt, KernelAbstractions, LinearAlgebra

# struct
"""
# Description

Launches n independent GMRES solves

# Members

- atol::FT (float) absolute tolerance
- rtol::FT (float) relative tolerance
- m::IT (int) size of vector in each independent instance
- n::IT (int) number of independent GMRES
- k_n::IT (int) Krylov Dimension for each GMRES. It is also the number of GMRES iterations before nuking the subspace
- residual::VT (vector) residual values for each independent linear solve
- b::VT (vector) permutation of the rhs. probably can be removed if memory is an issue
- x::VT (vector) permutation of the initial guess. probably can be removed if memory is an issue
- sol::VT (vector) solution vector, it is used twice. First to represent Aqⁿ (the latest Krylov vector without being normalized), the second to represent the solution to the linear system
- rhs::VT (vector) rhs vector.
- cs::VT (vector) Sequence of Gibbs Rotation matrices in compact form. This is implicitly the Qᵀ of the QR factorization of the upper hessenberg matrix H.
- H::AT (array) Upper Hessenberg Matrix. A factor of two in memory can be saved here.
- Q::AT (array) Orthonormalized Krylov Subspace
- R::AT (array) The R of the QR factorization of the UpperHessenberg matrix H. A factor of 2 or so in memory can be saved here
- reshape_tuple_f::TT1 (tuple), reshapes structure of array that plays nice with the linear operator to a format compatible with struct
- permute_tuple_f::TT1 (tuple). forward permute tuple. permutes structure of array that plays nice with the linear operator to a format compatible with struct
- reshape_tuple_b::TT2 (tuple). reshapes structure of array that plays nice with struct to play nice with the linear operator
- permute_tuple_b::TT2 (tuple). backward permute tuple. permutes structure of array that plays nice with struct to play nice with the linear operator

# Intended Use
Solving n linear systems iteratively

# Comments on Improvement
- Allocates all the memory at once: Could improve to something more dynamic
- Too much memory in H and R struct: Could use a sparse representation to cut memory use in half (or more)
- Needs to perform a transpose of original data structure into current data structure: Could perhaps do a transpose free version, but the code gets a bit clunkier and the memory would no longer be coalesced for the heavy operations
"""
struct IndGenMinRes{FT, IT, VT, AT, TT1, TT21} <: LS.AbstractIterativeLinearSolver
    atol::FT
    rtol::FT
    m::IT
    n::IT
    k_n::IT
    residual::VT
    b::VT
    x::VT
    sol::VT
    rhs::VT
    cs::VT
    Q::AT
    H::AT
    R::AT2
    reshape_tuple_f::TT1
    permute_tuple_f::TT1
    reshape_tuple_b::TT2
    permute_tuple_b::TT2
end

# So that the struct can be passed into kernels
Adapt.adapt_structure(to, x::ParallelGMRES) = ParallelGMRES(x.atol, x.rtol, x.m, x.n, x.k_n, adapt(to, x.residual), adapt(to, x.b), adapt(to, x.x),  adapt(to, x.sol), adapt(to, x.rhs), adapt(to, x.cs),  adapt(to, x.Q),  adapt(to, x.H), adapt(to, x.R), reshape_tuple_f, permute_tuple_f, reshape_tuple_b, permute_tuple_b)

"""
IndGenMinRes(Qrhs; m = length(Qrhs[:,1]), n = length(Qrhs[1,:]), subspace_size = m, atol = sqrt(eps(eltype(Qrhs))), rtol = sqrt(eps(eltype(Qrhs))), ArrayType = Array, reshape_tuple_f = size(Qrhs), permute_tuple_f = Tuple(1:length(size(Qrhs))), reshape_tuple_b = size(Qrhs), permute_tuple_b = Tuple(1:length(size(Qrhs))))

# Description
Generic constructor for IndGenMinRes

# Arguments
- `Qrhs`: (array) Array structure that linear_operator! acts on

# Keyword Arguments
- `m`: (int) size of vector space for each independent linear solve. This is assumed to be the same for each and every linear solve. DEFAULT = length(Qrhs[:,1])
- `n`: (int) number of independent linear solves, DEFAULT = length(Qrhs[1,:])
- `atol`: (float) absolute tolerance. DEFAULT = sqrt(eps(eltype(Qrhs)))
- `rtol`: (float) relative tolerance. DEFAULT = sqrt(eps(eltype(Qrhs)))
- `ArrayType`: (type). used for either using CuArrays or Arrays. DEFAULT = Array
- `reshape_tuple_f`: (tuple). used in the wrapper function for flexibility. DEFAULT = size(Qrhs). this means don't do anything
- `permute_tuple_f`: (tuple). used in the wrapper function for flexibility. DEFAULT = Tuple(1:length(size(Qrhs))). this means, don't do anything.
- `reshape_tuple_b`: (tuple). used in the wrapper function for flexibility. DEFAULT = size(Qrhs). this means don't do anything
- `permute_tuple_b`: (tuple). used in the wrapper function for flexibility. DEFAULT = Tuple(1:length(size(Qrhs))). this means, don't do anything.

# Return
instance of IndGenMinRes struct
"""
function IndGenMinRes(Qrhs; m = length(Qrhs[:,1]), n = length(Qrhs[1,:]), subspace_size = m, atol = sqrt(eps(eltype(Qrhs))), rtol = sqrt(eps(eltype(Qrhs))), ArrayType = Array, reshape_tuple_f = size(Qrhs), permute_tuple_f = Tuple(1:length(size(Qrhs))), reshape_tuple_b = size(Qrhs), permute_tuple_b = Tuple(1:length(size(Qrhs))))
    k_n = subspace_size
    residual = ArrayType(zeros(eltype(Qrhs), (k_n, n)))
    b = ArrayType(zeros(eltype(Qrhs), (m, n)))
    x = ArrayType(zeros(eltype(Qrhs), (m, n)))
    sol = ArrayType(zeros(eltype(Qrhs), (m, n)))
    rhs = ArrayType(zeros(eltype(Qrhs), (k_n + 1, n)))
    cs = ArrayType(zeros(eltype(Qrhs), (2 * k_n, n)))
    Q = ArrayType(zeros(eltype(Qrhs), (m, k_n+1 , n)))
    H = ArrayType(zeros(eltype(Qrhs), (k_n+1, k_n, n)))
    R  = ArrayType(zeros(eltype(Qrhs), (k_n+1, k_n, n)))
    return IndGenMinRes(atol, rtol, m, n, k_n, residual, b, x, sol, rhs, cs, Q, H, R, reshape_tuple_f, permute_tuple_f, reshape_tuple_b, permute_tuple_b)
end

# initialize function (1)
function LS.initialize!(linearoperator!, Q, Qrhs, solver::IndGenMinRes, args...)
    # body of initialize function in abstract iterative solver
    return false, 0
end

# iteration function (2)
function LS.doiteration!(linearoperator!, Q, Qrhs, solver::IndGenMinRes, threshold, args...)
    # body of iteration
    return Bool, Int, Float
end

# Kernels
"""
initialize_gmres_kernel!(gmres)

# Description
Initializes the gmres struct by calling
- initialize_arnoldi
- initialize_QR!
- update_arnoldi!
- update_QR!
- solve_optimization!
It is assumed that the first two krylov vectors are already constructed

# Arguments
- `gmres`: (struct) gmres struct

# Return
(implicitly) kernel abstractions function closure
"""
# m, n, k_n, residual, b, x, sol, rhs, cs, Q, H, R
@kernel function initialize_gmres_kernel!(gmres)
    I = @index(Global)
    initialize_arnoldi!(gmres, I)
    update_arnoldi!(1, gmres, I)
    initialize_QR!(gmres, I)
    update_QR!(1, gmres, I)
    solve_optimization!(1, gmres, I)
end

"""
gmres_update_kernel!(i, gmres, I)

# Description
kernel that calls
- update_arnoldi!
- update_QR!
- solve_optimization!
Which is the heart of the gmres algorithm

# Arguments
- `i`: (int) interation index
- `gmres`: (struct) gmres struct
- `I`: (int) thread index

# Return
kernel object from KernelAbstractions
"""
@kernel function gmres_update_kernel!(i, gmres)
    I = @index(Global)
    update_arnoldi!(i, gmres, I)
    update_QR!(i, gmres, I)
    solve_optimization!(i, gmres, I)
end

"""
construct_solution_kernel!(i, gmres)

# Description
given step i of the gmres iteration, constructs the "best" solution of the linear system for the given Krylov subspace

# Arguments
- `i`: (int) gmres iteration
- `gmres`: (struct) gmres struct

# Return
kernel object from KernelAbstractions
"""
@kernel function construct_solution_kernel!(i, gmres)
    M, I = @index(Global, NTuple)
    tmp = zero(eltype(gmres.b))
    @inbounds for j in 1:i
        tmp += gmres.Q[M, j, I] *  gmres.sol[j, I]
    end
    gmres.x[M , I] += tmp # since previously gmres.x held the initial value
end

# Configuration for Kernels
"""
initialize_gmres!(gmres; ndrange = gmres.n, cpu_threads = Threads.nthreads(), gpu_threads = 256)

# Description
Uses the initialize_gmres_kernel! for initalizing

# Arguments
- `gmres`: (struct) [OVERWRITTEN]

# Keyword Arguments
- `ndrange`: (int) or (tuple) thread structure to iterate over
- `cpu_threads`: (int) number of cpu threads. default = Threads.nthreads()
- `gpu_threads`: (int) number of gpu threads. default = 256

# Return
event. A KernelAbstractions object
"""
function initialize_gmres!(gmres::ParallelGMRES; ndrange = gmres.n, cpu_threads = Threads.nthreads(), gpu_threads = 256)
    if isa(gmres.b, Array)
        kernel! = initialize_gmres_kernel!(CPU(), cpu_threads)
    else
        kernel! = initialize_gmres_kernel!(CUDA(), gpu_threads)
    end
    event = kernel!(gmres, ndrange = ndrange)
    return event
end

"""
gmres_update!(i, gmres; ndrange = gmres.n, cpu_threads = Threads.nthreads(), gpu_threads = 256)

# Description
Calls the gmres_update_kernel!

# Arguments
- `i`: (int) iteration number
- `gmres`: (struct) gmres struct

# Keyword Arguments
- `ndrange`: (int) or (tuple) thread structure to iterate over
- `cpu_threads`: (int) number of cpu threads. default = Threads.nthreads()
- `gpu_threads`: (int) number of gpu threads. default = 256

# Return
event. A KernelAbstractions object
"""
function gmres_update!(i, gmres; ndrange = gmres.n, cpu_threads = Threads.nthreads(), gpu_threads = 256)
    if isa(gmres.b, Array)
        kernel! = gmres_update_kernel!(CPU(), cpu_threads)
    else
        kernel! = gmres_update_kernel!(CUDA(), gpu_threads)
    end
    event = kernel!(i, gmres, ndrange = ndrange)
    return event
end

"""
construct_solution!(i, gmres; ndrange = size(gmres.x), cpu_threads = Threads.nthreads(), gpu_threads = 256)

# Description
Calls construct_solution_kernel! for constructing the solution

# Arguments
- `i`: (int) iteration number
- `gmres`: (struct) gmres struct

# Keyword Arguments
- `ndrange`: (int) or (tuple) thread structure to iterate over
- `cpu_threads`: (int) number of cpu threads. default = Threads.nthreads()
- `gpu_threads`: (int) number of gpu threads. default = 256

# Return
event. A KernelAbstractions object
"""
function construct_solution!(i, gmres; ndrange = size(gmres.x), cpu_threads = Threads.nthreads(), gpu_threads = 256)
    if isa(gmres.b, Array)
        kernel! = construct_solution_kernel!(CPU(), cpu_threads)
    else
        kernel! = construct_solution_kernel!(CUDA(), gpu_threads)
    end
    event = kernel!(i, gmres, ndrange = ndrange)
    return event
end

end # end of module
