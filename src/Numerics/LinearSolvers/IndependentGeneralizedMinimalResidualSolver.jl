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
struct IndGenMinRes{FT, IT, VT, AT, TT1, TT2} <: LS.AbstractIterativeLinearSolver
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
    R::AT
    reshape_tuple_f::TT1
    permute_tuple_f::TT1
    reshape_tuple_b::TT2
    permute_tuple_b::TT2
end

# So that the struct can be passed into kernels
Adapt.adapt_structure(to, x::IndGenMinRes) = IndGenMinRes(x.atol, x.rtol, x.m, x.n, x.k_n, adapt(to, x.residual), adapt(to, x.b), adapt(to, x.x),  adapt(to, x.sol), adapt(to, x.rhs), adapt(to, x.cs),  adapt(to, x.Q),  adapt(to, x.H), adapt(to, x.R), reshape_tuple_f, permute_tuple_f, reshape_tuple_b, permute_tuple_b)

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
    return false, zero(eltype(Q))
end

# iteration function (2)
function LS.doiteration!(linearoperator!, Q, Qrhs, gmres::IndGenMinRes, threshold, args...)
    # initialize gmres.x
    convert_structure!(gmres.x, Q, gmres.reshape_tuple_f, gmres.permute_tuple_f)
    # apply linear operator to construct residual
    linearoperator!(Q, Qrhs, args...)
    r_vector = Qrhs .- Q
    # The following ar and rr are technically not correct in general cases
    ar = norm(r_vector)
    rr = norm(r_vector) / norm(Qrhs)
    # check if the initial guess is fantastic
    if (ar < gmres.atol) || (rr < gmres.rtol)
        return true, 0, atol
    end
    # initialize gmres.b
    convert_structure!(gmres.b, r_vector, gmres.reshape_tuple_f, gmres.permute_tuple_f)
    # apply linear operator to construct second krylov vector
    linearoperator!(Q, r_vector, args...)
    # initialize gmres.sol
    convert_structure!(gmres.sol, Q, gmres.reshape_tuple_f, gmres.permute_tuple_f)
    # initialize the rest of gmres
    event = initialize_gmres!(gmres)
    wait(event)
    ar, rr = compute_residuals(gmres, 1)
    # check if converged
    if (ar < gmres.atol) || (rr < gmres.rtol)
        event = construct_solution!(iterations, gmres)
        wait(event)
        convert_structure!(x, gmres.x, reshape_tuple_b, permute_tuple_b)
        return true, 1, atol
    end
    # body of iteration
    @inbounds for i in 2:gmres.k_n
        convert_structure!(r_vector, view(gmres.Q[:, i, :]), gmres.reshape_tuple_b, gmres.permute_tuple_b)
        linear_operator!(Q, r_vector)
        convert_structure!(gmres.sol, Q, gmres.reshape_tuple_f, gmres.permute_tuple_f)
        event = gmres_update!(i, gmres)
        wait(event)
        ar, rr = compute_residuals(gmres, i)
        # check if converged
        if (ar < gmres.atol) || (rr < gmres.rtol)
            event = construct_solution!(iterations, gmres)
            wait(event)
            convert_structure!(x, gmres.x, reshape_tuple_b, permute_tuple_b)
            return true, i, atol
        end
    end
    event = construct_solution!(iterations, gmres)
    wait(event)
    convert_structure!(x, gmres.x, reshape_tuple_b, permute_tuple_b)
    return Bool, Int, Float
end

# The function(s) that probably needs the most help
"""
function convert_structure!(x, y, reshape_tuple, permute_tuple)

# Description
Computes a tensor transpose and stores result in x
- This needs to be improved!

# Arguments
- `x`: (array) [OVERWRITTEN]. target destination for storing the y data
- `y`: (array). data that we want to copy
- `reshape_tuple`: (tuple) reshapes y to be like that of x, up to a permutation
- `permute_tuple`: (tuple) permutes the reshaped array into the correct structure

# Keyword Arguments
- `convert`: (bool). decides whether or not permute and convert. The default is true

# Return
nothing

# Comment
A naive kernel version of this operation is too slow
"""
@inline function convert_structure!(x, y, reshape_tuple, permute_tuple; convert = true)
    if convert
        alias_y = reshape(y, reshape_tuple)
        permute_y = permutedims(alias_y, permute_tuple)
        x[:] .= permute_y[:]
    end
    return nothing
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
function initialize_gmres!(gmres::IndGenMinRes; ndrange = gmres.n, cpu_threads = Threads.nthreads(), gpu_threads = 256)
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

# Helper Functions

"""
initialize_arnoldi!(g, I)

# Description
- First step of Arnoldi Iteration is to define first Krylov vector. Additionally sets things equal to zero

# Arguments
- `g`: (struct) [OVERWRITTEN] the gmres struct
- `I`: (int) thread index

# Return
nothing
"""
@inline function initialize_arnoldi!(gmres, I)
    # set (almost) everything to zero to be sure
    # the assumption is that gmres.k_n is small enough
    # to where these loops don't matter that much
    ft_zero = zero(eltype(gmres.H)) # float type zero

    @inbounds for i in 1:(gmres.k_n + 1)
        gmres.rhs[i, I] = ft_zero
        @inbounds for j in 1:gmres.k_n
            gmres.R[i,j,I] = ft_zero
            gmres.H[i,j,I] = ft_zero
        end
    end
    # gmres.x was initialized as the initial x
    # gmres.sol was initialized right before this function call
    # gmres.b was initialized right before this function call
    # compute norm
    @inbounds for i in 1:gmres.m
        gmres.rhs[1, I] += gmres.b[i, I] * gmres.b[i, I]
    end
    gmres.rhs[1, I] = sqrt(gmres.rhs[1, I])
    # now start computations
    @inbounds for i in 1:gmres.m
        gmres.sol[i, I] /= gmres.rhs[1, I]
        gmres.Q[i, 1, I] = gmres.b[i, I] / gmres.rhs[1, I] # First Krylov vector
    end
    return nothing
end

"""
initialize_QR!(gmres::IndGenMinRes, I)

# Description
initializes the QR decomposition of the UpperHesenberg Matrix

# Arguments
- `gmres`: (struct) [OVERWRITTEN] the gmres struct
- `I`: (int) thread index

# Return
nothing
"""
@inline function initialize_QR!(gmres, I)
    gmres.cs[1, I] = gmres.H[1,1, I]
    gmres.cs[2, I] = gmres.H[2,1, I]
    gmres.R[1, 1, I] = sqrt(gmres.cs[1, I]^2 + gmres.cs[2, I]^2)
    gmres.cs[1, I] /= gmres.R[1,1, I]
    gmres.cs[2, I] /= -gmres.R[1,1, I]
    return nothing
end

# The meat of gmres with updates that leverage information from the previous iteration
"""
update_arnoldi!(n, gmres, I)
# Description
Perform an Arnoldi iteration update

# Arguments
- `n`: current iteration number
- `gmres`: gmres struct that gets overwritten
- `I`: (int) thread index
# Return
- nothing
# linear_operator! Arguments
- `linear_operator!(x,y)`
# Description
- Performs Linear operation on vector and overwrites it
# Arguments
- `y`: (array)
# Return
nothing

"""
@inline function update_arnoldi!(n, gmres, I)
    # make new Krylov Vector orthogonal to previous ones
    @inbounds for j in 1:n
        gmres.H[j, n, I] = 0
        # dot products
        @inbounds for i in 1:gmres.m
            gmres.H[j, n, I] += gmres.Q[i, j, I] * gmres.sol[i, I]
        end
        # orthogonalize latest Krylov Vector
        @inbounds for i in 1:gmres.m
            gmres.sol[i, I] -= gmres.H[j, n, I] * gmres.Q[i,j, I]
        end
    end
    norm_q = 0.0
    @inbounds for i in 1:gmres.m
        norm_q += gmres.sol[i,I] * gmres.sol[i,I]
    end
    gmres.H[n+1, n, I] = sqrt(norm_q)
    @inbounds for i in 1:gmres.m
        gmres.Q[i, n+1, I] = gmres.sol[i, I] / gmres.H[n+1, n, I]
    end
    return nothing
end

"""
update_QR!(n, gmres, I)

# Description
Given a QR decomposition of the first n-1 columns of an upper hessenberg matrix, this computes the QR decomposition associated with the first n columns
# Arguments
- `gmres`: (struct) [OVERWRITTEN] the struct has factors that are updated
- `n`: (integer) column that needs to be updated
- `I`: (int) thread index
# Return
- nothing

# Comment
What is actually produced by the algorithm isn't the Q in the QR decomposition but rather Q^*. This is convenient since this is what is actually needed to solve the linear system
"""
@inline function update_QR!(n, gmres, I)
    # Apply previous Q to new column
    @inbounds for i in 1:n
        gmres.R[i, n, I] = gmres.H[i, n, I]
    end
    # apply rotation
    @inbounds for i in 1:n-1
        tmp1 = gmres.cs[1 + 2*(i-1), I] * gmres.R[i, n, I] - gmres.cs[2*i, I] * gmres.R[i+1, n, I]
        gmres.R[i+1, n, I] = gmres.cs[2*i, I] * gmres.R[i, n, I] + gmres.cs[1 + 2*(i-1), I] * gmres.R[i+1, n, I]
        gmres.R[i, n, I] = tmp1
    end
    # Now update, cs and R
    gmres.cs[1+2*(n-1), I] = gmres.R[n, n, I]
    gmres.cs[2*n, I] = gmres.H[n+1,n, I]
    gmres.R[n, n, I] = sqrt(gmres.cs[1+2*(n-1), I]^2 + gmres.cs[2*n, I]^2)
    gmres.cs[1+2*(n-1), I] /= gmres.R[n, n, I]
    gmres.cs[2*n, I] /= -gmres.R[n, n, I]
    return nothing
end

"""
solve_optimization!(iteration, gmres, I)

# Description
Solves the optimization problem in GMRES
# Arguments
- `iteration`: (int) current iteration number
- `gmres`: (struct) [OVERWRITTEN]
- `I`: (int) thread index
# Return
nothing
"""
@inline function solve_optimization!(n, gmres, I)
    # just need to update rhs from previous iteration
    # apply latest gibbs rotation
    tmp1 = gmres.cs[1 + 2*(n-1), I] * gmres.rhs[n, I] - gmres.cs[2*n, I] * gmres.rhs[n+1, I]
    gmres.rhs[n+1, I] = gmres.cs[2*n, I] * gmres.rhs[n, I] + gmres.cs[1 + 2*(n-1), I] * gmres.rhs[n+1, I]
    gmres.rhs[n, I] = tmp1
    # gmres.rhs[iteration+1] is the residual. Technically convergence should be checked here.
    gmres.residual[n, I] = abs.(gmres.rhs[n+1, I])
    # copy for performing the backsolve and saving gmres.rhs
    @inbounds for i in 1:n
        gmres.sol[i, I] = gmres.rhs[i, I]
    end
    # do the backsolve
    @inbounds for i in n:-1:1
        gmres.sol[i, I] /= gmres.R[i,i, I]
        @inbounds for j in 1:i-1
            gmres.sol[j, I] -= gmres.R[j,i, I] * gmres.sol[i, I]
        end
    end
    return nothing
end

"""
compute_residuals(gmres)

# Description
Compute atol and rtol of current iteration

# Arguments
- `gmres`: (struct)
- `i`: (current iteration)

# Return
- `atol`: (float) absolute tolerance
- `rtol`: (float) relative tolerance
"""
function compute_residuals(gmres, i)
    atol = maximum(gmres.residual[i])
    rtol = maximum(gmres.residual[i] ./ norm(gmres.R[:, 1]))
    return atol, rtol
end

end # end of module
