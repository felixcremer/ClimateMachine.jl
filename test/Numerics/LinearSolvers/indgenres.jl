using MPI
using Test
using LinearAlgebra
using Random
using StaticArrays
using CLIMA
using CLIMA.LinearSolvers
using CLIMA.IndGenMinResSolver
using CLIMA.MPIStateArrays
using CUDAapi
using Random
using KernelAbstractions
using CuArrays
Random.seed!(1235)


CLIMA.init();
ArrayType = CLIMA.array_type() #CHANGE ME FOR GPUS!
T = Float64
mpicomm = MPI.COMM_WORLD

@kernel function multiply_A_kernel!(x, A, y, n1, n2)
    I = @index(Global)
    for i in 1:n1
        tmp = zero(eltype(x))
        for j in 1:n2
            tmp += A[i, j, I] * y[j, I]
        end
        x[i, I] = tmp
    end
end

function multiply_by_A!(x, A, y, n1, n2; ndrange = size(x[1,:]), cpu_threads = Threads.nthreads(), gpu_threads = 256)
    if isa(x,Array)
        kernel! = multiply_A_kernel!(CPU(), cpu_threads)
    else
        kernel! = multiply_A_kernel!(CUDA(), gpu_threads)
    end
    return kernel!(x, A, y, n1, n2, ndrange = ndrange)
end
# for defining linear_operator
function closure_linear_operator_multi!(A, n1, n2, n3)
    function linear_operator!(x, y)
        event = multiply_by_A!(x, A, y, n1, n2, ndrange = n3)
        wait(event)
        return nothing
    end
end

n  = 100  # size of vector space
ni = 10 # number of independent linear solves
b = ArrayType(randn(n, ni)) # rhs
x = ArrayType(randn(n, ni)) # initial guess
A = ArrayType(randn((n, n, ni)) ./ sqrt(n) .* 1.0)
for i in 1:n
    A[i,i,:] .+= 1.0
end
ss = size(b)[1]
gmres = IndGenMinRes(b, ArrayType = ArrayType, subspace_size = ss)
for i in 1:ni
    x[:,i] = A[:, :, i] \ b[:, i]
end
sol = copy(x)
x += ArrayType(randn(n,ni) * 0.01 * maximum(abs.(x)))
y = copy(x)
linear_operator! = closure_linear_operator_multi!(A, size(A)...)
iters = linearsolve!(linear_operator!, gmres, x, b; max_iters = ss)
display(gmres.residual[79,:])
linear_operator!(y, x)
norm(y-b)

###
# MPISTateArray test
Random.seed!(1235)
n1 = 3
n2 = 2
n3 = 1
mpi_b = MPIStateArray{T}(mpicomm, ArrayType, n1, n2, n3)
mpi_x = MPIStateArray{T}(mpicomm, ArrayType, n1, n2, n3)
mpi_A = ArrayType(randn(n1*n2, n1*n2, n3))

# need to make sure that mpi_b and mpi_x are reproducible
mpi_b.data[:] .= ArrayType(randn(n1 * n2 * n3))
mpi_x.data[:] .= ArrayType(randn(n1 * n2 * n3))
mpi_y = copy(mpi_x)
#=
reshape_tuple_f = (n1, n2, n3)
permute_tuple_f = (1, 2, 3)
reshape_tuple_b = (n1, n2, n3)
permute_tuple_b = (1, 2, 3)
=#

# for defining linear_operator
function closure_linear_operator_mpi!(A, n1, n2, n3)
    function linear_operator!(x, y)
        alias_x = reshape(x.data,(n1, n3))
        alias_y = reshape(y.data,(n1, n3))
        event = multiply_by_A!(alias_x, A, alias_y, n1, n2, ndrange = n3)
        wait(event)
        return nothing
    end
end

gmres = IndGenMinRes(mpi_b, ArrayType = ArrayType, m = n1*n2, n = n3)

# Now define the linear operator
linear_operator! = closure_linear_operator_mpi!(mpi_A, size(mpi_A)...)
# linear_operator!(mpi_x, mpi_b)
iters = linearsolve!(linear_operator!, gmres, mpi_x, mpi_b; max_iters = n1*n2)
linear_operator!(mpi_y, mpi_x)
norm(mpi_y - mpi_b)
sol = mpi_A[:,:,1] \ mpi_b[:,:, 1][:]
