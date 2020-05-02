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
gmres = IndGenMinRes(b, ArrayType = ArrayType)
for i in 1:ni
    x[:,i] = A[:, :, i] \ b[:, i]
end
sol = copy(x)
x += ArrayType(randn(n,ni) * 0.01 * maximum(abs.(x)))
y = copy(x)
linear_operator! = closure_linear_operator_multi!(A, size(A)...)
linear_operator!(x, y)
iters = linearsolve!(linear_operator!, gmres, x, b; max_iters = length(b[:, 1]))
display(gmres.residual[79,:])

###
# MPISTateArray test
n1 = 4
n2 = 4
n3 = 4
mpi_b = MPIStateArray{T}(mpicomm, ArrayType, n1, n2, n3)
mpi_x = MPIStateArray{T}(mpicomm, ArrayType, n1, n2, n3)
mpi_A = ArrayType(randn(n1*n2, n1*n2, n3))

mpi_b.data[:] .= ArrayType(randn(n1 * n2 * n3))
mpi_x.data[:] .= ArrayType(randn(n1 * n2 * n3))

reshape_tuple_f = (Int(n1/2), Int(n1/2), n2, n3)
permute_tuple_f = (3, 1, 2, 4) #flip the indices
reshape_tuple_b = (n2, Int(n1/2), Int(n1/2), n3)
perpute_tuple_b = (3, 1, 2, 4)
