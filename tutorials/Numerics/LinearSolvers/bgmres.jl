# # Batched Generalized Minimal Residual
# In this tutorial we describe the basics of using the batched gmres iterative solver
# At the end you should be able to
# 1. Use BatchedGMRES to solve a linear system
# 2. Contruct a column-wise linear solver with BatchedGMRES

# ## What is the Generalized Minimal Residual Method?
# The  Generalized Minimal Residual Method (GMRES) is a [Krylov subspace](https://en.wikipedia.org/wiki/Krylov_subspace) method for solving linear systems:
# ```math
#  Ax = b
# ```
# See the [wikipedia](https://en.wikipedia.org/wiki/Generalized_minimal_residual_method) for more details.

# ## What is the Batched Generalized Minimal Residual Method?
# As the name suggests it solves a whole bunch of independent GMRES problems

# ## Basic Example
# First we must load a few things
using CLIMA, CLIMA.LinearSolvers, CLIMA.BatchedGeneralizedMinimalResidualSolver
using LinearAlgebra, Random

# Next we define two linear systems that we would like to solve simultaneously
# The matrix for the first linear system is
A1 = [
    2.0 -1.0 0.0
    -1.0 2.0 -1.0
    0.0 -1.0 2.0
];
# And the right hand side is
b1 = ones(typeof(1.0), 3);
# The exact solution to $A1 x1 = b1$ is
x1_exact = [1.5, 2.0, 1.5];
# The matrix for the first linear system is
A2 = [
    2.0 -1.0 0.0
    0.0 2.0 -1.0
    0.0 0.0 2.0
];
# And the right hand side is
b2 = ones(typeof(1.0), 3);
# The exact solution to $A2 x2 = b2$ is
x2_exact = [0.875, 0.75, 0.5];

# We now define a function that performs the action of each linear operator independently
function closure_linear_operator(A1, A2)
    function linear_operator!(x,y)
        mul!(view(x,:,1), A1, view(y,:,1))
        mul!(view(x,:,2), A2, view(y,:,2))
        return nothing
    end
    return linear_operator!
end;

# To understand how this works let us construct an instance
# of the linear operator and apply it to a vector
linear_operator! = closure_linear_operator(A1, A2);
# Let us see what the action of this linear operator is
y1 = ones(typeof(1.0), 3);
y2 = ones(typeof(1.0), 3) * 2.0;
y = [y1 y2];
x = copy(y);
linear_operator!(x,y);
display(x)
display([A1*y1 A2*y2])
# We see that the first column is A1 * [1 1 1]'
# and the second column is A2 * [2 2 2]'

# We are now ready to set up our BatchedGMRES solver


# Now we can set up the ConjugateGradient struct
linearsolver = ConjugateGradient(b, max_iter = 100);
.
# We need to define an initial guess for each linear system
x1 = ones(typeof(1.0), 3);
x2 = ones(typeof(1.0), 3);
x = [x1 x2]
# To solve the linear system we just need to pass to the linearsolve! function
iters = linearsolve!(linear_operator!, linearsolver, x, b)
