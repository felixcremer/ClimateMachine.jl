# # Generalized Minimal Residual (GMRES)
# In this tutorial we describe the basics of using the gmres iterative solver
# At the end you should be able to
# 1. Use GMRES to solve a linear system
# 2. Know when to not use it
# 3. Contruct a column-wise linear solver with IGMRES

# ## What is it?
# GMRES is a Krylov subspace method for solving linear systems:
# ```math
#  Ax = b
# ```
# !!! warning
#     The method can be quite wasteful with memory if too many iterations are required.
# See the [wikipedia](https://en.wikipedia.org/wiki/Generalized_minimal_residual_method) for more details.
