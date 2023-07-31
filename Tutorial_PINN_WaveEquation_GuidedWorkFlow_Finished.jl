#------------------------------------------------------
# 1D Wave Equation with Dirichlet Boundary conditions
#------------------------------------------------------

using NeuralPDE, Lux, Optimization, OptimizationOptimJL
import ModelingToolkit: Interval

@parameters t,x
@variables u(..)
Dxx = Differential(x)^2
Dtt = Differential(t)^2
Dt  = Differential(t)

#--------------------------------------
# STEP 1: DEFINE THE EQUATION (2D PDE)
#--------------------------------------

# Equation
C = 1
eq = Dtt(u(t, x)) ~ C^2 * Dxx(u(t, x))

# Initial & Boundary Conditions
bcs = [u(t, 0) ~ 0.0,  # for all t > 0
       u(t, 1) ~ 0.0,  # for all t > 0
       u(0, x) ~ x * (1.0 - x), #for all 0 < x < 1
       Dt(u(0, x)) ~ 0.0]   #for all  0 < x < 1]

# Time and Space and Domains
domains = [t ∈ Interval(0.0, 1.0),
           x ∈ Interval(0.0, 1.0)]

# Discretization
dx = 0.1

# Define a PDE system
indvar = [t,x]
depvar = u(t,x)

@named pde_system = PDESystem(eq, bcs, domains,indvar,depvar)

#------------------------------------------------
# STEP 2: CHOOSE a NEURAL NETWORK ARCHITECTURE
#------------------------------------------------

# Number of dimensions
dims = length(domains)

# Number of inner nodes
n = 16

# Multilayer-layer perceptron (3 layers, sigmoid activation function)
chain = Lux.Chain(Dense(dims, n, Lux.σ), Dense(n, n, Lux.σ), Dense(n, 1))

# Discretizer
discretization = PhysicsInformedNN(chain, GridTraining(dx))

#----------------------------------------------------------
# STEP 3: Convert PDE System into an OPTIMIZATION PROBLEM
#----------------------------------------------------------

# Convert the PDE system into an Optimization Problem using 'discretize'
prob = discretize(pde_system, discretization)

#---
# Call back function
callback = function (p,l)
    println("Current loss is: $l")
    return false
end

#------------------------------------------------
# STEP 4: SOLVE OPTIMIZATION PROBLEM
#------------------------------------------------

# Select an Optimizer: BFGS Algorithm
opt = OptimizationOptimJL.BFGS()

# Solve the Optimization Problem
res = Optimization.solve(prob,opt; callback = callback, maxiters=1200)
phi = discretization.phi

#------------------------------------------------
# PLOTTING & COMPARISON TO ANALYTICAL SOLUTION
#------------------------------------------------

using Plots

ts, xs = [infimum(d.domain):dx:supremum(d.domain) for d in domains]
function analytic_sol_func(t, x)
    sum([(8 / (k^3 * pi^3)) * sin(k * pi * x) * cos(C * k * pi * t) for k in 1:2:50000])
end

u_predict = reshape([first(phi([t, x], res.u)) for t in ts for x in xs],
                    (length(ts), length(xs)))
u_real = reshape([analytic_sol_func(t, x) for t in ts for x in xs],
                 (length(ts), length(xs)))

diff_u = abs.(u_predict .- u_real)
p1 = plot(ts, xs, u_real, linetype = :contourf, title = "analytic");
p2 = plot(ts, xs, u_predict, linetype = :contourf, title = "predict");
p3 = plot(ts, xs, diff_u, linetype = :contourf, title = "error");
plot(p1, p2, p3)

#------------------------------------------------