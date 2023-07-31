#------------------------------------------------------
# 1D Wave Equation with Dirichlet Boundary conditions
#------------------------------------------------------

using NeuralPDE, Lux, Optimization, OptimizationOptimJL
import ModelingToolkit: Interval

@parameters t, x
@variables u(..)

Dxx = Differential(x)^2
Dtt = Differential(t)^2
Dt = Differential(t)

#--------------------------------------
# STEP 1: DEFINE THE EQUATION (2D PDE)
#--------------------------------------

# Equation


# Initial & Boundary Conditions


# Time and Space Domains


# Discretization


# Define PDE system


#------------------------------------------------
# STEP 2: CHOOSE a NEURAL NETWORK ARCHITECTURE
#------------------------------------------------

# Number of dimensions


# Number of inner nodes


# Multilayer-layer perceptron (3 layers)


# Discretizer


#----------------------------------------------------------
# STEP 3: Convert PDE System into an OPTIMIZATION PROBLEM
#----------------------------------------------------------

# Convert the PDE system into an Optimization Problem using 'discretize'


# Call back function


#------------------------------------------------
# STEP 4: SOLVE OPTIMIZATION PROBLEM
#------------------------------------------------

# Select an Optimizer: BFGS Algorithm


# Solve the Optimization Problem


#------------------------------------------------
# PLOTTING & COMPARISON TO ANALYTICAL SOLUTION
#------------------------------------------------

