#---------------------------------------------
#  Hamiltonian Neural Networks
#        1D Frictionless Spring-Mass System
#---------------------------------------------

using Flux, DiffEqFlux, DifferentialEquations, Statistics, Plots, ReverseDiff

π_32 = Float32(π)

#----------------------
# DATA GENERATION
#----------------------

# Create a sequence including 1024 elements ranging from 0 to 1 


# INPUTS: q-vector (position) & p-vector (momentum)


# TARGETS: Derivatives of p & q (exact differentiation)


# DATASET: Create and Patch the Dataset ( Input Data (p & q) & Target Data (derivatives of p & q) )



#----------------------
# SETUP
#----------------------

# Construct a Hamiltonian NN
#       model: 2-layer neural network  (multi-layer perceptron) - Activation function: Rectified Linear Unit
# Dimensions


#number of nodes


# Create the Hamiltonian network


# The initial parameters of the neural network (in this case: p = nothing)


# Select ADAM optimization algorithm with learning rate of 0.01


# Define Loss function: mean squared error (MSE)
# hnn(x,p) predicts the output (y) given input x --> loss = mean((predicted - Target)^2)



#-------------------------------------------
# TRAINING THE HAMILTONIAN NEURAL NETWORK
#-------------------------------------------

# Use dataset in dataloader to train the HNN






#-----------------------------------------------------------
# SOLVING THE ODE USING TRAINED HAMILTONIAN NEURAL NETWORK
#-----------------------------------------------------------

# Contructs a Neural Hamiltonian DE Layer for solving Hamiltonian Problems



# Array of predicted results




#--------------------------------------
# PLOTTING
#--------------------------------------

