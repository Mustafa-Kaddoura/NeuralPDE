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
t = range(0.0f0, 1.0f0, length = 1024) 

# INPUTS: q-vector (position) & p-vector (momentum)
q_t = reshape(sin.(2π_32 * t), 1, :) # reshape into row vector
p_t = reshape(cos.(2π_32 * t), 1, :) # reshape into row vector

# TARGETS: Derivatives of p & q (exact differentiation)
dqdt = 2π_32 .* p_t
dpdt = -2π_32 .* q_t

# DATASET: Create and Patch the Dataset ( Input Data (p & q) & Target Data (derivatives of p & q) )
data = cat(q_t, p_t, dims = 1)
target = cat(dqdt, dpdt, dims = 1)

dataloader = Flux.Data.DataLoader((data, target); batchsize=256, shuffle=true)


#----------------------
# SETUP
#----------------------

# Construct a Hamiltonian NN
#       model: 2-layer neural network  (multi-layer perceptron) - Activation function: Rectified Linear Unit
# Dimensions
dims = 2
#number of nodes
n = 64
# Create the Hamiltonian network
hnn = HamiltonianNN(Chain(Dense(dims, n, relu), Dense(n, 1)))

# The initial parameters of the neural network (in this case: p = nothing)
p = hnn.p

# Select ADAM optimization algorithm with learning rate of 0.01
opt = ADAM(0.01)

# Define Loss function: mean squared error (MSE)
# hnn(x,p) predicts the output (y) given input x --> loss = mean((predicted - Target)^2)
loss(x, y, p) = mean((hnn(x, p) .- y) .^ 2) 
callback() = println("Loss Neural Hamiltonian DE = $(loss(data, target, p))")


#-------------------------------------------
# TRAINING THE HAMILTONIAN NEURAL NETWORK
#-------------------------------------------

# Use dataset in dataloader to train the HNN
epochs = 500
for epoch in 1:epochs
    # Loop over each input data (x) & target output (y) in dataloader (dataset)
    for (x, y) in dataloader 

        # Compute the gradients of the HNN layer for optimization
        gs = ReverseDiff.gradient(p -> loss(x, y, p), p) 

        # Optimize the network based on the computed gradient (gs) using the optimization algorithm defined in (opt)
        Flux.Optimise.update!(opt, p, gs)

    end
    if epoch % 100 == 1
        callback()
    end
end
callback()

#----- Now, we have a trained HNN that describes the mass-spring system

#-----------------------------------------------------------
# SOLVING THE ODE USING TRAINED HAMILTONIAN NEURAL NETWORK
#-----------------------------------------------------------

# Contructs a Neural Hamiltonian DE Layer for solving Hamiltonian Problems
model = NeuralHamiltonianDE( hnn, (0.0f0, 1.0f0), # The timespan to be solved on.
                             Tsit5(), # Tsit5() is a non-stiff Runge-Kutta method of Order 5
                             save_everystep = false, 
                             save_start = true, saveat = t)

# Array of predicted results
pred = Array(model(data[:, 1]))

#--------------------------------------
# PLOTTING
#--------------------------------------
plot(data[1, :], data[2, :], lw=4, label="Original")
plot!(pred[1, :], pred[2, :], lw=4, label="Predicted")
xlabel!("Position (q)")
ylabel!("Momentum (p)")