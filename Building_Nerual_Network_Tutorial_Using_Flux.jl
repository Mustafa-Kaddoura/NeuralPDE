#---------------------------------------------------------------
# Tutorial on Building and Training a Feedforward Neural Network
#  Model Using Flux.jl in Julia
#---------------------------------------------------------------


using Flux, Statistics, MLDatasets, DataFrames, OneHotArrays


#---------------------------------------------------------------
# (1) DATASET:
# Import dataset from MLDatasets.jl
#---------------------------------------------------------------

# Download Iris dataset
Iris()

# Assin x,y names to the data
x,y = Iris(as_df=false)[:]

#Look at the dataset
y

x |> summary

# Our next step would be to convert this data into a form that can be fed to a machine learning model. 
# convert x values in Float32
x = Float32.(x);

# convert y into one hot encoded using OneHotArrays
# Hot encoding convert categorial data (red, blue, etc.) into data that can be fed into ML.
# Map the categorial data into integers
y = vec(y);
const classes = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"];
y_onehot = onehotbatch(y, classes)


#---------------------------------------------------------------
# (2) Model Building:
# Import dataset from MLDatasets.jl
#---------------------------------------------------------------
model = Chain(Dense(4=>3), softmax)

# Weights and biases are initialized by Flux. Look at their values
model[1].weight, model[1].bias


#---------------------------------------------------------------
# (3) Training:
#---------------------------------------------------------------

# Define a Loss Function
function loss(model, x, y)
    ŷ = model(x)
    Flux.logitbinarycrossentropy(ŷ, y)
end;

# Define an Accuracy Function
accuracy(x, y) = mean(Flux.onecold(model(x), classes) .== y);

#------------
# Check the loss function value before training
loss(model, x, y_onehot)

# Define Training Function
function train_model()
    dLdm, _, _ = gradient(loss, model, x, y_onehot)
    @. model[1].weight = model[1].weight - 0.1 * dLdm[:layers][1][:weight]
    @. model[1].bias   = model[1].bias   - 0.1 * dLdm[:layers][1][:bias]
end;

# Run a Training loop 
for i = 1:2000
    train_model();
    accuracy(x,y) >= 0.98 && break #Break when accuracy is >=0.98 to avoid over-training
end


# Check accuracy and loss values after training
#@show accuracy(x, y)

loss(model, x, y_onehot)

