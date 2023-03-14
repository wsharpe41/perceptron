import numpy as np
# updated from Valerio Velardo's code for multi-layer perceptron by John R Williams


def MLP(num_inputs=3, hidden_layers=[3, 3], num_outputs=2):
    # A Multilayer Perceptron function.
    # Constructor for the MLP. Takes the number of inputs,
    # a variable number of hidden layers, and number of outputs
    # Args:
    # num_inputs (int): Number of inputs
    # hidden_layers (list): A list of ints for the hidden layers
    # num_outputs (int): Number of outputs

    # create a generic representation of the layers
    layers = [num_inputs] + hidden_layers + [num_outputs]
    print(f"layers: {layers}")
    # create random connection weights for the layers
    weights = []
    for i in range(len(layers)-1):
        # matrix of random weights between layers i and i+1
        w = np.random.rand(layers[i], layers[i+1])
        print(w.shape)   # this will be a 3,3 or 3,2 matrix
        weights.append(w)
    return layers, weights


def forward_propagate(inputs, weights):
    # Computes forward propagation of the network based on input signals.
    # Args:
    #    inputs (ndarray): Input signals
    #    weights (list): Weights of the network
    # Returns:
    #   activations (ndarray): Output values

    # the input layer activation is just the input itself
    activations = inputs

    # iterate through the network layers
    for w in weights:

        # calculate matrix multiplication between previous activation and weight matrix
        net_inputs = np.dot(activations, w)

        # apply sigmoid activation function
        activations = sigmoid(net_inputs)

    # return output layer activation
    return activations


def sigmoid(x):
    # Sigmoid activation function
    # Args:
    #   x (float): Value to be processed
    # Returns:
    #   y (float): Output

    y = 1.0 / (1 + np.exp(-x))
    return y


# create a Multilayer Perceptron
num_inputs = 3
hidden_layers = [3, 3]
num_outputs = 2
layers, weights = MLP(num_inputs, hidden_layers, num_outputs)
print(f"weights: {weights}")
# set random values for network's input
inputs = np.random.rand(num_inputs)

# perform forward propagation
output = forward_propagate(inputs, weights)

print("Network activation: {}".format(output))
