import numpy as np
from random import random
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

    # save derivatives per layer
    derivatives = []
    for i in range(len(layers) - 1):
        d = np.zeros((layers[i], layers[i + 1]))
        derivatives.append(d)

    # save activations per layer
    activations = []
    for i in range(len(layers)):
        a = np.zeros(layers[i])
        activations.append(a)

    return layers, weights, derivatives, activations


def forward_propagate(inputs, weights, layers, activations):
    # Computes forward propagation of the network based on input signals.
    # Args:
    #    inputs (ndarray): Input signals
    #    weights (list): Weights of the network
    # Returns:
    #   activations (ndarray): Output values

    # the input layer activation is just the input itself
    local_activations = inputs

    # save the activations for backpropogation
    activations[0] = inputs

    # iterate through the network layers
    for i, w in enumerate(weights):
        # calculate matrix multiplication between previous activation and weight matrix
        net_inputs = np.dot(local_activations, w)

        # apply sigmoid activation function
        local_activations = sigmoid(net_inputs)

        # save the activations for backpropogation
        activations[i + 1] = local_activations

    # return output layer activation
    return local_activations


def sigmoid(x):
    # Sigmoid activation function
    # Args:
    #   x (float): Value to be processed
    # Returns:
    #   y (float): Output

    y = 1.0 / (1 + np.exp(-x))
    return y


def sigmoid_derivative(x):
    """Sigmoid derivative function
    Args:
        x (float): Value to be processed
    Returns:
        y (float): Output
    """
    return x * (1.0 - x)


def mse(target, output):
    """Mean Squared Error loss function
    Args:
        target (ndarray): The ground trut
        output (ndarray): The predicted values
    Returns:
        (float): Output
    """
    return np.average((target - output) ** 2)


def back_propagate(activations, derivatives, weights, error):
    """Backpropogates an error signal.
    Args:
        error (ndarray): The error to backprop.
    Returns:
        error (ndarray): The final error of the input
    """

    # iterate backwards through the network layers
    for i in reversed(range(len(derivatives))):

        # get activation for previous layer
        local_activations = activations[i+1]

        # apply sigmoid derivative function
        delta = error * sigmoid_derivative(local_activations)

        # reshape delta as to have it as a 2d array
        delta_re = delta.reshape(delta.shape[0], -1).T

        # get activations for current layer
        current_activations = activations[i]

        # reshape activations as to have them as a 2d column matrix
        current_activations = current_activations.reshape(
            current_activations.shape[0], -1)

        # save derivative after applying matrix multiplication
        derivatives[i] = np.dot(current_activations, delta_re)

        # backpropogate the next error
        error = np.dot(delta, weights[i].T)


def gradient_descent(weights, derivatives, learningRate=1):
    """Learns by descending the gradient
    Args:
        learningRate (float): How fast to learn.
    """
    # update the weights by stepping down the gradient
    for i in range(len(weights)):
        local_derivatives = derivatives[i]
        weights[i] += local_derivatives * learningRate

    return weights, derivatives


def train(inputs, targets, epochs, learning_rate, weights, layers, activations, derivatives):
    """Trains model running forward prop and backprop
    Args:
        inputs (ndarray): X
        targets (ndarray): Y
        epochs (int): Num. epochs we want to train the network for
        learning_rate (float): Step to apply to gradient descent
    """
    # now enter the training loop
    for i in range(epochs):
        sum_errors = 0

        # iterate through all the training data
        for j, input in enumerate(inputs):
            target = targets[j]

            # activate the network!
            output = forward_propagate(
                input, weights, layers, activations)

            error = target - output

            back_propagate(activations, derivatives, weights, error)

            # now perform gradient descent on the derivatives
            # (this will update the weights
            weights, derivatives = gradient_descent(
                weights, derivatives, learning_rate)

            # keep track of the MSE for reporting later
            sum_errors += mse(target, output)

        # Epoch complete, report the training error
        print("Error: {} at epoch {}".format(sum_errors/len(targets), i+1))

    print("Training complete!")
    print("=====")


def driver():

    # create a dataset to train a network for the sum operation
    items = np.array([[random()/2 for i in range(2)] for j in range(1000)])
    targets = np.array([[i[0] + i[1]] for i in items])

    # create a Multilayer Perceptron with one hidden layer
    layers, weights, derivatives, activations = MLP(2, [5], 1)

    # train network
    train(items, targets, 50, 0.1, weights, layers, activations, derivatives)

    # create dummy data
    input = np.array([0.3, 0.1])
    target = np.array([0.4])

    # get a prediction
    output = forward_propagate(
        input, weights, layers, activations)

    print()
    print(
        "Our network believes that {} + {} is equal to {}".format(input[0], input[1], output[0]))


driver()
