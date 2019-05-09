import numpy as np
from math import exp


class Layer:
    def __init__(self, n_neurons):
        self.n_neurons = n_neurons
        self.exit = np.zeros((1, n_neurons))
        self.error = np.zeros((1, n_neurons))


class InitialLayer(Layer):
    def __init__(self, n_neurons, next_layer):
        super(InitialLayer, self).__init__(n_neurons)
        self.next_layer = next_layer

    def propagation(self, entry):
        self.exit = np.matrix(entry)
        return self.next_layer.propagation()

    def error_calc(self): pass
    def backpropagation(self, momentum, learning_ratio): pass


class IntermidiateLayer(Layer):
    def __init__(self, n_neurons, previous_layer, next_layer):
        super(IntermidiateLayer, self).__init__(n_neurons)
        self.weights = np.random.rand(previous_layer.n_neurons, n_neurons)
        # self.weights = np.zeros((previous_layer.n_neurons, n_neurons))
        self.v = np.vectorize(self.transfer_function)
        self.previous_layer = previous_layer
        self.next_layer = next_layer

    def transfer_function(self, sum):
        return(1 / (1 + exp(-sum)))

    def propagation(self):
        self.exit = self.v(np.dot(self.previous_layer.exit, self.weights))
        return self.next_layer.propagation()

    def error_calc(self):
        self.error = np.multiply(np.multiply(self.exit,
                                             (1 - self.exit)), (np.dot(self.next_layer.error,
                                                                          np.transpose(self.next_layer.weights))))
        self.previous_layer.error_calc()

    def backpropagation(self, momentum, learning_ratio):
        self.weights = (self.weights * momentum) + (
            np.dot(np.transpose(self.previous_layer.exit), self.error) * learning_ratio)
        self.previous_layer.backpropagation(momentum, learning_ratio)

class ExitLayer(Layer):
    def __init__(self, n_neurons, previous_layer, threshold):
        super(ExitLayer, self).__init__(n_neurons)
        self.weights = np.random.rand(previous_layer.n_neurons, n_neurons)
        # self.weights = np.zeros((previous_layer.n_neurons, n_neurons))
        self.v = np.vectorize(self.transfer_function)
        self.previous_layer = previous_layer
        self.threshold = threshold

    def transfer_function(self, sum):
        # return (1 if 1 / (1 + exp(-sum)) > self.threshold else 0)
        return 1 / (1 + exp(-sum))

    def propagation(self):
        self.exit = self.v(np.dot(self.previous_layer.exit, self.weights))
        biggest_index = np.argmax(self.exit)
        simplified_exit = np.zeros((1,7))
        # print(self.exit)
        simplified_exit[0][biggest_index] = 1
        return simplified_exit
        # print("--------------------------")

    def error_calc(self, result):
        self.error = np.multiply(np.multiply(
            self.exit, (1 - self.exit)), (result - self.exit))
        self.previous_layer.error_calc()

    def backpropagation(self, momentum, learning_ratio):
        self.weights = (self.weights * momentum) + (
            np.dot(np.transpose(self.previous_layer.exit), self.error) * learning_ratio)
        self.previous_layer.backpropagation(momentum, learning_ratio)
