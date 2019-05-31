from layer import *


class NeuralNet:
    def __init__(self, n_entries, n_intermidiate_layers, intermediate_layer_size, n_exits):
        self.__initial_layer = InitialLayer(n_entries, None)
        self.__intermediate_layers = []
        self.__exit_layer = ExitLayer(n_exits, self.__initial_layer)
        self.__initial_layer.next_layer = self.__exit_layer

        last_layer = self.__initial_layer

        for layer in range(n_intermidiate_layers):
            self.__intermediate_layers.append(IntermidiateLayer(intermediate_layer_size, last_layer, None))
            last_layer.next_layer = self.__intermediate_layers[-1]
            last_layer = self.__intermediate_layers[-1]
        self.__exit_layer.previous_layer = self.__intermediate_layers[-1]
        self.__exit_layer.weights = np.random.rand(self.__exit_layer.previous_layer.n_neurons, self.__exit_layer.n_neurons)
        self.__intermediate_layers[-1].next_layer = self.__exit_layer

    def train(self, momentum, learning_ratio, data, result):
        simplified_exit = self.__initial_layer.propagation(data)
        self.__exit_layer.error_calc(result)
        self.__exit_layer.backpropagation(momentum, learning_ratio)
        return simplified_exit


    def evaluate(self, data):
        simplified_exit = self.__initial_layer.propagation(data)
        return simplified_exit
