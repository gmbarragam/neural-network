from neural_net import NeuralNet
import matplotlib.pyplot as plt
import numpy as np

def main():
    n_entries = 19
    n_intermidiate_layers = 1
    intermediate_layer_size = 26
    n_exits = 7
    # neural net structure

    momentum = 1
    learning_ratio = 0.05
    n_epochs = 1500
    # sets neural net learning configs

    train_data = []
    evaluate_data = []
    # sets train and evaluate data

    types = {"BRICKFACE":   np.zeros((1,7)),
             "SKY":         np.zeros((1,7)),
             "FOLIAGE":     np.zeros((1,7)),
             "CEMENT":      np.zeros((1,7)),
             "WINDOW":      np.zeros((1,7)),
             "PATH":        np.zeros((1,7)),
             "GRASS":       np.zeros((1,7))}
    count = 0
    for type in types:
        (types[type])[0][count] = 1
        count += 1

    n_results = len(types)
    plt.ion()
    plt.imshow(np.zeros((n_results, n_results)))

    with open("segmentation.test", "r") as arq:
        for line in arq:
            line_vector = line.split(",")
            train_data.append([[float(line_vector[number])/1300
                          for number in range(1, len(line_vector))], types[line_vector[0]]])
        # runs through the file and creates an train array [[data], result]

    with open("segmentation.data", "r") as arq:
        for line in arq:
            line_vector = line.split(",")
            evaluate_data.append([[float(line_vector[number])/1300
                          for number in range(1, len(line_vector))], types[line_vector[0]]])
        # runs through the file and creates an evaluate array [[data], result]

    neural_net = NeuralNet(n_entries, n_intermidiate_layers, intermediate_layer_size, n_exits)
    # creates the neural net

    confusion_matrix = np.zeros((n_results, n_results))
    for epoch in range(n_epochs):
        confusion_matrix = np.zeros((n_results, n_results))
        for item in train_data:
            train_in = item[0]
            expected = item[1]
            train_exit = neural_net.train(momentum, learning_ratio, train_in, expected)
            confusion_matrix[expected[0].tolist().index(1)] += train_exit[0].tolist()
        if(epoch % 20 == 0):
            # every 20 epochs, update the visualization
            confusion_matrix = np.zeros((n_results, n_results))
            for item in evaluate_data:
                evaluate_in = item[0]
                expected = item[1]
                evaluate_exit = neural_net.evaluate(evaluate_in)
                confusion_matrix[expected[0].tolist().index(1)] += evaluate_exit[0].tolist()
            # print(confusion_matrix)
            plt.imshow(confusion_matrix)
            plt.draw()
            plt.pause(1)
            plt.savefig('final_confusion_matrix.png')
    # evaluates the neural net
if __name__ == '__main__':
    main()
