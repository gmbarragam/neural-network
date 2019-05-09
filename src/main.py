from neural_net import NeuralNet
import matplotlib.pyplot as plt
import numpy as np

def main():
    data = []
    train_data = []
    result_data = []
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
            data.append([[float(line_vector[number])/1300
                          for number in range(1, len(line_vector))], types[line_vector[0]]])

    neural_net = NeuralNet(19, 1, 26, 7)
    confusion_matrix = np.zeros((n_results, n_results))
    for epochs in range(200):
        confusion_matrix = np.zeros((n_results, n_results))
        for item in data:
            train_in = item[0]
            expected = item[1]
            train_exit = neural_net.train(1, 0.05, train_in, expected)
            confusion_matrix[expected[0].tolist().index(1)] += train_exit[0].tolist()
        if(epochs % 50 == 0):
            print(confusion_matrix)
            plt.imshow(confusion_matrix)
            plt.draw()
            plt.pause(1)

    confusion_matrix = np.zeros((n_results, n_results))
"       confusion_matrix = np.zeros((n_results, n_results))
        for item in data:
            train_in = item[0]
            expected = item[1]
            train_exit = neural_net.train(1, 0.05, train_in, expected)
            confusion_matrix[expected[0].tolist().index(1)] += train_exit[0].tolist()
        if(epochs % 50 == 0):
            print(confusion_matrix)
            plt.imshow(confusion_matrix)
            plt.draw()
            plt.pause(1)
"
if __name__ == '__main__':
    main()
