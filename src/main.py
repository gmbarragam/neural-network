from neural_net import NeuralNet
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

    with open("segmentation.test", "r") as arq:
        for line in arq:
            line_vector = line.split(",")
            data.append([[float(line_vector[number])/1300
                          for number in range(1, len(line_vector))], types[line_vector[0]]])

    neural_net = NeuralNet(19, 1, 26, 7)
    confusion_matrix = np.zeros((n_results, n_results))
    for epochs in range(500):
        confusion_matrix = np.zeros((n_results, n_results))
        for item in data:
            train_in = item[0]
            expected = item[1]
            train_exit = neural_net.train(1, 0.0005, train_in, expected)
            # print(train_exit)
            # print(type(train_exit))
            # print(train_exit.shape)
            # print(train_exit)
            confusion_matrix[expected.tolist().index(1)] += train_exit.tolist()[0]
            # print(confusion_matrix)
            # confusion_matrix[expected.index(1)] += np.transpose(train_exit[0])
        print(epochs)
        if(epochs % 50 == 0):
            print(confusion_matrix)
    print(neural_net)


if __name__ == '__main__':
    main()
