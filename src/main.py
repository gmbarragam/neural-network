import os
import sys
import time
import progressbar
import pickle
import logging
import numpy as np
import matplotlib.pyplot as plt

from neural_net import NeuralNet

if(len(sys.argv) < 2 or len(sys.argv) > 3):
    logging.warning('Arguments number incorrect: "python3 src/main.py <mode>"')
    logging.warning(f'{len(sys.argv)}')

MODE = int(sys.argv[1])
MAX_DATA_VALUE = 1300

# sets the progress bar configuration
PROGRESS_BAR_WIDGETS = [progressbar.Percentage(
), progressbar.Bar('█'), progressbar.ETA()]

# neural net structure
N_ENTRIES = 19
N_INTERMIDIATE_LAYERS = 1
INTERMIDIATE_LAYER_SIZE = 26
N_EXITS = 7

# sets neural net learning configs
MOMENTUM = 1
LEARNING_RATIO = 0.05
N_EPOCHS = 1500
TYPES = {"BRICKFACE":   np.zeros((1, 7)),
         "SKY":         np.zeros((1, 7)),
         "FOLIAGE":     np.zeros((1, 7)),
         "CEMENT":      np.zeros((1, 7)),
         "WINDOW":      np.zeros((1, 7)),
         "PATH":        np.zeros((1, 7)),
         "GRASS":       np.zeros((1, 7))}

# sets the counter to feed the exit array models
count = 0

for type in TYPES:
    (TYPES[type])[0][count] = 1
    count += 1

# sets the numer of exits
N_RESULTS = len(TYPES)
RESULTS_PATH = 'results'

try:
    os.mkdir(RESULTS_PATH)
except OSError:
    print ("Creation of the directory %s failed" % RESULTS_PATH)
else:
    print ("Successfully created the directory %s " % RESULTS_PATH)


def main():
    if(MODE == 1):
        train()
    if(MODE == 2):
        evaluate()
    if(MODE == 3):
        generate_metrics(N_EXITS, TYPES)


def train():
    # sets train and evaluate data array
    train_data = []
    evaluate_data = []

    # sets the labels on the matrix
    fig, ax = plt.subplots()
    plt.imshow(np.zeros((N_RESULTS, N_RESULTS)))
    ax.set_xticklabels(["0"] + list(TYPES.keys()))

    ax.set_yticklabels(["0"] + list(TYPES.keys()))
    plt.setp(ax.get_xticklabels(), rotation=10)

    # runs through the file and creates an evaluate array [[data], result]
    try:
        with open("segmentation.data", "r") as arq:
            for line in arq:
                line_vector = line.split(",")
                evaluate_data.append(
                    [[float(line_vector[number]) / MAX_DATA_VALUE for number in range(1, len(line_vector))], TYPES[line_vector[0]]])
    except:
        loggin.critical('segmentation data not found.')
        sys.exit()

    # runs through the file and creates a train array [[data], result]
    try:
        with open("segmentation.test", "r") as arq:
            for line in arq:
                line_vector = line.split(",")
                train_data.append([[float(line_vector[number]) / MAX_DATA_VALUE for number in range(
                    1, len(line_vector))], TYPES[line_vector[0]]])
    except:
        loggin.critical('segmentation data not found.')
        sys.exit()

    # creates the neural net
    neural_net = NeuralNet(N_ENTRIES, N_INTERMIDIATE_LAYERS,
                           INTERMIDIATE_LAYER_SIZE, N_EXITS)

    # creates the confusion matrix
    confusion_matrix = np.zeros((N_RESULTS, N_RESULTS))

    # creates the percent bar
    print('Epochs:')
    bar_epochs = progressbar.ProgressBar(
        widgets=PROGRESS_BAR_WIDGETS, max_value=N_EPOCHS)

    # starts the epochs loop
    for epoch in range(N_EPOCHS):
        confusion_matrix = np.zeros((N_RESULTS, N_RESULTS))
        # runs through the train data and feeds the neural net
        for item in train_data:
            train_in = item[0]
            expected = item[1]
            train_exit = neural_net.train(
                MOMENTUM, LEARNING_RATIO, train_in, expected)
            confusion_matrix[expected[0].tolist().index(
                1)] += train_exit[0].tolist()

        bar_epochs.update(epoch)

        # every 20 epochs, update the visualization, and saves both the neural net and the confusion matrix
        if(epoch % 20 == 0):
            confusion_matrix = np.zeros((N_RESULTS, N_RESULTS))
            for item in evaluate_data:
                evaluate_in = item[0]
                expected = item[1]
                evaluate_exit = neural_net.evaluate(evaluate_in)
                confusion_matrix[expected[0].tolist().index(
                    1)] += evaluate_exit[0].tolist()

            with open('results/neural_net_model.pkl', 'wb') as neural_net_file:
                pickle.dump(neural_net, neural_net_file)

            with open('results/confusion_matrix.pkl', 'wb') as matrix_file:
                pickle.dump(confusion_matrix, matrix_file)

            plt.imshow(confusion_matrix)
            plt.pause(1)
        plt.savefig('final_confusion_matrix.png')


def evaluate():
    # sets the evaluate data array
    evaluate_data = []

    # sets the labels on the matrix
    fig, ax = plt.subplots()
    plt.imshow(np.zeros((N_RESULTS, N_RESULTS)))
    ax.set_xticklabels(["0"] + list(TYPES.keys()))
    plt.ylabel('Esperado')
    plt.xlabel('Obtido')
    plt.title('Matriz Confusão')

    ax.set_yticklabels(["0"] + list(TYPES.keys()))
    plt.setp(ax.get_xticklabels(), rotation=45)

    try:
        with open('results/neural_net_model.pkl', 'rb') as file:
            neural_net_model = pickle.load(file)
    except:
        loggin.critical('Neural net model not found.')

    # runs through the file and creates an evaluate array [[data], result]
    try:
        with open("segmentation.data", "r") as arq:
            for line in arq:
                line_vector = line.split(",")
                evaluate_data.append(
                    [[float(line_vector[number]) / MAX_DATA_VALUE for number in range(1, len(line_vector))], TYPES[line_vector[0]]])

            # creates the loading bar
            bar_evaluate = progressbar.ProgressBar(
                widgets=PROGRESS_BAR_WIDGETS, max_value=N_EPOCHS)

            # runs throught the test dataset and feed the confusion matrix
            confusion_matrix = np.zeros((N_RESULTS, N_RESULTS))

            bar_index = 0
            for item in evaluate_data:
                evaluate_in = item[0]
                expected = item[1]
                evaluate_exit = neural_net_model.evaluate(evaluate_in)
                confusion_matrix[expected[0].tolist().index(
                    1)] += evaluate_exit[0].tolist()
                bar_index += 1
                bar_evaluate.update(bar_index)

            # evaluates the neural net
            for i in range(N_RESULTS):
                for j in range(N_RESULTS):
                    text = ax.text(
                        j, i, int(confusion_matrix[i, j]), ha="center", va="center", color="w")
            # generates a png image with the labeled matrix, with the result number in each field of it
            plt.savefig('final_confusion_matrix.png')
            plt.show()
    except:
        loggin.critical('Confusion matrix not found.')
        sys.exit()

def generate_metrics(N_EXITS, TYPES):
    try:
        with open('results/confusion_matrix.pkl', 'rb') as matrix_file:
            confusion_matrix = pickle.load(matrix_file)
    except:
        loggin.critical('Confusion matrix not found.')
        sys.exit()

    tp = {}
    fp = {}
    fn = {}
    tn = {}

    for key in range(N_EXITS):
        tp[key] = confusion_matrix[key][key]
        fp[key] = np.sum(confusion_matrix[:, key]) - tp[key]
        fn[key] = np.sum(confusion_matrix[key]) - tp[key]
        tn[key] = 0
        for line in range(N_EXITS):
            if(line == key):
                continue
            tn[key] = tn[key] + \
                np.sum(confusion_matrix[line]) - confusion_matrix[line][key]

    accuracy = {}
    error = {}
    recall = {}
    precision = {}
    specificity = {}
    tpr = {}
    fpr = {}
    types_names = [item for item in TYPES.keys()]

    for key in range(N_EXITS):
        accuracy[key] = (tp[key] + tn[key]) / \
            (tp[key] + fp[key] + tn[key] + fn[key])
        error[key] = 1 - accuracy[key]
        recall[key] = tp[key] / (tp[key] + fn[key])
        precision[key] = tp[key] / (tp[key] + fp[key])
        specificity[key] = tn[key] / (tn[key] + fp[key])
        tpr[key] = recall[key]
        fpr[key] = (fp[key] / (tn[key] + fp[key]))

    tpr_list = [item for item in tpr.values()]
    fpr_list = [item for item in fpr.values()]

    for key in range(len(TYPES)):
        print("type name: ", types_names[key])
        print("tp: ", tp[key])
        print("fp: ", fp[key])
        print("fn: ", fn[key])
        print("tn: ", tn[key])
        print("accuracy: ", accuracy[key])
        print("error: ", error[key])
        print("recall: ", recall[key])
        print("precision: ", precision[key])
        print("specificity: ", specificity[key])
        print("----------------------------------------------------------------------------------------------")

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr_list, tpr_list, 'bo')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig('roc_curve.png')
    plt.show()

if(__name__ == "__main__"):
    main()
