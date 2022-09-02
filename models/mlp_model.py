import os
import sys
curr_dir = os.getcwd()+'\models'
sys.path.append(curr_dir)

import numpy as np
import csv
import matplotlib.pyplot as plt
from initializations_functions import normal
import matplotlib.colors as clr

from activations_functions import default_activation


class MLP:
    """
    input: numpy matrix [num_features, num_examples]
    hidden_nodes: int
    initialization: function to initialize the weights. Default randn
    activation: activate function (see activations_functions)
    """

    def __init__(self, input=None, hidden_nodes=5, out=1, initialization=normal(0, 0.5), activation=default_activation,
                 learning_rate=0.01, batch_mode=True):

        self.hidden_nodes = hidden_nodes
        self.input = self.prepare_input(input)
        self.num_input = self.input.shape[0]
        self.num_out = out
        self.learning_rate = learning_rate
        self.weights = self.initialize_weights(initialization)
        self.activation = activation()

        self.hin = []
        self.h = []

        self.batch_mode = batch_mode


        print(f'num of hidden nodes {self.hidden_nodes}, lr {self.learning_rate}')

    @staticmethod
    def prepare_input(input):
        return np.vstack([input, np.ones(input.shape[1])])

    def initialize_weights(self, initialization):
        """
        Setting of the network weights following a normal distribution
        :param initialization: function from intilizations_functions to determine a normal distirbution,
                                with (mean, std, (num of hidden nodes, num of input)) as input
        :return: array with initial weights
        """
        weights = []

        weights.append(initialization(self.hidden_nodes, self.num_input))
        weights.append(initialization(self.num_out, self.hidden_nodes + 1))

        return weights

    def forward_pass(self, input_train):
        '''
        Frward pass for the training procedure
        :param input_train: input to the network
        :return: array with output values
        '''
        self.h = []

        last = self.activation(self.weights[0] @ input_train)
        last_v2 = np.vstack([last, np.ones(input_train.shape[1])])
        self.h.append(last_v2)

        oin = self.activation(self.weights[1] @ last_v2)
        self.h.append(oin)
        return oin

    """
    This models only works for 1 hidden layer
    """

    def backward_pass(self, targets):
        """
        Bacward pass from the training procedure
        :param targets: y_True values
        :return: weight updates for the ouput and hidden layers
        """
        delta_o = (self.h[1] - targets) * self.activation.derivative(self.h[1])
        delta_h = (self.weights[1].T @ delta_o) * self.activation.derivative(self.h[0])
        delta_h = delta_h[0:self.hidden_nodes, :]

        return delta_o, delta_h

    def weights_update(self, delta_o, delta_h, dw, alpha, input_train):
        """
        Update pass for the training procedure
        :param delta_o: weight output for the output layer
        :param delta_h: weight update for the hidden layer
        :param dw: updated weights from previous iterations
        :param alpha:
        :param input_train:
        :return: updated weights
        """
        if not dw:
            dw.append(delta_h @ input_train.T)
            dw.append(delta_o @ self.h[0].T)

        dw[0] = alpha * dw[0] - (1 - alpha) * delta_h @ input_train.T
        dw[1] = alpha * dw[1] - (1 - alpha) * delta_o @ self.h[0].T

        self.weights[0] += dw[0] * self.learning_rate
        self.weights[1] += dw[1] * self.learning_rate

        return dw

    def backprop(self, targets, epochs=1000, alpha=0.6, early_stopping=0.0001):
        """
        Backpropaation algorithm, including forward, backward and weight update steps
        :param targets: y_true values
        :param epochs: int with number of iterations
        :param alpha: parameter for weight update
        :return: Forward pass results from last epoch
        """
        dw = []
        error_list = [10]
        tol = 0

        im = 0

        for e in range(epochs):
            # Right now all samples are trained at once,
            # for a more difficult task with more data it would be useful
            # to implement batch
            error_prev = error_list[-1]
            oin = self.forward_pass(self.input)
            delta_o, delta_h = self.backward_pass(targets)
            dw = self.weights_update(delta_o, delta_h, dw, alpha, self.input)
            error = self.mse(self.forward_pass(self.input), targets)
            # print(f"mse {error}")
            error_list.append(error)
            dif = error_prev - error

            # Early stopping
            tol += 1 if dif < early_stopping else tol == 0

            if tol > 10:
                print(
                    f'stop because early stopping at epoch {e} with error {error}, improvement from previous error {dif}')
                return oin, error_list

        return oin, error_list

    def save_weights(self, path):
        """
            Save weights
            :param path: director to save the weights
        """
        path_weights_out = path + '_weights_out.csv'
        path_weights_h = path + '_weights_h.csv'
        with open(path_weights_out, "w") as f:
            wr = csv.writer(f)
            wr.writerows(self.weights[1])

        with open(path_weights_h, "w") as f:
            wr = csv.writer(f)
            wr.writerows(self.weights[0])

    def upload_weights(self, path):
        """
            Upload weights for prediction/retrain the model
            :param path: director from weights
        """

        path_weights_out = path + '_weights_out'
        path_weights_h = path + '_weights_h'

        self.weights = self.initialize_weights(initialization=normal(0, 0.5))

        with open(path_weights_h, 'r', newline='\n') as read_obj:
            csv_reader = csv.reader(read_obj)
            # convert string to list
            self.weights[0] = list(csv_reader)

        with open(path_weights_out, 'r', newline='\n') as read_obj:
            csv_reader = csv.reader(read_obj)
            # convert string to list
            self.weights[1] = list(csv_reader)

    def mse(self, oin, targets):
        """
        Compute mean square errors
        :param epochs: output values
        :param alpha: target values
        :return: mean square error
        """

        predict = oin.reshape(-1)
        return np.sum(np.square(predict - targets)) / self.input.shape[1]
        # return mean_squared_error(targets, predict)

    def accuracy(self, predictions, targets):
        """
        Compute accuracy of the network
        :param epochs: predictions values
        :param alpha: targets values
        :return: accuracy
        """
        predictions = predictions.reshape(-1)
        predictions = [-1 if x < 0 else 1 for x in predictions]
        return np.sum(targets == predictions) / self.input.shape[1]

    def get_weights(self):
        return self.weights

    def predict(self, input=None, binaryClass=True):
        """
        Prediction step
        :param input: input data
        :param binaryClass: boolean for binary classification
        :return: predictions
        """
        if input is None:
            input = self.input
        else:
            if input.shape[0] != self.weights[0].shape[1]:
                input = np.vstack((input, np.ones(input.shape[1])))
        last = self.activation(self.weights[0] @ input)
        last = np.vstack([last, np.ones(input.shape[1])])
        oin = self.activation(self.weights[1] @ last)
        if binaryClass:
            prediction = oin.reshape(-1)
            prediction[prediction > 0] = 1
            prediction[prediction <= 0] = -1
            return prediction
        return oin

    def reset(self):
        """
        Reset function to set all parameters to their initial values
        """

        self.hidden_nodes = None
        self.input = None
        self.num_input = None
        self.num_out = None
        self.weights = None
        self.activation = None
        self.learning_rate = None

        self.hin = []
        self.h = []

    def plot_decision_boundary(self, target_train, input_test, target_test, points=True, h=0.1):
        """
        Plot decision boundaries for a given area
        :param target: output data
        :param points: boolean to draw the trained points
        :return: predictions

        """

        xx, yy = np.meshgrid(np.arange(-2, 2, h), np.arange(-1, 1, h))
        grid_data = np.transpose(np.c_[xx.ravel(), yy.ravel()])
        Z = self.predict(grid_data)
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
        label_train = ['class A train', 'class B train']
        label_test = ['class A test', 'class B test']
        label = np.unique(target_test)
        cmap = plt.colormaps["winter"].with_extremes(under="magenta", over="yellow")

        if points:
            for i in range(2):
                idx_train = np.where(target_train==label[i])
                idx_test = np.where(target_test == label[i])
                plt.scatter(self.input[0, idx_train], self.input[1, idx_train], cmap=cmap,  label= label_train[i])
                plt.scatter(input_test[0, idx_test], input_test[1, idx_test], cmap=cmap, label = label_test[i])

        #plt.legend(['class A train', 'class B train', 'class A test', 'class B test'])
        plt.legend()
        plt.show()