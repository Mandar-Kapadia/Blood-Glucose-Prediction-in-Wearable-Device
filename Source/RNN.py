import utility
import numpy as np
import math

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

class RNN():
    def __init__(self, units, input_shape, output_size):
        self.units = units
        self.input_shape = input_shape
        self.output_size = output_size

        #Set Weight and Biases
        self.w1 = np.random.randn(self.units, input_shape[0]) * np.sqrt(1./input_shape[0])
        self.b1 = np.zeros((units, 1)) * np.sqrt(1./input_shape[0])
        self.w2 = np.random.randn(output_size, self.units) * np.sqrt(1./self.units)
        self.b2 = np.zeros((output_size, 1)) * np.sqrt(1./self.units)

    #Forward Pass with Softmax Activation for Class Probability
    def forward_pass(self, x):
        self.x_c = x
        self.z1_c = self.w1 @ self.x_c.T + self.b1
        self.a1_c = sigmoid(self.z1_c)
        return softmax(self.w2 @ self.a1_c + self.b2)

    #Update earlier layers with derivative sigmoid
    def backward_pass(self, dZ2):

        self.dW2 = (1/self.batch_size) * (dZ2 @ self.a1_c.T)
        self.db2 = (1/self.batch_size) * np.sum(dZ2)

        dA1 = self.w2.T @ dZ2
        aZ1 = sigmoid(self.z1_c)
        dZ1 = dA1 * (aZ1 * (1 - aZ1))

        self.dW1 = (1/self.batch_size) * (dZ1 @ self.x_c)
        self.db1 = (1/self.batch_size) * np.sum(dZ1)

    #Stochastic Gradient Descent optimization
    def sgd(self):
        self.w1 = self.w1 - (self.learning_rate * self.dW1)
        self.b1 = self.b1 - (self.learning_rate * self.db1)
        self.w2 = self.w2 - (self.learning_rate * self.dW2)
        self.b2 = self.b2 - (self.learning_rate * self.db2)

    #training function
    def fit(self, x_train, y_train, epochs, batch_size, learning_rate, validation_data):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.val_x = validation_data[0]
        self.val_y = validation_data[1]

        n_batches = int(x_train.shape[0] / batch_size)

        for i in range(epochs):
            x_train, y_train = utility.shuffle(x_train, y_train)

            for j in range(n_batches):
                s = self.batch_size * j
                e = min(x_train.shape[0], s + self.batch_size)
                batch_x, batch_y = x_train[s:e], y_train[s:e]
                output = self.forward_pass(batch_x)
                self.backward_pass(output - batch_y.T)
                self.sgd()

            output_train = self.forward_pass(x_train)
            output_test = self.forward_pass(self.val_x)
            train_accuracy, training_loss, _ = self.evaluate(y_train, output_train)
            test_accuracy, test_loss, acc_scors = self.evaluate(self.val_y, output_test)
            print(f"Current Epoch: {i} | Training Acc: {train_accuracy} | Training Loss: {training_loss} | Testing Acc: {test_accuracy} | Testing Loss: {test_loss}")
            print(f"Testing | Overall Accuracy: {acc_scors[0]} | Low Accuracy {acc_scors[1]} | High Accuracy: {acc_scors[2]}")

    # Calculate accuracies and cross entropy loss
    def evaluate(self, y, output):
        log_sum = np.sum(y.T * np.log(output))
        c_entropy_loss =  -(1 / y.shape[0]) * log_sum

        o_m, h_m, l_m = 0, 0, 0
        l_c, h_c = sum(1 for e in list(y) if list(e).index(1) == 0), sum(1 for e in list(y) if list(e).index(1) == 2)

        for i in range(y.shape[0]):
            comb = [output[0][i], output[1][i], output[2][i]]
            pred = comb.index(max(comb))
            tag = list(y[i]).index(1)
            if pred == tag:
                o_m += 1
            if tag == 0:
                if tag == pred:
                    l_m += 1
            if tag == 2:
                if tag == pred:
                    h_m += 1
        
        o_acc = o_m / y.shape[0]
        if l_c != 0:
            l_acc = l_m / l_c
        else:
            l_acc = 0
        if h_c != 0:
            h_acc = h_m / h_c
        else:
            h_acc = 0
        return np.mean(np.argmax(y, axis=-1) == np.argmax(output.T, axis=-1)), c_entropy_loss, (o_acc, l_acc, h_acc)
