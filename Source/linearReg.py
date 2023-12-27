import pandas as pd
import numpy as np
import random

HIGH_THRESHOLD = 140
LOW_THRESHOLD = 70
TRAINING_SPLIT = 0.75

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        for _ in range(self.num_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            predictions = sigmoid(linear_model)

            dw = (1 / num_samples) * np.dot(X.T, (predictions - y))
            db = (1 / num_samples) * np.sum(predictions - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        predictions = sigmoid(linear_model)
        return [1 if i > 0.5 else 0 for i in predictions]




def classify_glucose(g):
    if g > HIGH_THRESHOLD:
        return 2
    elif g < LOW_THRESHOLD:
        return 0
    return 1


def run_model(rate, iterations):

    model = LogisticRegression(learning_rate=rate, num_iterations=iterations)

    data_001 = pd.read_csv('<data_set_001>.csv')
    data_002 = pd.read_csv('<data_set_002>.csv')
    data_003 = pd.read_csv('<data_set_003>.csv')
    data_004 = pd.read_csv('<data_set_004>.csv')
    data_005 = pd.read_csv('<data_set_005>.csv')

    data_size =  sum(1 for _ in data_001.iterrows())
    data_size += sum(1 for _ in data_002.iterrows())
    data_size += sum(1 for _ in data_003.iterrows())
    data_size += sum(1 for _ in data_004.iterrows())
    data_size += sum(1 for _ in data_005.iterrows())

    training_size = int(TRAINING_SPLIT * data_size)

    Y_train, Y_test, X_train, X_test = [], [], [], []

    data = list(data_001.iterrows()) \
    + list(data_002.iterrows()) \
    + list(data_003.iterrows()) \
    + list(data_004.iterrows()) \
    + list(data_005.iterrows()) 

    random.shuffle(data)

    training_data = data[:training_size]
    testing_data = data[training_size:]

    for entry in training_data:
        X, Y = [], classify_glucose(entry[1][1])

        for i in range(2, len(entry[1])):
            X.append(entry[1][i])

        if any(np.isnan(i) for i in X):
            continue      

        Y_train.append(Y)
        X_train.append(X)

    for entry in testing_data:
        X, Y = [], classify_glucose(entry[1][1])
    
        for i in range(2, len(entry[1])):
            X.append(entry[1][i])

        if any(np.isnan(i) for i in X):
            continue      

        Y_test.append(Y)
        X_test.append(X)


    print("Training Model...")
    model.fit(np.array(X_train), np.array(Y_train))

    print("Making Predictions...")
    predictions = model.predict(X_test)

    n_test_samples = len(Y_test)

    total_correct = 0
    total_low = 0
    correct_low = 0
    correct_high = 0

    total_low = len([i for i in Y_test if i == 0])
    total_high = len([i for i in Y_test if i == 2])

    for i in range(len(predictions)):
        if predictions[i] == Y_test[i]:
            total_correct += 1
        if Y_test[i] == 2:
            if predictions[i] == 2:
                correct_high += 1
        if Y_test[i] == 0:
            if predictions[i] == 0:
                correct_low += 1

    total = total_correct / n_test_samples
    high = correct_high / total_high
    low = correct_low / total_low

    return (total, high, low)

if __name__ == "__main__":
    runs = 25
    total_sum, high_sum, low_sum = 0,0,0
    
    for i in range(runs):
        iteration_scores = run_model(rate=0.1, iterations=10000)
        total_sum += iteration_scores[0]
        low_sum += iteration_scores[1]
        high_sum += iteration_scores[2]

    print(total_sum / runs, low_sum / runs, high_sum / runs)

    