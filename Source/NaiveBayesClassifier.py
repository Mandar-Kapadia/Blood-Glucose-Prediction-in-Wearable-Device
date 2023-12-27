import pandas as pd
import numpy as np
import random
    
HIGH_THRESHOLD = 140
LOW_THRESHOLD = 70
TRAINING_SPLIT = 0.75

class NaiveBayesClassifier:
    def __init__(self):
        self.class_probabilities = {}
        self.feature_probabilities = {}

    def train(self, X, y):
        total_samples = len(y)
        classes, counts = np.unique(y, return_counts=True)
        for c, count in zip(classes, counts):
            self.class_probabilities[c] = count / total_samples

        for c in self.class_probabilities:
            indices = np.where(y == c)
            class_samples = X[indices]
            self.feature_probabilities[c] = {
                'mean': np.mean(class_samples, axis=0),
                'std': np.std(class_samples, axis=0) + 1e-9
            }

    def predict(self, X):
        predictions = []
        for sample in X:
            max_prob = -1
            predicted_class = None
            for c in self.class_probabilities:
                class_prob = self.class_probabilities[c]
                feature_prob = self.feature_probabilities[c]
                prob = class_prob

                for i, value in enumerate(sample):
                    mean = feature_prob['mean'][i]
                    std = feature_prob['std'][i]
                    exponent = np.exp(-((value - mean) ** 2) / (2 * std ** 2))
                    prob *= (1 / (np.sqrt(2 * np.pi) * std)) * exponent

                if prob > max_prob:
                    max_prob = prob
                    predicted_class = c

            predictions.append(predicted_class)
        return predictions
    

def classify_glucose(g):
    if g > HIGH_THRESHOLD:
        return 2
    elif g < LOW_THRESHOLD:
        return 0
    return 1

def run_model():

    model = NaiveBayesClassifier()

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
    model.train(X=np.array(X_train), y=np.array(Y_train))

    print("Making Predictions...")
    predictions = model.predict(np.array(X_test))

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

    return (total, low, high)
    

if __name__ == "__main__":
    runs = 25
    total_sum, high_sum, low_sum = 0,0,0

    for i in range(runs):
        iteration_scores = run_model()
        total_sum += iteration_scores[0]
        low_sum += iteration_scores[1]
        high_sum += iteration_scores[2]

    print(total_sum / runs, low_sum / runs, high_sum / runs)
        