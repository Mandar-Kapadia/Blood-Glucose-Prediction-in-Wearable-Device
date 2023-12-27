from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from data import init_data
import random
import numpy as np
from RNN import RNN

HEALTHY_CLASS_THROTTLE = 5
BALANCED_TRAINING = True

def main():
    # Get processed X, Y samples from data sets
    X, Y = init_data()

    #One hot encode tag values
    encoder = OneHotEncoder(sparse=False)
    Y = encoder.fit_transform(Y.reshape(-1, 1))

    X_test, X_train, Y_test, Y_train = train_test_split(X, Y, test_size=0.5, random_state=0)
    
    X_train_n, Y_train_n = [], []
    #Remove Over-Popular Training Samples from Training Set
    if BALANCED_TRAINING:
        for i in range(len(Y_train)):
            index = np.where(Y_train[i] == 1)[0][0]
            if index == 1:
                a = random.randint(0, HEALTHY_CLASS_THROTTLE)
                if a == 0:
                    continue
            X_train_n.append(X_train[i])
            Y_train_n.append(Y_train[i])

        X_train = np.array(X_train_n)
        Y_train = np.array(Y_train_n)

    model = RNN(5000, input_shape=(X_train.shape[1], 1), output_size=Y_train.shape[1])
    model.fit(
        x_train=X_train,
        y_train=Y_train,
        epochs=10000,
        batch_size=8,
        learning_rate=1e-3,
        validation_data=(X_test, Y_test)
    )
    
if __name__  == "__main__":
    main()
