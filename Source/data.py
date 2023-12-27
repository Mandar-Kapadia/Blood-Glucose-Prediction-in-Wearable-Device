import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from utility import classify_glucose

# parse dataframe entries // inefficient (non-loop based)
# implementation was replaced at some point but made it's way back in
def make_samples(data):
    X, Y = [], []
    for entry in data:
        y = classify_glucose(entry[1][1])
        x = [
            entry[1][2],
            entry[1][3],
            entry[1][4],
            entry[1][5],
            entry[1][6],
            entry[1][7],
            entry[1][8],
            entry[1][9],
            entry[1][10],
            entry[1][11],
            entry[1][12],
            entry[1][13],
            entry[1][14],
            entry[1][15]
        ]

        if any(np.isnan(i) for i in x):
            continue      
        
        Y.append(y)
        X.append(x)

    return np.array(X), np.array(Y)

def init_data():
    df = pd.read_csv('../Datasets/001.csv')
    df2 = pd.read_csv('../Datasets/002.csv')
    df3 = pd.read_csv('../Datasets/003.csv')
    df4 = pd.read_csv('../Datasets/004.csv')
    df4 = pd.read_csv('../Datasets/005.csv')

    data = list(df.iterrows()) \
        + list(df2.iterrows()) \
        + list(df3.iterrows()) \
        + list(df4.iterrows())
    random.shuffle(data)
    X, Y = make_samples(data)
    return X, Y
