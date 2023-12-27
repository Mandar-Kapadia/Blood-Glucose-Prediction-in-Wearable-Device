import random
import numpy as np

HIGH_THRESHOLD = 140
LOW_THRESHOLD = 70

def shuffle(d1, d2):
    join, d1_n, d2_n = [], [], []

    if len(d1) != len(d2):
        return d1, d2
    else:
        for i in range(len(d1)):
            join.append((d1[i], d2[i]))

    random.shuffle(join)

    for e in join:
        d1_n.append(e[0])
        d2_n.append(e[1])

    return np.array(d1_n), np.array(d2_n)


def classify_glucose(g):
    if g > HIGH_THRESHOLD:
        return 2
    elif g < LOW_THRESHOLD:
        return 0
    return 1