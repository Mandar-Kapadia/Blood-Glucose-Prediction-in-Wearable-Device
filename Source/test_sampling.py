from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from data import init_data
import matplotlib.pyplot as plt
import numpy as np 

X, Y = init_data()

encoder = OneHotEncoder(sparse=False)
#One hot encode tag values
Y = encoder.fit_transform(Y.reshape(-1, 1))
highs = []
lows = []
healthy = []
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.80, random_state=1)
for i in range(len(X)):
    if list(Y[i]).index(1) == 0:
        lows.append(X[i])
    elif list(Y[i]).index(1) == 2:
        highs.append(X[i])
    else:
        healthy.append(X[i])

lows_dict = dict(enumerate(lows[0]))
for e in lows_dict:
    lows_dict[e] = []
highs_dict = dict(enumerate(highs[0]))
for e in highs_dict:
    highs_dict[e] = []

for entry in lows[1:]:
    for i in range(len(entry)):
        lows_dict[i].append(entry[i])

for entry in highs[1:]:
    for i in range(len(entry)):
        #entry_ = dict(enumerate(entry))
        highs_dict[i].append(entry[i])

highs_average_dict, highs_std_dict, lows_average_dict, lows_std_dict, highs_range_dict, lows_range_dict = {},{},{},{}, {}, {}

# for i in range(len(lows_dict.keys())):
#     lows_average_dict[i] = (lows_dict[i] / len(lows))

# highs_average_dict = {}

for i in range(len(highs_dict.keys())):
    highs_average_dict[i] = np.mean(highs_dict[i])
    highs_std_dict[i] = np.std(highs_dict[i])
    highs_range_dict[i] = np.ptp(highs_dict[i])

for i in range(len(lows_dict.keys())):
    lows_average_dict[i] = np.mean(lows_dict[i])
    lows_std_dict[i] = np.std(lows_dict[i])
    lows_range_dict[i] = np.ptp(lows_dict[i])


print("****************HIGHS AVG*************")
print(highs_average_dict) 
print("****************HIGHS STD*************")
print(highs_std_dict)
print("****************HIGHS RANGE*************")
print(highs_range_dict)

print("****************LOWS AVGS*************")
print(lows_average_dict)
print("****************LOWS STD*************")
print(lows_std_dict)
print("****************HIGHS RANGE*************")
print(lows_range_dict)


exit()
# Normalize X values
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)