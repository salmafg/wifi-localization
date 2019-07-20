from operator import itemgetter

import matplotlib.pyplot as plt
import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import LabelEncoder

from map import map
from utils import flatten


le = LabelEncoder()
le_dict = {}


def tidy_data(X):
    global le_dict
    data = []
    lengths = []
    for _, v in X.items():
        for e in v:
            data.append([next((index for (index, d) in enumerate(map)
                               if d['properties']['ref'] == e))])
        lengths.append(len(v))
    print(flatten(data))
    new_data = le.fit_transform(np.ravel(data))
    le_dict = dict(zip(le.classes_, le.transform(le.classes_)))
    print(le_dict)
    print(new_data)
    return new_data, lengths


def fit(X):
    # print(X.values()
    states = list(set(flatten(X.values())))
    # states = map
    X, len_X = tidy_data(X)
    X = np.atleast_2d(X).T

    model = hmm.MultinomialHMM(algorithm='viterbi', init_params='ste', n_components=len(
        states)).fit(X, len_X)

    # plt.plot(labels, ".-", label="observations", ms=6,
    #          mfc="orange", alpha=0.7)
    # plt.legend(loc='best')
    # plt.show()

    return model


def predict(model, obs):
    """
    Takes an observed room number as input and returns the predicted output
    """
    X = next((index for (index, d) in enumerate(map)
              if d['properties']['ref'] == obs))
    X = le.transform([X])
    _, seq = model.decode(np.atleast_2d(X).T)
    seq = le.inverse_transform(seq)
    label = [map[i]['properties']['ref'] for i in seq][0]
    return label


def predict_all(model, obs):
    """
    Takes a dictionary of observations and returns a sequence of predictions
    """
    X = []
    len_X = []
    for _, rooms in obs.items():
        temp = []
        for r in rooms:
            print(r)
            temp.append(next((index for (index, d) in enumerate(map)
                              if d['properties']['ref'] == r)))
        X.append(temp)
        len_X.append(len(temp))
    # print('x:', X)
    X = le.transform(flatten(X))
    # print('transformed x:', X)
    _, seq = model.decode(np.atleast_2d(X).T, len_X)
    # print('seq:', seq)
    seq = le.inverse_transform(seq)
    # print('transformed seq:', seq)
    labels = [map[i]['properties']['ref'] for i in seq]
    # print('labels: ', labels)
    return labels
