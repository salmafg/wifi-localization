from operator import itemgetter

import matplotlib.pyplot as plt
import numpy as np
from hmmlearn import hmm
from more_itertools import unique_everseen

from map import map
from utils import flatten

states = [
    '00.11.065', '00.11.062', '00.11.059', '00.11.056', '00.11.055',
    '00.11.054', '00.11.053', '00.11.051', 'the corridor'
]


def tidy_data(X):
    data = []
    lengths = []
    for _, v in X.items():
        for e in v:
            data.append([next((index for (index, d) in enumerate(map)
                               if d['properties']['ref'] == e))])
        lengths.append(len(v))
    data = np.ravel(data)
    return data, lengths


def fit(obs):

    start_prob = np.full(len(states), 1/len(states))
    trans_prob = 0.5 * np.identity(len(states))
    trans_prob[:, len(states)-1] = 0.5
    trans_prob[len(states)-1, :len(states)-1] = 0.85/(len(states)-1)
    trans_prob[len(states)-1, len(states)-1] = 0.15
    print(trans_prob)
    emission_prob = np.array([
        [0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3],  # 65
        [0.0, 0.65, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3],  # 62
        [0.0, 0.1, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3],  # 59
        [0.0, 0.0, 0.0, 0.6, 0.1, 0.1, 0.0, 0.0, 0.2],  # 56
        [0.0, 0.0, 0.0, 0.1, 0.6, 0.0, 0.1, 0.0, 0.2],  # 55
        [0.0, 0.0, 0.0, 0.2, 0.0, 0.5, 0.1, 0.0, 0.2],  # 54
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.5, 0.1, 0.2],  # 53
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.6, 0.2],  # 51
        [0.075, 0.075, 0.075, 0.075, 0.075, 0.075, 0.075, 0.075, 0.4]  # corr
    ])

    X, len_X = tidy_data(obs)
    X = np.atleast_2d(X).T

    model = hmm.MultinomialHMM(n_components=len(states))
    model.startprob_ = start_prob
    model.transmat_ = trans_prob
    model.emissionprob_ = emission_prob
    return model


def predict_all(model, obs, alg):
    """
    Takes a dictionary of observations and returns a sequence of predictions
    """
    X = []
    len_X = []
    for _, rooms in obs.items():
        temp = [states.index(i) for i in rooms]
        X.append(temp)
        len_X.append(len(temp))
    # print('x:', X)

    obs_labels = [states[i] for i in flatten(X)]
    # print('Obs: ', obs_labels)

    _, seq = model.decode(np.atleast_2d(X).T, len_X, algorithm=alg)
    # print('seq:', seq)

    pred_labels = [states[i] for i in seq]
    # print('HMM preds: ', pred_labels)

    # Plot obervations and predictions
    plt.plot(obs_labels, ".-", label="observations", ms=6,
             mfc="blue", alpha=0.7)
    plt.legend(loc='best')
    plt.plot(pred_labels, ".-", label="predictions", ms=6,
             mfc="orange", alpha=0.7)
    plt.legend(loc='best')
    if alg == 'map':
        plt.title('Maximum A Posteriori Estimation')
    else:
        plt.title('Viterbi')
    plt.show()
