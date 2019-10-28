import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hmmlearn import hmm as hmmlearn
from pomegranate import *
from seqlearn import hmm as seqlearn
from sklearn.metrics import accuracy_score

from config import STATES
from mi import MAP
from utils import plot_confusion_matrix

warnings.filterwarnings("ignore", category=RuntimeWarning)
matplotlib.rcParams.update({
    'font.size': 20,
    'font.family': 'serif',
    'xtick.labelsize': 'small',
    'ytick.labelsize': 'small',
    'legend.fontsize': 'small',
    'figure.autolayout': True
})


def create(data):
    """
    Generic HMM
    """
    start_prob = np.full(len(STATES), 1/len(STATES))
    trans_prob = np.array([
        [0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3],
        [0.0, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3],
        [0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3],
        [0.0, 0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 0.0, 0.3],
        [0.0, 0.0, 0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 0.3],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.0, 0.0, 0.3],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.0, 0.3],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.3],
        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2]
    ])
    emission_prob = np.array([
        [0.7, 0.1, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.25],  # 65
        [0.01, 0.6, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.3],  # 62
        [0.01, 0.1, 0.7, 0.01, 0.01, 0.01, 0.01, 0.01, 0.3],  # 59
        [0.01, 0.01, 0.01, 0.6, 0.1, 0.1, 0.01, 0.01, 0.2],  # 56
        [0.01, 0.01, 0.01, 0.1, 0.6, 0.01, 0.1, 0.01, 0.2],  # 55
        [0.01, 0.01, 0.01, 0.4, 0.01, 0.6, 0.1, 0.01, 0.4],  # 54
        [0.01, 0.01, 0.01, 0.01, 0.2, 0.01, 0.5, 0.1, 0.2],  # 53
        [0.01, 0.01, 0.01, 0.01, 0.01, 0.1, 0.4, 0.5, 0.2],  # 51
        [0.1, 0.4, 0.1, 0.1, 0.075, 0.075, 0.1, 0.075, 0.5]  # corr
    ])

    hmm = hmmlearn.MultinomialHMM(n_components=len(
        STATES), algorithm='map', init_params='ste')
    hmm.startprob_ = start_prob
    hmm.transmat_ = trans_prob
    hmm.emissionprob_ = emission_prob

    X, len_X = tidy_data(data)
    X = np.atleast_2d(X).T
    hmm.fit(X, len_X)
    return hmm


def fit(obs):
    """
    Unsupervised HMM
    """
    X, len_X = tidy_data(obs)
    X = np.atleast_2d(X).T
    hmm = hmmlearn.MultinomialHMM(n_components=len(STATES), algorithm='map')
    hmm.fit(X, len_X)
    return hmm


def train(obs, labels, test_X, test_y, alg):
    """
    Supervised HMM
    """
    X, len_z = tidy_data(obs)
    y, _ = tidy_data(labels)
    z, len_z = tidy_data(test_X)
    w, _ = tidy_data(test_y)
    # hmm = HiddenMarkovModel.from_samples(
    #     DiscreteDistribution, n_components=len(STATES), X=np.atleast_2d(X).T, labels=np.atleast_2d(y).T)
    start_prob = np.full(len(STATES), 1/len(STATES))
    trans_prob = np.array([
        [0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3],
        [0.0, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3],
        [0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3],
        [0.0, 0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 0.0, 0.3],
        [0.0, 0.0, 0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 0.3],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.0, 0.0, 0.3],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.0, 0.3],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.3],
        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2]
    ])
    d0 = DiscreteDistribution(
        {0: 0.7, 1: 0.1, 2: 0.01, 3: 0.01, 4: 0.01, 5: 0.01, 6: 0.01, 7: 0.01, 8: 0.25})
    d1 = DiscreteDistribution(
        {0: 0.01, 1: 0.6, 2: 0.01, 3: 0.01, 4: 0.01, 5: 0.01, 6: 0.01, 7: 0.01, 8: 0.3})
    d2 = DiscreteDistribution(
        {0: 0.01, 1: 0.1, 2: 0.7, 3: 0.01, 4: 0.01, 5: 0.01, 6: 0.01, 7: 0.01, 8: 0.3})
    d3 = DiscreteDistribution(
        {0: 0.01, 1: 0.01, 2: 0.01, 3: 0.6, 4: 0.1, 5: 0.1, 6: 0.01, 7: 0.01, 8: 0.2})
    d4 = DiscreteDistribution(
        {0: 0.01, 1: 0.01, 2: 0.01, 3: 0.6, 4: 0.1, 5: 0.1, 6: 0.01, 7: 0.01, 8: 0.2})
    d5 = DiscreteDistribution(
        {0: 0.01, 1: 0.01, 2: 0.01, 3: 0.4, 4: 0.01, 5: 0.6, 6: 0.1, 7: 0.01, 8: 0.4})
    d6 = DiscreteDistribution(
        {0: 0.01, 1: 0.01, 2: 0.01, 3: 0.01, 4: 0.2, 5: 0.01, 6: 0.5, 7: 0.1, 8: 0.2})
    d7 = DiscreteDistribution(
        {0: 0.01, 1: 0.01, 2: 0.01, 3: 0.01, 4: 0.01, 5: 0.1, 6: 0.4, 7: 0.5, 8: 0.2})
    d8 = DiscreteDistribution(
        {0: 0.1, 1: 0.4, 2: 0.1, 3: 0.1, 4: 0.075, 5: 0.075, 6: 0.1, 7: 0.075, 8: 0.5})
    dists = [d0, d1, d2, d3, d4, d5, d6, d7, d8]
    hmm = HiddenMarkovModel.from_matrix(trans_prob, dists, start_prob)
    # hmm.fit(np.atleast_2d(X).T, labels=np.atleast_2d(y).T, algorithm='baum-welch')
    seq = hmm.predict(z, algorithm=alg)
    # print(seq)
    print('dataset 1:', hmm.score(X, y))
    print('dataset 2:', hmm.score(z, w))
    plot(test_X, seq, test_y, len_z, alg)
    return hmm


def testing():
    X = [[1, 1], [1, 2], [2, 2], [3, 4], [4, 4], [4, 3],
         [1, 2], [1, 2], [2, 2], [3, 2], [4, 2], [2, 3]]
    y = [[1, 1], [1, 1], [2, 2], [3, 3], [4, 4], [4, 4],
         [1, 1], [1, 1], [2, 2], [3, 3], [4, 4], [2, 2]]
    len_X = len(X)*[2]
    print(y)
    X = np.atleast_2d(np.ravel(X)).T
    y = np.atleast_2d(np.ravel(y)).T
    hmm = seqlearn.MultinomialHMM(decode='bestfirst')
    hmm.fit(X, y, len_X)
    print(hmm.predict(X))
    print(hmm.score(X, y, len_X))


def predict_all(hmm, obs, truth, alg):
    """
    Takes a dictionary of observations and returns a sequence of predictions
    """
    X, len_X = tidy_data(obs)
    # _, seq = hmm.decode(np.atleast_2d(X).T, len_X, algorithm=alg)
    seq = hmm.predict(np.atleast_2d(X).T)
    print(seq)
    plot(obs, seq, truth, len_X, alg)
    return seq


def tidy_data(X):
    data = []
    lengths = []
    for _, v in X.items():
        for e in v:
            data.append([next((index for (index, d) in enumerate(MAP)
                               if d['properties']['ref'] == e))])
        lengths.append(len(v))
    data = np.ravel(data)
    return data, lengths


def tupelize_data(X):
    data = []
    for _, v in X.items():
        for e in v:
            data.append([next((index for (index, d) in enumerate(MAP)
                               if d['properties']['ref'] == e))])
    data = np.ravel(data)
    tuples = list(list(i) for i in zip(data, data[1::]))
    lengths = len(tuples)*[2]
    # tuples = np.ravel(tuples)
    return tuples, lengths


def plot(obs, preds, truth, len_X, alg):
    X, len_X = tidy_data(obs)
    y, _ = tidy_data(truth)
    obs_labels = [STATES[i] for i in X]
    pred_labels = [STATES[i] for i in preds[1:]]
    truth_labels = [STATES[i] for i in y]
    print('Observation score:', accuracy_score(y, X))
    print('Prediction score:', accuracy_score(y, preds))
    # plot_confusion_matrix(y, preds, normalize=True,
    #                       title='Truth vs. Predictions')
    # plot_confusion_matrix(X, preds, normalize=True,
    #                       title='Observations vs. Predictions')
    STATES.reverse()
    truth_df = pd.DataFrame(
        {'col1': range(0, len(truth_labels)), 'col2': truth_labels})
    obs_df = pd.DataFrame(
        {'col1': range(0, len(obs_labels)), 'col2': obs_labels})
    pred_df = pd.DataFrame(
        {'col1': range(0, len(pred_labels)), 'col2': pred_labels})
    plt.figure(figsize=(14.0, 8.0))
    sentinel, = plt.plot(
        np.repeat(truth_df.col1.values[0], len(STATES)), STATES)
    sentinel.remove()
    plt.step(truth_df['col1'], truth_df['col2'],
             label="Truth", ms=6, color="g", alpha=0.7)
    plt.step(obs_df['col1'], obs_df['col2'], '--', label="Localization",
             ms=6, color="tab:blue", alpha=0.7)
    plt.legend(loc='best')
    plt.xlabel('Sample')
    plt.ylabel('Room')
    # plt.title(
    #     'Time-series localization data collected walking through TU campus Garching')
    plt.figure(figsize=(14.0, 8.0))
    sentinel, = plt.plot(
        np.repeat(truth_df.col1.values[0], len(STATES)), STATES)
    sentinel.remove()
    plt.step(truth_df['col1'], truth_df['col2'],
             label="Truth", ms=6, color="g", alpha=0.7)
    plt.step(pred_df['col1'], pred_df['col2'], '--',
             label="HMM", ms=6, color="orange", alpha=0.7)
    plt.legend(loc='best')
    plt.xlabel('Sample')
    plt.ylabel('Room')
    # plt.title('HMM predictions, alg=%s' % alg)
    plt.figure(figsize=(14.0, 8.0))
    sentinel, = plt.plot(
        np.repeat(truth_df.col1.values[0], len(STATES)), STATES)
    sentinel.remove()
    plt.step(obs_df['col1'], obs_df['col2'], label="Localization",
             ms=6, color="tab:blue", alpha=0.7)
    plt.step(pred_df['col1'], pred_df['col2'], '--',
             label="HMM", ms=6, color="orange", alpha=0.7)
    plt.legend(loc='best')
    plt.xlabel('Sample')
    plt.ylabel('Room')
    # plt.title('HMM predictions, alg=%s' % alg)
    plt.show()
