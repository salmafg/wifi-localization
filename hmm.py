import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hmmlearn import hmm as hmmlearn
from seqlearn import hmm as seqlearn
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from config import STATES
from mi import MAP
from utils import plot_confusion_matrix, tidy_rss

warnings.filterwarnings("ignore", category=RuntimeWarning)
NORMALIZER = None


def create(obs):
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
        [0.10625, 0.10625, 0.10625, 0.10625,
            0.10625, 0.10625, 0.10625, 0.10625, 0.15]
    ])
    emission_prob = np.array([
        [0.7, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.3],  # 65
        [0.01, 0.65, 0.05, 0.01, 0.01, 0.01, 0.01, 0.01, 0.3],  # 62
        [0.01, 0.1, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3],  # 59
        [0.01, 0.01, 0.01, 0.6, 0.1, 0.1, 0.01, 0.01, 0.2],  # 56
        [0.01, 0.01, 0.01, 0.1, 0.6, 0.01, 0.1, 0.01, 0.2],  # 55
        [0.01, 0.01, 0.01, 0.3, 0.01, 0.4, 0.1, 0.01, 0.2],  # 54
        [0.01, 0.01, 0.01, 0.01, 0.01, 0.2, 0.5, 0.1, 0.2],  # 53
        [0.01, 0.01, 0.01, 0.01, 0.01, 0.1, 0.1, 0.6, 0.2],  # 51
        [0.075, 0.075, 0.075, 0.075, 0.075, 0.075, 0.075, 0.075, 0.4]  # corr
    ])

    hmm = hmmlearn.MultinomialHMM(n_components=len(
        STATES), algorithm='map', init_params='ste')
    hmm.startprob_ = start_prob
    hmm.transmat_ = trans_prob
    hmm.emissionprob_ = emission_prob
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


def train(obs, labels):
    """
    Supervised HMM
    """
    global NORMALIZER
    X, len_X = tidy_rss(obs)
    X = np.atleast_2d(X)
    NORMALIZER = preprocessing.Normalizer().fit(X)
    norm_X = NORMALIZER.transform(X)
    print(norm_X)
    y, len_y = tidy_data(labels)
    print(len_X, len_y)
    # print(y)
    alg = 'viterbi'
    # hmm = seqlearn.MultinomialHMM(decode=alg)
    # hmm = DecisionTreeClassifier()
    hmm = RandomForestClassifier(n_estimators=10, criterion='entropy')
    hmm.fit(norm_X, y)
    p = hmm.predict(norm_X)
    print(hmm.score(norm_X, y))
    # plot(labels, p, len_X, alg)
    return hmm


def predict_room(model, sample):
    norm_sample = NORMALIZER.transform(sample)
    pred = model.predict(norm_sample)[0]
    probs = model.predict_proba(norm_sample)[0]
    return pred, probs[pred]


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
    _, seq = hmm.decode(np.atleast_2d(X).T, len_X, algorithm=alg)
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
    pred_labels = [STATES[i] for i in preds]
    truth_labels = [STATES[i] for i in y]
    print('Prediction score:', accuracy_score(y, preds))
    print('Observation score:', accuracy_score(y, X))
    plot_confusion_matrix(y, preds, normalize=True)
    plot_confusion_matrix(X, preds, normalize=True)
    STATES.reverse()
    truth_df = pd.DataFrame(
        {'col1': range(0, len(truth_labels)), 'col2': truth_labels})
    obs_df = pd.DataFrame(
        {'col1': range(0, len(obs_labels)), 'col2': obs_labels})
    pred_df = pd.DataFrame(
        {'col1': range(0, len(pred_labels)), 'col2': pred_labels})
    sentinel, = plt.plot(
        np.repeat(truth_df.col1.values[0], len(STATES)), STATES)
    sentinel.remove()
    plt.step(truth_df['col1'], truth_df['col2'], '--',
             label="truths", ms=6, color="g", alpha=0.7)
    plt.step(obs_df['col1'], obs_df['col2'], label="observations",
             ms=6, color="tab:blue", alpha=0.7)
    plt.legend(loc='best')
    plt.title(
        'Sequence of localization data collected walking through TU campus Garching')
    plt.figure()
    sentinel, = plt.plot(
        np.repeat(truth_df.col1.values[0], len(STATES)), STATES)
    sentinel.remove()
    plt.step(truth_df['col1'], truth_df['col2'], '--',
             label="truths", ms=6, color="g", alpha=0.7)
    plt.step(pred_df['col1'], pred_df['col2'],
             label="predictions", ms=6, color="orange", alpha=0.7)
    plt.legend(loc='best')
    plt.title('HMM predictions, alg=%s' % alg)
    plt.show()
