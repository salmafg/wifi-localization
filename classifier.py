import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import class_weight

from config import STATES, TRILATERATION
from mi import MAP
from utils import flatten, tidy_rss, tidy_obs_data, ml_plot

normalizer = None


def train(obs, labels, model_type):
    """
    Supervised HMM
    """
    global normalizer
    X, len_X = tidy_rss(obs)
    X = np.atleast_2d(X)
    normalizer = preprocessing.Normalizer().fit(X)
    norm_X = normalizer.transform(X)
    # print(norm_X)
    y, len_y = tidy_obs_data(labels)
    # print(len_X, len_y)
    # print(y)
    cw = class_weight.compute_class_weight('balanced', np.unique(y), y)
    if model_type == 'rf':
        classifier = RandomForestClassifier(n_estimators=10)
    elif model_type == 'knn':
        classifier = KNeighborsClassifier(n_neighbors=5, weights='distance')
    elif model_type == 'svm':
        classifier = LinearSVC(class_weight='balanced')
    classifier.fit(norm_X, y)
    p = classifier.predict(norm_X)
    print(classifier.score(norm_X, y))
    ml_plot(labels, p, len_X, model_type)
    return classifier


def predict_room(model, sample):
    norm_sample = normalizer.transform(sample)
    pred = model.predict(norm_sample)[0]
    # pred = np.where(model.predict_proba(norm_sample)[0] ==
    #     max(model.predict_proba(norm_sample)[0]))[0][0]
    probs = model.predict_proba(norm_sample)[0]
    # probs = model._predict_proba_lr(norm_sample)[0]
    print(probs)
    return pred, probs[pred]
