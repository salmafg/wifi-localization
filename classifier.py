import csv
from itertools import cycle

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import AllKNN
from scipy import interp
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Normalizer, StandardScaler, label_binarize
from sklearn.svm import SVC
from sklearn.utils import class_weight

from config import ML, STATES
from utils import plot_confusion_matrix

matplotlib.rcParams.update({
    'font.size': 22,
    'font.family': 'serif',
    'xtick.labelsize': 'small',
    'ytick.labelsize': 'small',
    'legend.fontsize': 'small',
    'figure.autolayout': True,
    # 'figure.figsize': (12, 8)
})

NORMALIZER = None
SCALER = None


def train(model_type):
    """
    Supervised HMM
    """
    global NORMALIZER, SCALER
    X, y = parse_csv('data/ml/samsung.csv')
    # X_test, y_test = parse_csv('data/ml/george.csv')
    if model_type == 'svm':
        SCALER = StandardScaler().fit(X)
        X = SCALER.transform(X)
    else:
        NORMALIZER = Normalizer().fit(X)
        X = NORMALIZER.transform(X)

    # print(X)

    # Plot class weights
    # counts = []
    # for i in np.unique(y):
    #     counts.append(list(y).count(i))
    # print(counts)
    # plt.bar(STATES, counts, width=0.35)
    # plt.show()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5)
    # X_train = X
    # y_train = y
    # counts = []
    # for i in np.unique(y):
    #     counts.append(list(y).count(i))
    # print(counts)
    # plt.bar(STATES, counts, width=0.35)
    # plt.show()

    errors = []
    for k in range(1, 11):
        classifier = KNeighborsClassifier(n_neighbors=k, weights='distance')
        scores = cross_val_score(classifier, X, y, cv=10)
        print('CV Scores:', scores)
        print('Accuracy: %0.2f (+/- %0.2f)' % (scores.mean(), scores.std() * 2))
        classifier.fit(X_train, y_train)
        print('Score:', classifier.score(X_test, y_test))
        errors.append(scores.mean())
        # plot_confusion_matrix(y_test, classifier.predict(X_test), normalize=True)
    print(errors)
    plt.figure(figsize=(12.0, 8.0))
    plt.plot(range(1, 11), errors, '-s', ms=10)
    plt.xlabel('Number of Neighbors ($k$)')
    plt.ylabel('Cross-validation Score')
    plt.show()
    return errors


def predict_room(model, sample):
    if isinstance(model, SVC):
        sample = SCALER.transform(sample)
    else:
        sample = NORMALIZER.transform(sample)
    pred = model.predict(sample)[0]
    # pred = np.where(model.predict_proba(sample)[0] ==
    #     max(model.predict_proba(sample)[0]))[0][0]
    probs = model.predict_proba(sample)[0]
    # probs = model._predict_proba_lr(norm_sample)[0]
    print(probs)
    return pred, probs[pred]


def parse_csv(filename):
    X = []
    y = []
    with open(filename, 'r') as csvFile:
        reader = csv.reader(csvFile)
        for row in reader:
            X.append(list(map(int, row[1: len(row)])))
            y.append(STATES.index(row[0]))
    csvFile.close()
    return np.atleast_2d(X), y


def roc():
    global NORMALIZER, SCALER
    X, y = parse_csv('data/ml/samsung.csv')
    NORMALIZER = Normalizer().fit(X)
    X = NORMALIZER.transform(X)

    smote_tomek = SMOTETomek(random_state=0)
    X, y = smote_tomek.fit_resample(X, y)
    cw = list(class_weight.compute_class_weight('balanced', np.unique(y), y))
    cw = dict(enumerate(cw))

    y = label_binarize(y, classes=range(0, len(STATES)))
    n_classes = y.shape[1]

    # shuffle and split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5)
    print(y_test.shape)
    classifier = OneVsRestClassifier(
        SVC(gamma='auto', kernel='poly', probability=True))
    y_score = classifier.fit(X_train, y_train).decision_function(X_test)
    # y_score = classifier.fit(X_train, y_train).predict_proba(X_test)
    # y_pred = np.argmax(y_score, axis=1)
    # y_true = [np.where(r==1)[0][0] for r in y_test]
    # plot_confusion_matrix(y_true, y_pred)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], threshold = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        j_scores = tpr[i]-fpr[i]
        j_ordered = sorted(zip(j_scores, threshold))
        print('Optimal threshold for %s is %f' % (STATES[i], j_ordered[-1][1]))

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(
        y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red',
                    'green', 'orange', 'violet'])
    for i, color in zip(range(n_classes-1), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='Room {0} (area = {1:0.2f})'
                 ''.format(STATES[i], roc_auc[i]))
    plt.plot(fpr[8], tpr[8], color='purple', lw=2,
            label='Corridor (area = %.2f)' % roc_auc[8])
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.show()
    return classifier
