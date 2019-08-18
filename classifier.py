from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
from scipy import interp
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Normalizer, label_binarize
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import class_weight
from sklearn.utils.multiclass import unique_labels

from config import STATES, TRILATERATION
from mi import MAP
from utils import flatten, ml_plot, tidy_obs_data, tidy_rss

normalizer = None


def train(obs, labels, model_type):
    """
    Supervised HMM
    """
    global normalizer
    X, len_X = tidy_rss(obs)
    X = np.atleast_2d(X)
    normalizer = Normalizer().fit(X)
    norm_X = normalizer.transform(X)
    # print(norm_X)
    y, _ = tidy_obs_data(labels)
    # print(np.array(STATES)[y])
    counts = []
    for i in np.unique(y):
        counts.append(list(y).count(i))
    plt.bar(STATES, counts, width=0.35)
    plt.show()
    # print(len_X, len_y)
    cw = list(class_weight.compute_class_weight('balanced', np.unique(y), y))
    cw = dict(enumerate(cw))
    if model_type == 'rf':
        classifier = RandomForestClassifier(
            n_estimators=10, class_weight=cw)
    elif model_type == 'knn':
        classifier = KNeighborsClassifier(n_neighbors=5, weights='distance')
    elif model_type == 'svm':
        classifier = LinearSVC(class_weight=cw)
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


def plot_confusion_matrix(y_true, y_pred, title=None, normalize=False, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    classes = np.array(STATES)[unique_labels(y_true, y_pred)]

    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           ylabel='True label',
           xlabel='Predicted label',
           ylim=(len(classes)-0.5, -0.5))

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()


def roc(obs, labels):
    global normalizer
    X, _ = tidy_rss(obs)
    X = np.atleast_2d(X)
    normalizer = Normalizer().fit(X)
    X = normalizer.transform(X)
    # print(norm_X)
    y, _ = tidy_obs_data(labels)

    y = label_binarize(y, classes=range(0, len(STATES)))
    n_classes = y.shape[1]

    # shuffle and split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5)
    classifier = OneVsRestClassifier(
        SVC(kernel='linear', probability=True))
    y_score = classifier.fit(X_train, y_train).decision_function(X_test)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
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
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
             ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
             ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red',
                    'purple', 'green', 'black', 'orange', 'violet', 'pink'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of room {0} (area = {1:0.2f})'
                 ''.format(STATES[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()
