import gmplot
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

from config import BUILDING, FIREBASE, GMPLOT
from utils import read_from_firebase


def clean(data):
    X = np.empty((0, 2))
    for d in data:
        sample = np.array([d['lat'], d['lng']])
        X = np.append(X, [sample], axis=0)
    return X


def cluster():

    # Clean data
    data = read_from_firebase(FIREBASE['table'])
    X = clean(data)
    # print("X = ", X)

    # Plot data
    plt.scatter(X[:, 0], X[:, 1])
    plt.xlabel('Latitude')
    plt.ylabel('Longitude')
    plt.title('True Position')
    plt.xlim(min(X[:, 0]), max(X[:, 0]))
    plt.ylim(min(X[:, 1]), max(X[:, 1]))
    plt.show()

    # Apply K-means clustering
    kmeans = KMeans().fit(X)
    K = kmeans.n_clusters
    # print("K = ", K)

    # Plot clusters
    plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='rainbow')
    plt.scatter(kmeans.cluster_centers_[
                :, 0], kmeans.cluster_centers_[:, 1], color='black')
    plt.xlabel('Latitude')
    plt.ylabel('Longitude')
    plt.title('K=%i' % K)
    plt.xlim(min(X[:, 0]), max(X[:, 0]))
    plt.ylim(min(X[:, 1]), max(X[:, 1]))
    plt.show()

    gmap = gmplot.GoogleMapPlotter(48.2630651, 11.6673068, 20)
    gmap.apikey = GMPLOT['apiKey']
    gmap.scatter(X[:, 0], X[:, 1], '#3B0B3', size=0.5, marker=False)
    gmap.draw('./map.html')
