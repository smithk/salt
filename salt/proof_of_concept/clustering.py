from matplotlib import pyplot as pl
from matplotlib import cm
from sklearn.cluster import DBSCAN, k_means
import numpy as np


def get_sample():
    means = [10, 30, 50]
    stdevs = [3, 10, 3]
    num_points = [25, 250, 250]

    list_samples = list(np.random.normal(means[i], stdevs[i], num_points[i]) for i in xrange(len(means)))
    print(list_samples)
    sample = np.concatenate(list_samples)
    print(sample)
    sample = np.atleast_2d(sample).T
    return sample


def plot_clusters(sample, labels):
    cluster_ids = np.unique(labels)
    colors = cm.rainbow(np.linspace(0, 1, len(cluster_ids)))
    indices = np.arange(len(labels))
    pl.hist(sample, 100, normed=True)
    for k, color in zip(np.unique(labels), colors):
        if k == -1:
            color = 'k'
        pl.plot(sample[labels == k], indices[labels == k] / 350., 'o', markerfacecolor=color)
    pl.show(block=True)


def get_labels_DBSCAN(sample):
    cluster = DBSCAN(eps=0.5, min_samples=5)
    cluster.fit(sample)
    labels = cluster.labels_
    return labels


def get_labels_k_means(sample):
    means, labels, _ = k_means(sample, 3)
    print(means)
    return labels


if __name__ == '__main__':
    sample = get_sample()

    labels = get_labels_k_means(sample)

    plot_clusters(sample, labels)
