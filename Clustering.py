import collections
from collections import defaultdict

import numpy as np


def get_clusters_indices(estimator):
    cluster_labels = estimator.labels_

    d = defaultdict(list)

    for i in range(len(cluster_labels)):
        d[cluster_labels[i]].append(i)

    return d


def get_clusters_distribution(estimator, y):
    d = get_clusters_indices(estimator)

    np_y = np.array(y)

    return collections.OrderedDict(
        sorted({k: normalize_counter(collections.Counter(np_y[d[k]])) for k in d}.items())
    )


def normalize_counter(c):
    return [y for _,y in [(i, round(c[i] / sum(c.values()) * 100.0)) for i in c]]


def get_clusters_labels(estimator, y):
    d = get_clusters_indices(estimator)

    np_y = np.array(y)

    return collections.OrderedDict(sorted({k: np.unique(np_y[d[k]]) for k in d}.items()))


def get_clusters_sizes(estimator):
    d = get_clusters_indices(estimator)

    return {k: len(d[k]) for k in d}


def get_clusters_sizes_percent(estimator):
    d = get_clusters_sizes(estimator)

    s = sum(d.values())

    return collections.OrderedDict(sorted({k: 100 * d[k]/s for k in d}.items()))


def get_clusters_labels_sizes(estimator, y):
    labels = get_clusters_labels(estimator, y)
    sizes = get_clusters_sizes_percent(estimator)

    return collections.OrderedDict(sorted({k: [list(labels[k]), sizes[k]] for k in labels}.items()))
