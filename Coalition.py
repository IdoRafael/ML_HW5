import itertools
import collections
import itertools
import os
import pickle
import time
from collections import Counter
from collections import defaultdict
from operator import itemgetter
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import completeness_score, v_measure_score, homogeneity_score, adjusted_rand_score, \
    calinski_harabaz_score, silhouette_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from DataPreparation import split_label
from Prediction import load_prepared_data
from ReadWrite import read_data

MODELS_FOLDER = 'Models'
ID_COLUMN = 'IdentityCard_Num'


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
    return [y for _, y in [(i, round(c[i] / sum(c.values()) * 100.0)) for i in c]]


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

    return collections.OrderedDict(sorted({k: 100 * d[k] / s for k in d}.items()))


def get_clusters_labels_sizes(estimator, y):
    labels = get_clusters_labels(estimator, y)
    sizes = get_clusters_sizes_percent(estimator)

    return collections.OrderedDict(sorted({k: [list(labels[k]), sizes[k]] for k in labels}.items()))


def test_model(model, name, parameters, train_x, train_y, score):
    cv = 5

    print(name)
    start_time = time.time()
    classifier = GridSearchCV(
        model,
        parameters,
        cv=cv, scoring=score,
        n_jobs=-1
    ).fit(train_x, train_y)
    print(time.time() - start_time)
    return classifier


def run_experiments(train_x, train_y, names):
    # KMeans
    models = [
        test_model(
            KMeans(),
            names[0],
            [{'n_clusters': list(range(2, 100))}],
            train_x, train_y,
            make_scorer(completeness_score)
        ),

        test_model(
            KMeans(),
            names[1],
            [{'n_clusters': list(range(2, 100))}],
            train_x, train_y,
            make_scorer(homogeneity_score)
        ),

        test_model(
            KMeans(),
            names[2],
            [{'n_clusters': list(range(2, 100))}],
            train_x, train_y,
            make_scorer(v_measure_score)
        ),

        test_model(
            KMeans(),
            names[3],
            [{'n_clusters': list(range(2, 100))}],
            train_x, train_y,
            make_scorer(calinski_harabaz_score)
        ),

        test_model(
            KMeans(),
            names[4],
            [{'n_clusters': list(range(2, 100))}],
            train_x, train_y,
            make_scorer(silhouette_score)
        ),

        test_model(
            KMeans(),
            names[5],
            [{'n_clusters': list(range(2, 100))}],
            train_x, train_y,
            make_scorer(adjusted_rand_score)
        )
    ]

    save_models(models, names)

    return models


def optimize_models_parameters(train, rerun_experiments=False):
    train_x, train_y = split_label(train)
    names = [
        'KMeans_completeness',
        'KMeans_homogeneity',
        'KMeans_v_measure',
        'KMeans_calinski_harabaz',
        'KMeans_silhouette',
        'KMeans_adjusted_rand'
    ]
    models = run_experiments(train_x, train_y, names) if rerun_experiments else load_experiments(names)
    return models, names


def load_experiments(names):
    models = [read_model(name) for name in names]
    for model, name in zip(models, names):
        print_model(model, name)
    return models


def save_models(models, names):
    for model, name in zip(models, names):
        save_model(model, name)


def save_model(model, name):
    # Store data (serialize)
    with open(os.path.join(MODELS_FOLDER, name + '.pickle'), 'wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read_model(name):
    # Load data (deserialize)
    with open(os.path.join(MODELS_FOLDER, name + '.pickle'), 'rb') as handle:
        return pickle.load(handle)


def print_model(model, name):
    print('=' * 100)
    print(name)
    print("Best parameters set found on development set:")
    print(model.best_params_)
    print()
    print("Grid scores on development set:")
    means = model.cv_results_['mean_test_score']
    stds = model.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, model.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print('=' * 100)


def run_k_means_all_data():
    ID_COLUMN = 'IdentityCard_Num'
    test_new_x = read_data('test_new.csv', index=ID_COLUMN)
    test_new_y = read_data('results.csv', index=ID_COLUMN)

    X, y = test_new_x, test_new_y.PredictVote.astype('category').cat.codes

    for k in [6, 9, 10, 11, 12]:
        print(k, '=========')
        kmeans = KMeans(n_clusters=k).fit(X)
        d = get_clusters_labels(kmeans, y)
        s = get_clusters_sizes_percent(kmeans)
        dist = get_clusters_distribution(kmeans, y)

        for i, v in d.items():
            print('{:>2} {:>6}%'.format(i, s[i]), v)
            print('{:>10}'.format('Percent'), np.array(dist[i]))

        print('=========')


def test_combinations():
    train, validate, test, test_new = load_prepared_data()
    train_x, train_y = split_label(train)
    labels = sorted(list(filter(lambda x: x not in {3, 4, 7}, train_y.unique())))

    test_new_x = read_data('test_new.csv', index=ID_COLUMN)
    test_new_y = read_data('results.csv', index=ID_COLUMN)['PredictVote'].astype('category').cat.codes

    result = []

    for r in range(1, min(len(labels) + 1, 11)):
        print(r)
        for c in itertools.combinations(labels, r):
            y = test_new_y.map(lambda x: 1 if x in c else 0)
            counter = Counter(y)
            if {i: counter[i] / len(y) * 100.0 for i in counter}[1] < 51:
                continue
            else:
                score = calinski_harabaz_score(test_new_x, y)
                print(c, score)
                result.append((c, score))

    return result


def get_best_coalitions():
    l = sorted(test_combinations(), key=itemgetter(1), reverse=True)

    print('Best coalitions:')
    for coalition in l[:3]:
        print(coalition)


if __name__ == '__main__':
    run_k_means_all_data()
    get_best_coalitions()
