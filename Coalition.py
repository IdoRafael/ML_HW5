import collections
import os
import pickle
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import completeness_score, v_measure_score, homogeneity_score, adjusted_rand_score, \
    calinski_harabaz_score, silhouette_score
from sklearn.metrics import make_scorer, f1_score, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV

from Clustering import get_clusters_labels, get_clusters_sizes_percent, get_clusters_distribution
from DataPreparation import split_label
from ReadWrite import read_data, df_as_csv

MODELS_FOLDER = 'Models'


def load_prepared_data():
    return (read_data(x) for x in ['train.csv', 'validate.csv', 'test.csv'])


def load_unprepared_data():
    return (read_data(x) for x in ['train_original.csv', 'validate_original.csv', 'test_original.csv'])


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


def get_best_model(validate, models, names):
    validate_x, validate_y = split_label(validate)
    evaluated_models = [
        [model, name, f1_score(validate_y, model.predict(validate_x), average='weighted')]
        for model, name in zip(models, names)
    ]

    evaluated_models = sorted(evaluated_models, key=lambda t: t[2], reverse=True)

    print('=' * 100)
    print('Models Evaluated F1 Score:')

    # Print results in a nice format using pd.Dataframe
    print(pd.DataFrame(
        np.matrix([[name, f1] for _, name, f1 in evaluated_models]).transpose(), ['Model Name', 'F1 Score']
    ).transpose())

    print()
    best = evaluated_models[0]
    print('Best Model Is:')
    print(best[1], best[2])
    print('=' * 100)

    return best[0], best[1]


def predict_test_and_save_results(model, name, test):
    test_x, test_y = split_label(test)
    pred_y = model.predict(test_x)
    print('=' * 100)
    print(
        '%s Test F1 (shhh, we\'re not supposed to know this):' % name,
        f1_score(test_y, pred_y, average='weighted')
    )
    print('=' * 100)

    code_to_name = dict(enumerate(test['Vote'].astype('category').cat.categories))
    results = pd.DataFrame(pred_y, test_x.index.values, columns=['Vote'])
    results['Vote'] = results['Vote'].map(code_to_name).astype('category')

    vote_distribution = results['Vote'].value_counts()
    vote_distribution = vote_distribution.divide(sum(vote_distribution.values))
    vote_distribution = vote_distribution.multiply(100)

    plt.figure(figsize=(10, 10))
    bar_plot = vote_distribution.plot.bar(
        color=[c[:-1] for c in results['Vote'].value_counts().index.values],
        edgecolor='black',
        width=0.8
    )

    for p in bar_plot.patches:
        bar_plot.annotate("{:.1f}".format(p.get_height()), (p.get_x() + 0.2, p.get_height() + 0.2))

    bar_plot.set_xlabel('Party')
    bar_plot.set_ylabel('Vote %')

    plt.savefig('vote_distribution.png')
    df_as_csv(results, 'results')

    print('=' * 100)
    print('Confusion Matrix:')
    print(confusion_matrix(test_y, pred_y))
    print('=' * 100)
    print("Test Error (1-accuracy):")
    print(1 - accuracy_score(test_y, pred_y))
    print('=' * 100)

    print(code_to_name)


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
    train, validate, test = load_prepared_data()
    df = pd.concat([train, validate, test])

    # X, y = df.drop(['Vote'], axis=1), df['Vote']
    X, y = split_label(df)

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


if __name__ == '__main__':
    run_k_means_all_data()
