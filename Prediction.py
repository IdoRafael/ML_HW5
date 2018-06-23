import os
import pickle
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import make_scorer, f1_score, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from DataPreparation import split_label
from ReadWrite import read_data, df_as_csv
import matplotlib.pyplot as plt

MODELS_FOLDER = 'Models'
INDEX_COLUMN = 'Index'
ID_COLUMN = 'IdentityCard_Num'


def load_prepared_data():
    for x in ['train.csv', 'validate.csv', 'test.csv']:
        yield read_data(x, index=INDEX_COLUMN)
    yield read_data('test_new.csv', index=ID_COLUMN)


def test_model(model, name, parameters, train_x, train_y):
    score = make_scorer(f1_score, average='weighted')
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
    # SVM
    svc = test_model(
        SVC(),
        'SVC',
        [{'kernel': ['rbf'], 'gamma': 10.0 ** np.arange(-9, 4, 1), 'C': 10.0 ** np.arange(-2, 11, 1)},
         {'kernel': ['linear'], 'C': [1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7]}],
        train_x, train_y
    )

    # KNN
    knn = test_model(
        KNeighborsClassifier(),
        'KNN',
        [{'n_neighbors': list(range(1, 10, 1))}],
        train_x, train_y
    )

    # RANDOM_FOREST
    random_forest = test_model(
        RandomForestClassifier(),
        'RANDOM_FOREST',
        [{'max_depth': [13, 14, 15, 16, 17, 18, 19, None],
          'max_features': ['sqrt', 'log2', 5, 6, 7, 8, 9, 10, None]}],
        train_x, train_y
    )

    # GBC
    gbc = test_model(
        GradientBoostingClassifier(),
        'GBC',
        [{'max_depth': [3, 7, 13, None], 'max_features': [3, 7, 10, 13, None]}],
        train_x, train_y
    )

    # MLP
    mlp = test_model(
        MLPClassifier(),
        'MLP',
        [{'hidden_layer_sizes': [(15,), (15, 15,), (100, 100,), (500, 500,)],
          'alpha': [1e-4, 1.5e-4],
          'learning_rate_init': [1e-3, 1.5e-3]}],
        train_x, train_y
    )
    models = [svc, knn, random_forest, gbc, mlp]
    save_models(models, names)

    return models


def load_experiments(names):
    models = [read_model(name) for name in names]
    for model, name in zip(models, names):
        print_model(model, name)
    return models


def optimize_models_parameters(train, rerun_experiments=False):
    train_x, train_y = split_label(train)
    names = ['SVC', 'KNN', 'RANDOM_FOREST', 'GBC', 'MLP']
    models = run_experiments(train_x, train_y, names) if rerun_experiments else load_experiments(names)
    return models, names


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


def predict_test_and_save_results(model, name, test, test_new):
    test_x, test_y = split_label(test)
    print('=' * 100)
    print(
        '%s Old Test F1:' % name,
        f1_score(test_y, model.predict(test_x), average='weighted')
    )
    print('=' * 100)

    pred_y = model.predict(test_new)

    code_to_name = dict(enumerate(test['Vote'].astype('category').cat.categories))
    results = pd.DataFrame(pred_y, test_new.index.values, columns=['PredictVote'])
    results['PredictVote'] = results['PredictVote'].map(code_to_name).astype('category')

    vote_distribution = results['PredictVote'].value_counts()
    vote_distribution = vote_distribution.divide(sum(vote_distribution.values))
    vote_distribution = vote_distribution.multiply(100)

    plt.figure(figsize=(10, 10))
    bar_plot = vote_distribution.plot.bar(
        color=[c[:-1] for c in results['PredictVote'].value_counts().index.values],
        edgecolor='black',
        width=0.8
    )

    for p in bar_plot.patches:
        bar_plot.annotate("{:.1f}".format(p.get_height()), (p.get_x() + 0.2, p.get_height() + 0.2))

    bar_plot.set_xlabel('Party')
    bar_plot.set_ylabel('Vote %')

    plt.savefig('vote_distribution.png')
    df_as_csv(results, 'results', 'IdentityCard_Num')

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


def load_optimized_models(train):
    train_x, train_y = split_label(train)
    models = [
        SVC(kernel='rbf', C=100000, gamma=0.01),
        KNeighborsClassifier(n_neighbors=3),
        RandomForestClassifier(max_depth=9, max_features=14),
        GradientBoostingClassifier(max_depth=7, max_features=10),
        MLPClassifier(alpha=1.5e-4, hidden_layer_sizes=(500, 500,))
    ]

    return [model.fit(train_x, train_y) for model in models], ['SVC', 'KNN', 'RANDOM_FOREST', 'GBC', 'MLP']


def retrain_best_model_using_all_data(best_model, train, validate, test):
    if hasattr(best_model, 'best_estimator_'):
        best_model = best_model.best_estimator_

    return best_model.fit(
        *split_label(
            pd.concat([train, validate, test])
        )
    )


def load_optimize_fit_select_and_predict():
    train, validate, test, test_new = load_prepared_data()

    # To rerun experiments, change to True. Takes quite a while...
    models, names = optimize_models_parameters(train, rerun_experiments=True)

    best_model, name = get_best_model(validate, models, names)

    best_model = retrain_best_model_using_all_data(best_model, train, validate, test)

    predict_test_and_save_results(best_model, name, test, test_new)


if __name__ == '__main__':
    load_optimize_fit_select_and_predict()
