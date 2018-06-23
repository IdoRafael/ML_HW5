from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from Coalition import load_prepared_data, load_unprepared_data
from DataPreparation import split_label


def normalize(l):
    return np.array(l) / sum(l)


def plot_feature_ranks(party, ranker, X, y, ranking_method):
    plt.figure()
    plt.title('{} - {}'.format(party, ranking_method))
    ranks = 100 * normalize(ranker(X, y))
    plt.barh(X.columns.values, ranks)
    plt.xlabel('Importance %')
    plt.ylabel('Feature')
    plt.show()


def party_feature_mi(X, y):
    univariate_filter_mi = SelectKBest(mutual_info_classif, k='all').fit(X, y)
    return univariate_filter_mi.scores_


def get_feature_for_all_parties():
    train, validate, test = load_prepared_data()
    df = pd.concat([train, validate, test])
    X, y = df.drop(['Vote'], axis=1), df['Vote']

    for party in np.unique(df['Vote']):
        plot_feature_ranks(
            party,
            party_feature_mi,
            X,
            y.map(lambda p: 1 if p == party else 0),
            'MI'
        )


def get_feature_for_coalition_opposition():
    train, validate, test = load_prepared_data()
    df = pd.concat([train, validate, test])
    X, y = df.drop(['Vote'], axis=1), df['Vote'].map(lambda x: 0 if x in {'Reds', 'Greys', 'Oranges'} else 1)

    plot_feature_ranks(
        'Coalition/Opposition',
        party_feature_mi,
        X,
        y,
        'MI'
    )


def calculate_mean_feature(train, validate, test, f):
    df = pd.concat([train, validate, test]).dropna(subset=[f])

    print('Total:', np.mean(df[f]))
    print('Coalition')
    print('Purples', np.mean(df[df.Vote == 'Purples'][f]), np.var(df[df.Vote == 'Purples'][f]))
    print('Browns', np.mean(df[df.Vote == 'Browns'][f]), np.var(df[df.Vote == 'Browns'][f]))
    print('Pinks', np.mean(df[df.Vote == 'Pinks'][f]), np.var(df[df.Vote == 'Pinks'][f]))
    print('Opposition')
    print('Greys', np.mean(df[df.Vote == 'Greys'][f]), np.var(df[df.Vote == 'Greys'][f]))
    print('Reds', np.mean(df[df.Vote == 'Reds'][f]), np.var(df[df.Vote == 'Reds'][f]))
    print('Oranges', np.mean(df[df.Vote == 'Oranges'][f]), np.var(df[df.Vote == 'Oranges'][f]))


def calculate_feature_before_after_dp(f):
    calculate_mean_feature(*load_prepared_data(), f)
    calculate_mean_feature(*load_unprepared_data(), f)


def count_feature(train, validate, test, f):
    df = pd.concat([train, validate, test]).dropna(subset=[f])

    print('Coalition')
    print('Purples', Counter(df[df.Vote == 'Purples'][f]))
    print('Browns', Counter(df[df.Vote == 'Browns'][f]))
    print('Pinks', Counter(df[df.Vote == 'Pinks'][f]))
    print('Opposition')
    print('Greys', Counter(df[df.Vote == 'Greys'][f]))
    print('Reds', Counter(df[df.Vote == 'Reds'][f]))
    print('Oranges', Counter(df[df.Vote == 'Oranges'][f]))


def count_feature_before_after_dp(f):
    count_feature(*load_prepared_data(), f)
    count_feature(*load_unprepared_data(), f)


def manipulate_winning_party(manipulation, f):
    train, validate, test = load_prepared_data()

    gbc = GradientBoostingClassifier(max_depth=7, max_features=10).fit(*split_label(train))

    test = pd.concat([validate, test])
    test[f] = test[f].map(
        manipulation
    )
    test_x, _ = split_label(test)

    return test, gbc.predict(test_x)


def manipulate_and_plot_distribution(manipulation, f):
    test, pred_y = manipulate_winning_party(manipulation, f)

    test_x, _ = split_label(test)

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
