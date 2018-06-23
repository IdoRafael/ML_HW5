# I have no idea what I'm doing
# ~~ Proverbs 4:20

from FeatureImportance import load_prepared_data
from DataPreparation import split_label
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV, KFold


def get_LDA_classifier(X_train, y_train, X_test, y_test):
    # TODO: which solver?
    n_splits = 5
    # model = GridSearchCV(LinearDiscriminantAnalysis(), param_grid=[{'n_components': list(range(3,7))}], cv=5)
    # model.fit(X_train, y_train)
    # print('score for optimal is {}'.format(model.score(X_test, y_test)))
    # kf = KFold(n_splits=n_splits)
    # score_total = 0
    # for train_index, val_index in kf.split(X_train):
    #     lda = LinearDiscriminantAnalysis(solver='eigen')
    #     X_train_fold, X_val_fold = np.array(X_train)[train_index], np.array(X_train)[val_index]
    #     y_train_fold, y_val_fold = np.array(y_train)[train_index], np.array(y_train)[val_index]
    #     score = lda.fit(X_train_fold, y_train_fold).score(X_val_fold, y_val_fold)
    #     score_total += score
    # avg_score = score_total/n_splits
    # print("average accuracy for LDA using {}-fold CV is {:.2f}".format(n_splits, avg_score))
    # TODO: should this be done using CV?
    # X_train_new = lda.fit_transform(X_train, y_train)
    lda = LinearDiscriminantAnalysis(solver='eigen')
    lda.fit(X_train, y_train)
    score = lda.score(X_test, y_test)
    print('score for LDA classifier is {}'.format(score))
    return lda


def get_LDA_coefficients(lda):
    coefficients = lda.coef_
    max_args = np.argmax(coefficients, axis=1)
    # print(coefficients)
    max_args_abs = np.argmax(np.abs(coefficients), axis=1)
    return max_args, max_args_abs, coefficients


def plot_leading_features(max_args, max_args_abs, coefficients, df):
    code_to_name = dict(enumerate(df['Vote'].astype('category').cat.categories))
    attribute_index_to_name = dict(enumerate(df.columns.values))

    for i, max_arg_index in enumerate(max_args_abs):
        print('for party {} the most important feature was {} with a linear coefficient of {:.4}'
              .format(code_to_name[i],
                      attribute_index_to_name[max_arg_index],
                      coefficients[i][max_arg_index]))

        plt.figure(figsize=(8, 6))
        plt.subplots_adjust(left=0.4)
        plt.title('{} most important factors\n(larger (abs) is better)'.format(code_to_name[i]))
        plt.barh(X_test.columns.values, coefficients[i])
        plt.tight_layout()

        # commented out saving since we have the graphics
        # plt.savefig('{}_LDA.png'.format(code_to_name[i]))

        plt.show()
        plt.close()

        plt.figure(figsize=(8, 6))
        plt.subplots_adjust(left=0.2)
        plt.title('{} most important factors %'.format(code_to_name[i]))
        plt.barh(X_test.columns.values, 100 * (np.abs(coefficients[i]) / np.sum(np.abs(coefficients[i]))))
        plt.xlabel('Importance %')

        # commented out saving since we have the graphics
        # plt.savefig('{}_LDA_percent.png'.format(code_to_name[i]))

        plt.show()
        plt.close()


def get_data():
    train, validate, test = load_prepared_data()
    df = pd.concat([train, validate])
    X_train, y_train = split_label(df)
    X_test, y_test = split_label(test)
    return X_train, y_train, X_test, y_test, df


if __name__ == '__main__':
    X_train, y_train, X_test, y_test, df = get_data()
    lda = get_LDA_classifier(X_train, y_train, X_test, y_test)
    max_args, max_args_abs, coefficients = get_LDA_coefficients(lda)
    plot_leading_features(max_args, max_args_abs, coefficients, df)
