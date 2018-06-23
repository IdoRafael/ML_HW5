import pandas as pd
from DataPreparation import ID_COLUMN, INDEX_COLUMN, LABEL_COLUMN

FILES_DIR = 'CSVFiles\\'


def read_data(filename, index=None):
    if index is None:
        return pd.read_csv(FILES_DIR + filename, header=0)
    else:
        return pd.read_csv(FILES_DIR + filename, header=0, index_col=index)


def save_as_csv_original(train, validate, test):
    train.to_csv(FILES_DIR + "train_original.csv", index_label=INDEX_COLUMN)
    validate.to_csv(FILES_DIR + "validate_original.csv", index_label=INDEX_COLUMN)
    test.to_csv(FILES_DIR + "test_original.csv", index_label=INDEX_COLUMN)


def save_as_csv(train, validate, test, test_new):
    train.to_csv(FILES_DIR + "train.csv", index_label=INDEX_COLUMN)
    validate.to_csv(FILES_DIR + "validate.csv", index_label=INDEX_COLUMN)
    test.to_csv(FILES_DIR + "test.csv", index_label=INDEX_COLUMN)
    test_new.to_csv(FILES_DIR + "test_new.csv", index_label=ID_COLUMN)


def save_features_selected(original_features, new_features):
    selected_features = [f for f in original_features if f in new_features]
    new_selected_features = [f for f in new_features if f not in original_features]

    with open(FILES_DIR + 'selected_features.txt', 'w') as file:
        file.write('Selected Features:\n')
        for f in selected_features:
            if f != LABEL_COLUMN:
                file.write("%s\n" % f)

        file.write('\n')

        file.write('New Selected Features:\n')
        for f in new_selected_features:
            if f != LABEL_COLUMN:
                file.write("%s\n" % f)


def df_as_csv(df, name):
    df.to_csv(FILES_DIR + "%s.csv" % name, index_label=INDEX_COLUMN)