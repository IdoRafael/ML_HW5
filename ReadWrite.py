import pandas as pd


FILES_DIR = 'CSVFiles\\'
LABEL_COLUMN = 'Vote'


def read_data(filename, online=False, index=True):
    if online:
        return pd.read_csv(
            'https://webcourse.cs.technion.ac.il/236756/Spring2018/ho/WCFiles/ElectionsData.csv?7959',
            header=0
        )
    else:
        if index:
            return pd.read_csv(FILES_DIR + filename, header=0, index_col='Index')
        else:
            return pd.read_csv(FILES_DIR + filename, header=0)


def save_as_csv_original(train, validate, test):
    train.to_csv(FILES_DIR + "train_original.csv", index_label='Index')
    validate.to_csv(FILES_DIR + "validate_original.csv", index_label='Index')
    test.to_csv(FILES_DIR + "test_original.csv", index_label='Index')


def save_as_csv(train, validate, test):
    train.to_csv(FILES_DIR + "train.csv", index_label='Index')
    validate.to_csv(FILES_DIR + "validate.csv", index_label='Index')
    test.to_csv(FILES_DIR + "test.csv", index_label='Index')


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
    df.to_csv(FILES_DIR + "%s.csv" % name, index_label='Index')