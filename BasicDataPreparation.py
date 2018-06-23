from DataPreparation import split_label


def most_basic_preparation(train, validate, test):
    train_x, _ = split_label(train)
    object_features = train_x.select_dtypes(include='object').columns.values

    train = train.drop(object_features, axis=1).dropna()
    validate = validate.drop(object_features, axis=1).dropna()
    test = test.drop(object_features, axis=1).dropna()

    return train, validate, test
