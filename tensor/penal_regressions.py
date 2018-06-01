"""Class and function for penalized regressions with tensorflow."""
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import scale
from tensorflow_example import download_data
from sklearn_penal_regression import sklearn_models
from tensorflow_penal_regression import tensorflow_models
import os
import pandas as pd
import numpy as np


def get_test_data(download_path='tensor/data'):
    """Download and processing of test data."""
    DATA_PATH = os.path.abspath(download_path)
    TESTDATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data"
    download_data(TESTDATA_URL, DATA_PATH)

    testdata = pd.read_csv(os.path.join(DATA_PATH, 'sonar.all-data'), header=None)
    print("Test data has ", testdata.shape[0], "rows")
    print("Test data has ", testdata.shape[1], "features")
    X = scale(testdata.iloc[:, :-1])

    y = testdata.iloc[:, -1].values
    encoder = LabelEncoder()
    encoder.fit(np.unique(y))
    y = encoder.transform(y)

    return X, y


if __name__ == '__main__':
    DATA_PATH = 'tensor/data'
    model_comparision_file = os.path.join(DATA_PATH, 'model.comparisions')
    X, y = get_test_data(DATA_PATH)

    sk_models = sklearn_models(X, y, model_comparision_file, True)
    tf_models = tensorflow_models(X, y, model_comparision_file, True)
    sk_models.l1_model()
    sk_models.l2_model()
    tf_models.model(regular='l1')
    tf_models.model(regular='l2')
