import os
from time import time

import numpy as np
import pandas
from memory_profiler import profile
from sklearn.model_selection import train_test_split, cross_val_predict, cross_validate
from sklearn.neural_network import MLPRegressor

from util import method_results


@profile
def mlp_regressor():
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data.csv')
    dataframe = pandas.read_csv(csv_path, header=0)

    X = dataframe.drop('ИН', axis=1)
    y = dataframe['ИН']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    model = MLPRegressor(max_iter=200,
                         solver='lbfgs')
    scores = cross_validate(model, X_train, y_train, cv=10, return_train_score=True)

    predict_start = time()
    predictions = cross_val_predict(model, X_test, y_test, cv=10)
    predict_end = time()
    predict_time = predict_end - predict_start

    model.fit(X_train, y_train)
    # print(model.coefs_)
    print(model.n_layers_)
    print(model.n_outputs_)
    method_results(model, scores, y_test, predictions, [np.shape(a) for a in model.coefs_], predict_time, X_train)


if __name__ == '__main__':
    mlp_regressor()
