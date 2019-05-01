import os
from time import time

import pandas
from memory_profiler import profile
from sklearn import neighbors
from sklearn.model_selection import train_test_split, cross_val_predict, cross_validate

from util import method_results


@profile
def k_nearest():
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data.csv')
    dataframe = pandas.read_csv(csv_path, header=0)

    X = dataframe.drop('ИН', axis=1)
    y = dataframe['ИН']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    model = neighbors.KNeighborsRegressor(n_neighbors=10)
    scores = cross_validate(model, X_train, y_train, cv=10, return_train_score=True)

    predict_start = time()
    predictions = cross_val_predict(model, X_test, y_test, cv=10)
    predict_end = time()
    predict_time = predict_end - predict_start

    method_results(model, scores, y_test, predictions, None, predict_time, X_train)


if __name__ == '__main__':
    k_nearest()
