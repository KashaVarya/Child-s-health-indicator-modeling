from time import time

import graphviz
import pandas
from memory_profiler import profile
from sklearn import tree
from sklearn.model_selection import train_test_split, cross_val_predict, cross_validate

from util import method_results


@profile
def decision_tree():
    dataframe = pandas.read_csv('data.csv', header=0)

    X = dataframe.drop('ИН', axis=1)
    y = dataframe['ИН']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    model = tree.DecisionTreeRegressor(max_depth=100)
    scores = cross_validate(model, X_train, y_train, cv=10, return_train_score=True)

    predict_start = time()
    predictions = cross_val_predict(model, X_test, y_test, cv=10)
    predict_end = time()
    predict_time = predict_end - predict_start

    model.fit(X_train, y_train)
    method_results(model, scores, y_test, predictions, model.feature_importances_.shape[0], predict_time)

    graph_data = tree.export_graphviz(model, out_file=None, filled=True)
    graph = graphviz.Source(graph_data)
    graph.render("children", view=True)


if __name__ == '__main__':
    decision_tree()
