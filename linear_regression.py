from memory_profiler import profile
import pandas
from sklearn.model_selection import train_test_split, cross_val_predict, cross_validate
from sklearn.linear_model import LinearRegression

from util import method_results


@profile
def linear_regression():
    dataframe = pandas.read_csv('data.csv', header=0)

    X = dataframe.drop('ИН', axis=1)
    y = dataframe['ИН']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    model = LinearRegression()
    scores = cross_validate(model, X_train, y_train, cv=10, return_train_score=True)

    predictions = cross_val_predict(model, X_test, y_test, cv=10)

    method_results(model, scores, y_test, predictions)


if __name__ == '__main__':
    linear_regression()
