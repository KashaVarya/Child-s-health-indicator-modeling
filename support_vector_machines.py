from memory_profiler import profile
import pandas
from sklearn.model_selection import train_test_split, cross_val_predict, cross_validate
from sklearn.svm import SVR
from sklearn.preprocessing import RobustScaler

from util import method_results


@profile
def svm():
    dataframe = pandas.read_csv('data.csv', header=0)

    X = dataframe.drop('ИН', axis=1)
    y = dataframe['ИН']

    rbX = RobustScaler()
    X = rbX.fit_transform(X)
    rbY = RobustScaler()
    y = rbY.fit_transform(y.values.reshape(-1, 1))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # kernel: rbf, sigmoid, linear, poly   gamma=0.1, C=100, epsilon=0.1
    model = SVR(kernel='rbf', gamma='auto')

    scores = cross_validate(model, X_train, y_train.ravel(), cv=10, return_train_score=True)
    predictions = cross_val_predict(model, rbX.transform(X_test), y_test.ravel(), cv=10)

    model.fit(X_train, y_train.ravel())
    params_shape = model.support_vectors_.shape[0] * model.support_vectors_.shape[1]
    method_results(model, scores, y_test, predictions, params_shape)


if __name__ == '__main__':
    svm()
