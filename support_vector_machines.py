import sys
from memory_profiler import profile
import pandas
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import RobustScaler


@profile
def svm():
    dataframe = pandas.read_csv('data.csv', header=0)

    X = dataframe.drop('ИН', axis=1)
    y = dataframe['ИН']

    rbX = RobustScaler()
    X = rbX.fit_transform(X)
    rbY = RobustScaler()
    y = rbY.fit_transform(y.values.reshape(-1, 1))
    # y = column_or_1d(y, warn=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    start = time.time()
    # kernel: rbf, sigmoid, linear, poly   gamma=0.1, C=100, epsilon=0.1
    model = SVR(kernel='rbf')
    model.fit(X_train, y_train.ravel())
    end = time.time()

    predictions = model.predict(rbX.transform(X_test))
    # predictions = rbY.inverse_transform(predictions.reshape(-1, 1))

    print('Час побудови моделі (мс): ', end - start)
    print('Об’єм пам’яті, займаємий моделю (MiB): ', sys.getsizeof(model))
    print('Помилка моделі для навчальної вибірки (%): ', 100 - model.score(X_train, y_train) * 100)
    print('Помилка моделі для тестової вибірки (%): ', 100 - model.score(X_test, y_test) * 100)

    # y_test = rbY.inverse_transform(y_test)

    plt.scatter(y_test, predictions)
    plt.xlabel("Реальні значення")
    plt.ylabel("Спрогнозовані значення")
    plt.show()


if __name__ == '__main__':
    svm()
