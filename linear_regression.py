import sys
from memory_profiler import profile
import pandas
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


@profile
def linear_regression():
    dataframe = pandas.read_csv('data.csv', header=0)

    X = dataframe.drop('ИН', axis=1)
    y = dataframe['ИН']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    start = time.time()
    model = LinearRegression()
    model.fit(X_train, y_train)
    end = time.time()

    predictions = model.predict(X_test)

    print('Час побудови моделі (мс): ', end - start)
    print('Об’єм пам’яті, займаємий моделю (MiB): ', sys.getsizeof(model))
    print('Помилка моделі для навчальної вибірки (%): ', 100 - model.score(X_train, y_train) * 100)
    print('Помилка моделі для тестової вибірки (%): ', 100 - model.score(X_test, y_test) * 100)

    plt.scatter(y_test, predictions)
    plt.xlabel("Реальні значення")
    plt.ylabel("Спрогнозовані значення")
    plt.show()


if __name__ == '__main__':
    linear_regression()
