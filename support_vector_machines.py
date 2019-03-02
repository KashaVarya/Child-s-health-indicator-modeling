import sys
from memory_profiler import profile
import pandas
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_predict, cross_validate
from sklearn.svm import SVR
from sklearn.preprocessing import RobustScaler
from statistics import mean
import pickle


@profile
def svm():
    dataframe = pandas.read_csv('data.csv', header=0)

    X = dataframe.drop('ИН', axis=1)
    y = dataframe['ИН']

    rbX = RobustScaler()
    X = rbX.fit_transform(X)
    rbY = RobustScaler()
    y = rbY.fit_transform(y.values.reshape(-1, 1))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    # kernel: rbf, sigmoid, linear, poly   gamma=0.1, C=100, epsilon=0.1
    model = SVR(kernel='rbf', gamma='auto')

    scores = cross_validate(model, X_train, y_train.ravel(), cv=10, return_train_score=True)
    predictions = cross_val_predict(model, rbX.transform(X_test), y_test.ravel(), cv=10)

    print(scores)

    p = pickle.dumps(model)
    model_size = sys.getsizeof(p)
    print('Час побудови моделі (мс): ', mean(scores['fit_time']))
    print('Об’єм пам’яті, займаємий моделю (MiB): ', model_size)
    print('Помилка моделі для навчальної вибірки (%): ', 100 - mean(scores['train_score']) * 100)
    print('Помилка моделі для тестової вибірки (%): ', 100 - mean(scores['test_score']) * 100)

    plt.scatter(y_test, predictions)
    plt.xlabel("Реальні значення")
    plt.ylabel("Спрогнозовані значення")
    plt.show()


if __name__ == '__main__':
    svm()
