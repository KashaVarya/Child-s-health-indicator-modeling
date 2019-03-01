import sys
from memory_profiler import profile
import pandas
import time
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import tree
import graphviz


@profile
def decision_tree():
    dataframe = pandas.read_csv('data.csv', header=0)

    X = dataframe.drop('ИН', axis=1)
    y = dataframe['ИН']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    start = time.time()
    model = tree.DecisionTreeRegressor(max_depth=100)
    model.fit(X_train, y_train)
    end = time.time()

    regr_2 = tree.DecisionTreeRegressor(max_depth=200)
    regr_2.fit(X_train, y_train)

    predictions = model.predict(X_test)
    y_2 = regr_2.predict(X_test)

    print('Час побудови моделі (мс): ', end - start)
    print('Об’єм пам’яті, займаємий моделю (MiB): ', sys.getsizeof(model))
    print('Помилка моделі для навчальної вибірки (%): ', 100 - model.score(X_train, y_train) * 100)
    print('Помилка моделі для тестової вибірки (%): ', 100 - model.score(X_test, y_test) * 100)

    plt.scatter(y_test, predictions)
    plt.xlabel("Реальні значення")
    plt.ylabel("Спрогнозовані значення")
    plt.show()

    graph_data = tree.export_graphviz(model, out_file=None, filled=True)
    graph = graphviz.Source(graph_data)
    graph.render("breast_cancer", view=True)


if __name__ == '__main__':
    decision_tree()
