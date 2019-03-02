import sys
from memory_profiler import profile
import pandas
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_predict, cross_validate
from sklearn import tree
import graphviz
from statistics import mean


@profile
def decision_tree():
    dataframe = pandas.read_csv('data.csv', header=0)

    X = dataframe.drop('ИН', axis=1)
    y = dataframe['ИН']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    model = tree.DecisionTreeRegressor(max_depth=100)
    scores = cross_validate(model, X_train, y_train, cv=10, return_train_score=True)

    predictions = cross_val_predict(model, X_test, y_test, cv=10)

    print('Час побудови моделі (мс): ', mean(scores['fit_time']))
    print('Об’єм пам’яті, займаємий моделю (MiB): ', sys.getsizeof(model))  # XXX
    print('Помилка моделі для навчальної вибірки (%): ', 100 - mean(scores['train_score']) * 100)
    print('Помилка моделі для тестової вибірки (%): ', 100 - mean(scores['test_score']) * 100)

    plt.scatter(y_test, predictions)
    plt.xlabel("Реальні значення")
    plt.ylabel("Спрогнозовані значення")
    plt.show()

    model.fit(X_train, y_train)
    graph_data = tree.export_graphviz(model, out_file=None, filled=True)
    graph = graphviz.Source(graph_data)
    graph.render("children", view=True)


if __name__ == '__main__':
    decision_tree()
