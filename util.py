import sys
import matplotlib.pyplot as plt
from statistics import mean
import pickle


def method_results(model, scores, y_test, predictions, params_shape):
    print(scores)

    model_size = sys.getsizeof(pickle.dumps(model))
    print('Час побудови моделі (мс): ', mean(scores['fit_time']))
    print('Об’єм пам’яті, займаємий моделю (bytes): ', model_size)
    print('Кількість параметрів моделі, які можливо налаштувати (ваги): ', params_shape)
    print('Помилка моделі для навчальної вибірки (%): ', 100 - mean(scores['train_score']) * 100)
    print('Помилка моделі для тестової вибірки (%): ', 100 - mean(scores['test_score']) * 100)

    plt.scatter(y_test, predictions)
    plt.xlabel("Реальні значення")
    plt.ylabel("Спрогнозовані значення")
    plt.show()
