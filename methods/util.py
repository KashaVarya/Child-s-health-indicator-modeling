import sys
import matplotlib.pyplot as plt
from statistics import mean
import pickle


def method_results(
        model,
        scores,
        y_test,
        predictions,
        params_shape,
        predict_time,
        train_set
):
    # print(scores)

    model_size = sys.getsizeof(pickle.dumps(model))
    print('Час побудови моделі (мс): ', mean(scores['fit_time']) * 1000)
    print('Час роботи моделі (мс): ', predict_time * 1000)
    print('Об’єм пам’яті, займаємий моделю (bytes): ', model_size)
    print('Кількість параметрів моделі, які можливо налаштувати (ваги): ', params_shape)

    try:
        if type(params_shape) == int:
            K0 = train_set.shape[1] * train_set.shape[0] / params_shape
        else:
            K0 = train_set.shape[1] * train_set.shape[0] / 8300
        print('Коэффициент обобщения моделью обучающих данных Ko: ', K0)
    except Exception:
        pass

    print('Помилка моделі для навчальної вибірки (%): ', 100 - mean(scores['train_score']) * 100)
    print('Помилка моделі для тестової вибірки (%): ', 100 - mean(scores['test_score']) * 100)

    plt.scatter(y_test, predictions)
    plt.xlabel("Реальні значення")
    plt.ylabel("Спрогнозовані значення")
    plt.show()
