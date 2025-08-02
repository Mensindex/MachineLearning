import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from gradient_descent_mse import GradientDescentMse

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)
data = pd.read_csv('data.csv')

X = data.drop(['target'], axis=1)
Y = data['target']

model = LinearRegression()
model.fit(X, Y)

"""Алгоритм нахождения производной по одной переменной β"""
# Стартовая точка:
initial_betas = np.ones(X.shape[1])
# Перемножаем координаты initial_betas (наши веса β) со всеми признаками каждого объекта β_1⋅d_i1+...+β_n⋅d_in
scalar_value = np.dot(X, initial_betas.reshape(-1, 1)).ravel()
# Находим loss function (MAE) по каждому объекту β_1⋅d_i1+...+β_n⋅d_in-y_i
scalar_value = (scalar_value - Y).values
# Возьмем столбик со значениями 1 признака
d_i1 = X.values[:, 0]
# Умножим каждый объект на соответствующее значение признака d_i1⋅(β_1⋅d_i1+...+β_n⋅d_in-y_i)
scalar_value = scalar_value * d_i1
# Наконец, умножим все на 2 и усреднимся,
# чтобы получить значение производной по первому параметру Q'β_1 = 2/n ⋅ SUM_i_N d_i1⋅(β_1⋅d_i1+...+β_n⋅d_in-y_i)
2 * np.mean(scalar_value, axis=0)


def make_learning_paths_graph():
    fig = plt.figure()

    fig.set_size_inches(13, 10)

    thresholds = [0.01, 0.001, 0.0001, 0.00001]

    rates = [0.1, 0.05, 0.01, 0.005, 0.001]

    # Для каждого из threshold нарисуем по одной системе координат
    for i in range(len(thresholds)):
        thresh = thresholds[i]

        ax_ = fig.add_subplot(2, 2, i + 1)

        Q_values = []

        # В каждой системе координат для определенного threshold нарисуем 5 графиков для каждого learning rate
        for lr in rates:
            # Инициализируем класс, выполняющий градиентный спуск и запускаем алгоритм
            GD = GradientDescentMse(samples=X, targets=Y, learning_rate=lr, threshold=thresh)
            GD.add_constant_feature()
            GD.learn()

            learning_path = GD.iteration_loss_dict
            # Для отрисовки графика даем множество точек по оси x в виде массива итераций
            # и множество точке по оси y в виде массива значений функционала качества Q
            plt.plot(learning_path.keys(), learning_path.values())
            plt.title(f'Threshold = {thresh}')
            plt.ylim(0, 100)
            plt.xlim(0, 2000)

            Q_values.append(str(round(number=list(learning_path.values())[-1], ndigits=4)))

        plt.ylabel('Среднеквадратическая ошибка')
        plt.xlabel('Номер итерации')
        plt.legend([f'Learning rate equals to {rates[i]}' + ' with Q = ' + Q_values[i] for i in range(len(rates))])

    # Дадим отступы между системами координат
    fig.tight_layout()

    plt.show()


def print_result():
    # Сравним значения весов для модели, полученных с помощью sklearn
    # cо значениями весов, полученных с помощью нашего класса GradientDescentMse
    print(model.coef_)
    print(model.intercept_)
    print()
    print()
    print()
    GD = GradientDescentMse(samples=X, targets=Y, learning_rate=0.1, threshold=0.0000000001)
    GD.add_constant_feature()
    GD.learn()
    print('Веса модели при переменных d0, d1, ..., d10 равны соответственно: \n\n' + str(GD.beta))

    # Нарисуем наглядные графики для разных thresholds и learning rates
    make_learning_paths_graph()
