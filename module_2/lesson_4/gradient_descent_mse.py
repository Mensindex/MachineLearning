import numpy as np
import pandas as pd


class GradientDescentMse:
    """
        Базовый класс для реализации градиентного спуска в задаче линейной МНК регрессии
        """

    def __init__(self, samples: pd.DataFrame, targets: pd.DataFrame,
                 learning_rate: float = 1e-3, threshold=1e-6, copy: bool = True):
        """
                :param samples: матрица объектов
                :param targets: вектор (матрица с 1 столбцом) ответов
                :param learning_rate: параметр learning_rate для корректировки нормы градиента
                :param threshold: величина, меньше которой изменение в loss-функции означает
                :param copy: копировать сэмплы или делать изменения in-place (см. add_constant_feature)
                """
        if copy:
            self.samples = samples.copy()
        else:
            self.samples = samples
        self.targets = targets
        self.learning_rate = learning_rate
        self.threshold = threshold
        # массив весов β
        self.beta = np.ones(self.samples.shape[1])
        self.iteration_num = 0
        self.iteration_loss_dict = {}

    def add_constant_feature(self):
        """
                Метод для создания константной фичи в матрице объектов samples
                """
        self.samples['constant'] = 1
        # Добавляем так же дополнительную β в массив для соответствия колонке-признаку 'constant'
        self.beta = np.append(self.beta, 1)  # β_0

    def calculate_mse_loss(self) -> float:
        """
        Метод для расчета среднеквадратической ошибки

        :return: среднеквадратическая ошибка при текущих весах модели : float
        """
        # Q(a(x), X) = 1/N ⋅ SUM_i_N (β_1⋅d_i1+...+β_n⋅d_in-y_i)^2
        loss = np.dot(self.samples,
                      self.beta) - self.targets.values
        return np.mean(loss ** 2)

    def calculate_gradient(self) -> np.ndarray:
        """
        Метод для вычисления вектора-градиента
        Метод возвращает вектор-градиент, содержащий производные по каждому признаку.
        Сначала матрица признаков скалярно перемножается на вектор self.beta, и из каждой колонки
        полученной матрицы вычитается вектор таргетов. Затем полученная матрица скалярно умножается на матрицу признаков.
        Наконец, итоговая матрица умножается на 2 и усредняется по каждому признаку.

        :return: вектор-градиент, т.е. массив, содержащий соответствующее количество производных по каждой переменной : np.ndarray
        """
        # β_1⋅d_i1+...+β_n⋅d_in-y_i
        shift = np.dot(self.samples, self.beta) - self.targets
        # Q'β_1 = 2/n ⋅ SUM_i_N d_i1⋅(β_1⋅d_i1+...+β_n⋅d_in-y_i)
        derivatives = 2 * np.dot(shift, self.samples) / self.samples.shape[0]
        return derivatives

    def iteration(self):
        """
                Обновляем веса модели в соответствии с текущим вектором-градиентом
                """
        self.beta = self.beta - self.learning_rate * self.calculate_gradient()  # β_next = β_start-η⋅∇Q(β_start)

    def learn(self):
        """
        Итеративное обучение весов модели до срабатывания критерия останова
        Запись mse и номера итерации в iteration_loss_dict

        Описание алгоритма работы для изменения бет:
            Фиксируем текущие beta -> start_betas
            Делаем шаг градиентного спуска
            Записываем новые beta -> new_betas
            Пока |L(new_beta) - L(start_beta)| >= threshold:
                Повторяем первые 3 шага

        Описание алгоритма работы для изменения функции потерь:
            Фиксируем текущие mse -> previous_mse
            Делаем шаг градиентного спуска
            Записываем новые mse -> next_mse
            Пока |(previous_mse) - (next_mse)| >= threshold:
                Повторяем первые 3 шага
        """
        previous_mse = self.calculate_mse_loss()

        self.iteration()

        next_mse = self.calculate_mse_loss()

        self.iteration_loss_dict[0] = previous_mse
        self.iteration_loss_dict[1] = next_mse

        self.iteration_num = 1

        while abs(next_mse - previous_mse) >= self.threshold:
            previous_mse = next_mse

            self.iteration()

            next_mse = self.calculate_mse_loss()

            self.iteration_loss_dict[self.iteration_num + 1] = next_mse

            self.iteration_num += 1
