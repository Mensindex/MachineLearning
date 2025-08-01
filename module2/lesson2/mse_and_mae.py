from math import log

from main_2_2 import X
from main_2_2 import Y

error_1 = (((X['prediction_1'] - Y) ** 2).mean())
error_2 = (((X['prediction_2'] - Y) ** 2).mean())

absolute_error_1 = abs(X['prediction_1'] - Y).mean()
absolute_error_2 = abs(X['prediction_2'] - Y).mean()

counter_1 = sum(abs(X['prediction_1'] - Y) >= 500)
counter_2 = sum(abs(X['prediction_2'] - Y) >= 500)

X['prediction_1'] = X['prediction_1'].apply(lambda x: max(x, 0))
X['prediction_2'] = X['prediction_2'].apply(lambda x: max(x, 0))

under_estimating = ((log(90 + 1) - log(100 + 1)) ** 2) ** (1 / 2)
over_estimating = ((log(110 + 1) - log(100 + 1)) ** 2) ** (1 / 2)

msle_1 = (((Y + 1).apply(log) - (X['prediction_1'] + 1).apply(log)) ** 2).mean()
msle_2 = (((Y + 1).apply(log) - (X['prediction_2'] + 1).apply(log)) ** 2).mean()

over_predicted_1 = sum(X['prediction_1'] - Y > 0)
under_predicted_1 = sum(X['prediction_1'] - Y < 0)
over_predicted_2 = sum(X['prediction_2'] - Y > 0)
under_predicted_2 = sum(X['prediction_2'] - Y < 0)


def print_result():
    print(f"MSE первой модели равно: {int(error_1)}")
    print(f"MSE второй модели равно: {int(error_2)}")

    print(f"RMSE первой модели равно: {int(error_1 ** (1 / 2))}")
    print(f"RMSE второй модели равно: {int(error_2 ** (1 / 2))}")

    print(f"MAE первой модели равно: {int(absolute_error_1)}")
    print(f"MAE второй модели равно: {int(absolute_error_2)}")

    print(f"Количество отклонений >= 500 от верного ответа для первой модели равно: {counter_1}")
    print(f"Количество отклонений >= 500 от верного ответа для второй модели равно: {counter_2}")

    print(f"Ошибка RMSLE при недопредсказании: {round(under_estimating, 3)}")
    print(f"Ошибка RMSLE при перепредсказании: {round(over_estimating, 3)}")

    print(f"RMSLE первой модели равно: {msle_1 ** (1 / 2)}")
    print(f"RMSLE второй модели равно: {msle_2 ** (1 / 2)}")

    print(f"Предсказания первой модели оказались больше действительных в {over_predicted_1} случаях")
    print(f"Предсказания первой модели оказались меньше действительных в {under_predicted_1} случаях")
    print(f"Предсказания второй модели оказались больше действительных в {over_predicted_2} случаях")
    print(f"Предсказания второй модели оказались меньше действительных в {under_predicted_2} случаях")
