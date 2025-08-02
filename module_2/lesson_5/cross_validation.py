import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

initial_data = pd.read_csv('initial_data.csv', index_col='id')

initial_cols = ['vendor_id', 'passenger_count', 'pickup_longitude',
                'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',
                'trip_duration']

initial_data = initial_data[initial_cols]
# Добавим новую колонку log_trip_duration, где значения равны log(y_i+1) от значений колонки trip_duration
initial_data = initial_data.assign(log_trip_duration=np.log1p(initial_data['trip_duration']))
initial_data = initial_data.drop('trip_duration', axis=1)

X = initial_data.drop('log_trip_duration', axis=1)
y = initial_data['log_trip_duration']

# Выделим test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Применим K-Fold на оставшуюяся валидационную часть X_train, y_train
splitter = KFold(n_splits=20, shuffle=True, random_state=33)

# Список Loss-значений для каждой из 20 тестовой-валидационной выборки
losses_test = []
# Список Loss-значений для каждой из 20 тренеровочной-валидационной выборки
losses_train = []

for train_index, test_index in splitter.split(X_train):
    X_subtrain, X_valid = X_train.values[train_index], X_train.values[test_index]
    Y_subtrain, Y_valid = y_train.values[train_index], y_train.values[test_index]
    # Строим модель для одного разбиения K-fold
    model = LinearRegression()
    model.fit(X_subtrain, Y_subtrain)
    # MSE на validation-выборке
    losses_test.append(np.mean((model.predict(X_valid) - Y_valid) ** 2))
    # MSE на subtrain-выборке
    losses_train.append(np.mean((model.predict(X_subtrain) - Y_subtrain) ** 2))

# Получим среднее значение качества для всех моделей (прогнав на валидационных данных) на кросс-валидации
cross_validate_test_1 = np.mean(losses_test)

# Альтернативный метод "из коробки" для нахождения среднего значения качества на на кросс-валидации
cv_result = cross_validate(model, X_train, y_train,
                           scoring='neg_mean_squared_error',
                           cv=splitter, return_train_score=True)
cv_result = np.mean(-cv_result['test_score'])

# Теперь построим модель на всей тренировочной выборке
model = LinearRegression()
model.fit(X_train, y_train)
# и замерим качество на тесте!
losses_main_test_1 = np.mean((model.predict(X_test) - y_test) ** 2)

""" Модель #2 """
processed_data = pd.read_csv('processed_data.csv', index_col='id')
# Замерять будем MSLE-качество по формуле MSLE(X,y,a) = 1/n Sum_i=1_l (log(y_i+1)-log(a(x_i)+1))^2
# Можно показать, что для оптимизации MSLE,
# Достаточно логарифмировать таргетную переменную,
# а потом оптимизировать привычные MSE
processed_data = processed_data.assign(log_trip_duration=np.log1p(processed_data['trip_duration']))
processed_data = processed_data.drop('trip_duration', axis=1)

X_2 = processed_data.drop('log_trip_duration', axis=1)
y_2 = processed_data['log_trip_duration']

# Важно! Когда сравниваем модели по их качеству на валидации и на тесте, не шаффлим данные заново!
test_indexes = X_test.index
train_indexes = X_train.index

X_train_2 = X_2[X_2.index.isin(train_indexes)]
y_train_2 = y_2[y_2.index.isin(train_indexes)]

X_test_2 = X_2[X_2.index.isin(test_indexes)]
y_test_2 = y_2[y_2.index.isin(test_indexes)]

losses_test_2 = []
for train_index, test_index in splitter.split(X_train_2):
    X_subtrain, X_valid = X_train_2.values[train_index], X_train_2.values[test_index]
    Y_subtrain, Y_valid = y_train_2.values[train_index], y_train_2.values[test_index]
    # Строим модель для одного разбиения K-fold
    model = LinearRegression()
    model.fit(X_subtrain, Y_subtrain)
    # MSLE на validation-выборке
    losses_test_2.append(np.mean((model.predict(X_valid) - Y_valid) ** 2))

cross_validate_test_2 = np.mean(losses_test_2)

# Теперь построим модель на всей тренировочной выборке
model = LinearRegression()
model.fit(X_train_2, y_train_2)
# и замерим качество на тесте!
losses_main_test_2 = np.mean((model.predict(X_test_2) - y_test_2) ** 2)


def print_result():
    print(f"cross_validate_test_1 is {cross_validate_test_1}")
    print(f"cv_result is {cv_result}")
    print(f"losses_main_test_1 is {losses_main_test_1}")
    print(f"cross_validate_test_2 is {cross_validate_test_2}")
    print(f"losses_main_test_2 is {losses_main_test_2}")
