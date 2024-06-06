import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)
taxiDB = pd.read_csv('taxi_dataset.csv')

taxiDB['pickup_datetime'] = pd.to_datetime(taxiDB['pickup_datetime'])
taxiDB['dropoff_datetime'] = pd.to_datetime(taxiDB['dropoff_datetime'])
taxiDB['trip_duration'] = (taxiDB['dropoff_datetime'] - taxiDB['pickup_datetime']).dt.total_seconds()

taxiDB = taxiDB.drop('dropoff_datetime', axis=1)

taxiDB['vendor_id'] = taxiDB['vendor_id'] - 1

taxiDB['store_and_fwd_flag'] = taxiDB['store_and_fwd_flag'].map(pd.Series({'N': 0, 'Y': 1}))

# Соберем список из всех широт (как точек старта, так и конца)
allLat = list(taxiDB['pickup_latitude']) + list(taxiDB['dropoff_latitude'])

# Посчитаем медиану:
# Это некоторое "Центральное значение" в отсортированном массиве всех значений
# Иными словами, такое число, меньше и больше которого примерно равное количество объектов
medianLat = sorted(allLat)[int(len(allLat) / 2)]

# Теперь, для из каждого значения широты вычтем медианное значение
# Результат переведем в километры
latMultiplier = 111.32
taxiDB['pickup_latitude'] = latMultiplier * (taxiDB['pickup_latitude'] - medianLat)
taxiDB['dropoff_latitude'] = latMultiplier * (taxiDB['dropoff_latitude'] - medianLat)
# Итого, для latitude колонок получили следующие выражения:
# На сколько примерно километров севернее или южнее (в зависимости от знака) точка находится относительно средней широты

# Используя полученную медиану и множитель, на который стоит корректировать все долготы,
# получим корректные longitude признаки по аналогии
allLong = list(taxiDB['pickup_longitude']) + list(taxiDB['dropoff_longitude'])
medianLong = sorted(allLong)[int(len(allLong) / 2)]
longMultiplier = np.cos(medianLat * (np.pi / 180.0)) * 111.32
taxiDB['pickup_longitude'] = longMultiplier * (taxiDB['pickup_longitude'] - medianLong)
taxiDB['dropoff_longitude'] = longMultiplier * (taxiDB['dropoff_longitude'] - medianLong)


# Почему мы вычисляли через медианы: они позволяют нам во время вычисления расстояния преобразовать изначальные
# longtitude/latitude колонки в "отдаленности точек старта/конца поездок" от медианных точек. Кажется,
# что это прикольно :) Есть подозрение, что медианная для поездок точка города - это, на практике, точка скопления
# вечерних пробок. Нам может быть вполне важно знать, насколько далеко от такого эпицентра ужаса мы начинаем и
# заканчиваем поездку (насколько севернее/южнее/...) и выделить поверх этой информации дополнительные признаки. В
# домашнем задании это использоваться не будет, но это ещё один пример, как можно работать с признаками


# Наконец, вычислим географическое расстояние distance_km:
def calculate_distance_pythagoras(lat1, lon1, lat2, lon2):
    delta_lat = (lat2 - lat1)
    delta_lon = (lon2 - lon1)
    return np.sqrt(delta_lat ** 2 + delta_lon ** 2)


taxiDB['distance_km'] = taxiDB.apply(
    lambda row: calculate_distance_pythagoras(
        lat1=row['pickup_latitude'],
        lon1=row['pickup_longitude'],
        lat2=row['dropoff_latitude'],
        lon2=row['dropoff_longitude'],
    ),
    axis=1,
)

# Уберем старые признаки!
taxiDB = taxiDB.drop(['pickup_longitude', 'dropoff_longitude',
                      'pickup_latitude', 'dropoff_latitude'], axis=1)

# Какой это признак, на ваш взгляд: вещественный, категориальный, порядковый?

# С одной стороны, можно воспринимать его как обычный вещественный признак. Ведь само по себе количество пассажиров
# (без дополнительной обработки) - это некоторое число, которое может принимать большое количество различных значений.

# С другой стороны, мы с Вами наверняка знаем, что количество пассажиров от поездки к поездке ограничено. Вряд ли
# если к нам придут новые данные, мы увидим числа бОльшие, чем у нас в датасете. Тогда рассуждаем следующим образом:
# раз множество значений признака ограничено, то он категориальный (или, в данном случае, даже порядковый! Ведь у нас
# могут быть какие-то логичные предположения о том, что количество пассажиров может влиять на модель машины и,
# соответственно, скорость ее передвижения и скорость поездки!)

# Какой подход выбрать лучше заранее наверняка не узнаешь. Нужны эксперименты с данными и моделями.
# Тем не менее, я предлагаю Вам предположить, что данный признак является категориальным,
# и попробовать отточить навыки кодировки таких фичей!

# Предлагаю Вам реализовать прием с Mean-target encoding'ом, как в практическом занятии.
# Замените колонку passenger_count колонкой category_encoded
taxiDB['passenger_count'] = taxiDB['passenger_count'].map(taxiDB.groupby(['passenger_count'])['trip_duration'].mean())

# Кажется, мы достаточно близки с Вами к тому, чтобы получить в итоге табличку, полностью состояющую из чиселок и,
# казалось бы, осмысленных признаков! Остались две колонки: id, pickup_datetime id можно использовать как обычный
# идентификатор нашего объекта, поэтому поместите данную колонку в качестве индекса нашей таблички:
taxiDB = taxiDB.set_index('id')

if __name__ == '__main__':
    print(taxiDB.head(20))
