import pandas as pd

import linear_regression

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

df = pd.read_csv('taxi_dataset_with_predictions.csv', index_col=0)

X = df.drop('trip_duration', axis=1)
Y = df['trip_duration']

if __name__ == '__main__':
    # mse_and_mae.print_result()
    linear_regression.print_result()
