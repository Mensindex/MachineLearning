import numpy as np
from sklearn.linear_model import LinearRegression

from main import X
from main import Y

print(X['prediction_2'])
X = X.drop(['pickup_datetime', 'prediction_1', 'prediction_2', ], axis=1)
model = LinearRegression()
model.fit(X, Y)


def get_coef():
    result = []
    for coef_ in model.coef_:
        result.append(round(coef_, 3))
    return result


def linear_regression_by_matrix(X: np.array, Y: np.array, fit_intercept: bool = True):
    if fit_intercept:
        X['constant'] = 1
    xxt = np.dot(X.T, X)
    xxt_inv = np.linalg.inv(xxt)
    xxt_inv_xxt = np.dot(xxt_inv, X.T)
    final_betas = np.dot(xxt_inv_xxt, Y)
    return final_betas


def print_result():
    print(get_coef())
    print(linear_regression_by_matrix(X=X, Y=Y))
    print(np.dot(X, linear_regression_by_matrix(X=X, Y=Y)))
