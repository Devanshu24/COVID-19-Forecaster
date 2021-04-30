import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)

from data import get_pure_cases_df
from utils import sliding_window


def print_score():
    df = get_pure_cases_df()
    X, y = sliding_window(df, 28, 14)
    assert X.shape[0] == y.shape[0]
    X = X.numpy()
    y = y.numpy()
    mae = []
    rmse = []
    mape = []
    for i in range(X.shape[0]):
        mae.append(
            mean_absolute_error(y[i], np.broadcast_to(X[i][-1], (y.shape[1], 1)))
        )
        rmse.append(
            mean_squared_error(
                y[i], np.broadcast_to(X[i][-1], (y.shape[1], 1)), squared=False
            )
        )
        mape.append(
            mean_absolute_percentage_error(
                y[i], np.broadcast_to(X[i][-1], (y.shape[1], 1))
            )
        )

    mae = np.array(mae)
    rmse = np.array(rmse)
    mape = np.array(mape)
    print(f"RMSE: {rmse.mean()}, MAE: {mae.mean()}, MAPE: {mape.mean()}")
