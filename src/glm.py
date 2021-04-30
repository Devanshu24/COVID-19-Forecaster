import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)


def get_glm_prediction():
    confirmed_cases_url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"  # noqa
    confirmed_cases = pd.read_csv(confirmed_cases_url, sep=",")
    gg = confirmed_cases[confirmed_cases["Country/Region"] == "India"]
    gg = gg.T
    gg["cases"] = gg[147]
    df = gg.drop(147, axis=1).iloc[4:]
    df = df.diff().iloc[1:]
    rmse = []
    mae = []
    mapes = []
    arr = df["cases"].to_numpy(dtype=np.float64)
    df["one_old"] = df["cases"].shift(1)
    df["two_old"] = df["cases"].shift(2)
    df["three_old"] = df["cases"].shift(3)
    df = df[5:]
    x_train = df[["one_old", "two_old", "three_old"]]
    y_train = df["cases"]
    x_test = x_train[275:300]
    y_test = y_train[275:300]
    x_train = x_train[:275]
    y_train = y_train[:275]
    xtrain = x_train.to_numpy(dtype=np.float64)
    ytrain = y_train.to_numpy(dtype=np.float64)
    xtest = x_test.to_numpy(dtype=np.float64)
    ytest = y_test.to_numpy(dtype=np.float64)

    poisson_training_results = sm.GLM(ytrain[:], xtrain[:, :]).fit()

    poisson_predictions = poisson_training_results.get_prediction(xtest)
    predictions_summary_frame = poisson_predictions.summary_frame()
    predictions_summary_frame["Actual"] = ytest
    predictions_summary_frame["mean"] = predictions_summary_frame["mean"]

    for i in range(14):
        xtest = np.hstack(
            (np.array(predictions_summary_frame["mean"]).reshape(-1, 1), xtest[:, :-1])
        )
        ytest = np.vstack(
            (
                ytest[1:].reshape(-1, 1),
                (np.array(df["cases"][300 + i : 300 + i + 1], dtype="float")).reshape(
                    -1, 1
                ),
            )
        )
        predictions_summary_frame["mean"] = poisson_training_results.get_prediction(
            xtest
        ).summary_frame()["mean"]
        rmse.append(
            mean_squared_error(
                (predictions_summary_frame["mean"]), (ytest), squared=False
            )
        )
        mae.append(mean_absolute_error((predictions_summary_frame["mean"]), (ytest)))
        mapes.append(
            mean_absolute_percentage_error(predictions_summary_frame["mean"], ytest)
        )

    rmse = np.array(rmse)
    mae = np.array(mae)
    mapes = np.array(mapes)
    print(f"RMSE: {rmse.mean()}, MAE: {mae.mean()}, MAPE: {mapes.mean()}")
