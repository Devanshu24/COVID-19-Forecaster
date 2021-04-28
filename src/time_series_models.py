import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from fbprophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#################
# Model: SARIMAX
#################


def run_sarimax(df, window=42, train=28, test=14, plot=False):
    new_cases = df["new_cases"]
    r2_score_val = []
    mae = []
    mse = []
    metrics = {}

    for i in range(0, len(new_cases), window):
        X_train = new_cases[i : i + train]
        X_test = new_cases[(i + train) : (i + train + test)]
        if len(X_test) < test:
            break
        mod = sm.tsa.statespace.SARIMAX(
            X_train,
            order=(7, 1, 0),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        results = mod.fit()
        forecast = results.get_forecast(steps=test).predicted_mean
        if plot:
            plt.plot(range(len(X_test)), X_test, label="Actual")
            plt.plot(range(len(forecast)), forecast, label="Predicted")
            plt.legend()
            plt.show()
        r2_score_val.append(r2_score(X_test, forecast))
        mae.append(mean_absolute_error(X_test, forecast))
        mse.append(mean_squared_error(X_test, forecast))

    metrics["r2"] = np.mean(r2_score_val)
    metrics["mse"] = np.mean(mse)
    metrics["mae"] = np.mean(mae)
    return metrics


def fb_prophet(df, window=42, train=28, test=14):
    new_cases = df["new_cases"]
    window = 42
    train = 28
    test = 14
    r2_score_val = []
    mae = []
    mse = []
    metrics = {}

    for i in range(0, len(new_cases), window):
        X_train = new_cases[i : i + train]
        X_test = new_cases[(i + train) : (i + train + test)]
        if len(X_test) < test:
            break
        fb_data = pd.DataFrame({"y": X_train, "ds": df.index.values[i : i + train]})
        m = Prophet()
        m.fit(fb_data)
        future = m.make_future_dataframe(periods=test)
        forecast = m.predict(future)
        y_pred = forecast[-14:]["yhat"]
        mse.append(mean_squared_error(X_test, y_pred))
        mae.append(mean_absolute_error(X_test, y_pred))
        r2_score_val.append(r2_score(X_test, y_pred))

    metrics["r2"] = np.mean(r2_score_val)
    metrics["mse"] = np.mean(mse)
    metrics["mae"] = np.mean(mae)
    return metrics
