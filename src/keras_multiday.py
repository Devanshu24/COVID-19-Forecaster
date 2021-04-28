import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import pandas as pd
from keras.layers import LSTM, Dense
from keras.models import Sequential
from numpy import array
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import TimeSeriesSplit


def sliding_window(sequence: list, seq_len: int, j=1):
    xs = []
    ys = []
    j -= 1
    for i in range(0, len(sequence) - seq_len - j, seq_len + 1 + j):
        x = sequence[i : (i + seq_len)]
        y = sequence[(i + seq_len) : i + seq_len + 1 + j]
        xs.append(x)
        ys.append(y)
    return array(xs), array(ys)


# split a univariate sequence into samples


def split_sequence(sequence, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the sequence
        if out_end_ix > len(sequence):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


confirmed_cases_url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"  # noqa
df = pd.read_csv(confirmed_cases_url)
df = df[df["Country/Region"] == "India"]
arr = df.drop(
    ["Province/State", "Country/Region", "Lat", "Long"], axis=1
).T.values.flatten()
arr = arr[60:]
# define input sequence
arr = np.diff(arr)
raw_seq = arr[:-42]
# raw_seq = np.arange(400, step = 10)
# choose a number of time steps
n_steps_in, n_steps_out = 28, 14
# split into samples
X, y = sliding_window(raw_seq, n_steps_in, n_steps_out)
print(X.shape, y.shape)
# print(X)
# print(y)
# exit(0)
# reshape from [samples, timesteps] into [samples, timesteps, features]

tscv = TimeSeriesSplit()
print("##############")
for train_index, test_index in tscv.split(X):
    X_train, y_train = (X[train_index]), (y[train_index])
    X_test, y_test = (X[test_index]), (y[test_index])
    print(X_train.shape, y_train.shape)
    # print(X_train, y_train)

    print(X_test.shape, y_test.shape)
    # print(X_test, y_test)

    n_features = 1
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    # define model
    model = Sequential()
    model.add(
        LSTM(
            100,
            activation="relu",
            return_sequences=True,
            input_shape=(n_steps_in, n_features),
        )
    )
    model.add(LSTM(100, activation="relu"))
    model.add(Dense(n_steps_out))
    model.compile(optimizer="adam", loss="mse")
    # fit model
    model.fit(X, y, epochs=50, verbose=0)
    # demonstrate prediction
    x_input = arr[-42:-14]
    x_input = x_input.reshape((1, n_steps_in, n_features))
    yhat = model.predict(x_input, verbose=0)
    print("r2: ", r2_score(yhat.flatten(), arr[-14:]))
    print("mae: ", mean_absolute_error(yhat.flatten(), arr[-14:]))
    print("mse: ", mean_squared_error(yhat.flatten(), arr[-14:]))
    print("mape:", mean_absolute_percentage_error(yhat.flatten(), arr[-14:]))
    print("@@@@@@@@@@@@@@@@@@@@@@@")
