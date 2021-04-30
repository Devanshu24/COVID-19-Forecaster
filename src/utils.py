import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from scipy.signal import savgol_filter

mpl.rcParams.update(mpl.rcParamsDefault)


def smoothen_data(data):
    return savgol_filter(data, 51, 3)


def plot_forecast(train_pred, train_true, test_pred, test_true):
    plt.plot(
        list(range(0, len(train_pred))),
        train_pred,
        label="Train Predicted",
    )
    plt.plot(list(range(0, len(train_true))), train_true, label="Train True")

    plt.plot(
        list(range(len(train_pred), len(train_pred) + len(test_pred))),
        test_pred,
        label="Test Predicted",
        linestyle="dotted",
    )
    plt.plot(
        list(range(len(train_true), len(train_true) + len(test_true))),
        test_true,
        label="Test True",
        linestyle="dotted",
    )

    plt.legend()

    plt.show()


def plot_SIR(trace, data_begin, confirmed_cases):
    sns.set()
    sns.set_context("paper")
    sns.color_palette("tab10")
    num_days_seen = trace["new_I_past"].shape[1]
    num_days_forecast = trace["new_I_no_change"].shape[1]

    plt.figure(figsize=(12, 7))
    percentiles = np.percentile(trace.new_I_past, q=12.5, axis=0), np.percentile(
        trace.new_I_past, q=87.5, axis=0
    )
    plt.plot(
        np.arange(1, num_days_seen + 1),
        np.mean(trace["new_I_past"], axis=0),
        color="tab:orange",
        label="Observed Predicted (75% CI)",
    )
    plt.fill_between(
        np.arange(1, num_days_seen + 1),
        percentiles[0],
        percentiles[1],
        alpha=0.3,
        color="tab:orange",
    )

    percentiles = (
        np.percentile(trace["new_I_no_change"], q=12.5, axis=0),
        np.percentile(trace["new_I_no_change"], q=87.5, axis=0),
    )
    plt.plot(
        np.arange(1, num_days_forecast + 1) + num_days_seen,
        np.mean(trace["new_I_no_change"], axis=0),
        color="tab:green",
        label="Unobserved Predicted (75% CI)",
    )
    plt.fill_between(
        np.arange(1, num_days_forecast + 1) + num_days_seen,
        percentiles[0],
        percentiles[1],
        alpha=0.3,
        color="tab:green",
    )

    plt.plot(
        np.arange(1, num_days_seen + 2),
        np.diff(
            np.array(
                confirmed_cases.loc[
                    confirmed_cases["Country/Region"] == "India", data_begin:"3/13/21"
                ]
            )[0]
        ),
        alpha=0.5,
        color="0",
    )
    plt.plot(
        np.arange(1, num_days_seen + 1),
        np.diff(
            np.array(
                confirmed_cases.loc[
                    confirmed_cases["Country/Region"] == "India", data_begin:"3/12/21"
                ]
            )[0]
        ),
        label="True Observed",
    )

    plt.plot(
        np.arange(1, num_days_forecast + 1) + num_days_seen,
        np.diff(
            np.array(
                confirmed_cases.loc[
                    confirmed_cases["Country/Region"] == "India", "3/12/21":"3/26/21"
                ]
            )[0]
        ),
        "r",
        label="True Unobserved",
    )
    plt.xticks(np.arange(1, 42, step=2))
    plt.title("Prediction using SIR")
    plt.ylabel("Case Counts")
    plt.legend(loc="best")
    plt.savefig("India(1-5--29-5--2020).svg")


def sliding_window(df, seq_len: int, j=0, start=0, max_num=None):
    df = df.drop("date", axis=1)
    df = df.dropna()
    df = df.iloc[start:]
    j -= 1
    xs = []
    ys = []
    num = 0
    for i in range(0, len(df) - seq_len - j, seq_len + 1 + j):
        x = df[i : (i + seq_len)].to_numpy()
        y = df[(i + seq_len) : i + seq_len + 1 + j]["new_cases"].to_numpy()
        xs.append(x)
        ys.append(y)
        num += 1
        if max_num and num >= max_num:
            break
    return torch.tensor(xs, dtype=torch.float64), torch.tensor(ys, dtype=torch.float64)
