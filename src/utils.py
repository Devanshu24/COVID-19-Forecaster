import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


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
