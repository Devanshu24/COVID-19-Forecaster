import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit
from tqdm import trange

from data import Data
from model import CoronaVirusPredictor

config = {
    "num_epochs": 1000,
    "features": ["new_cases", "7_day_lagged_parks_percent_change_from_baseline"],
}


def train_model(model, config, dataset: Data, test_data=None, test_labels=None):
    loss_fn = torch.nn.MSELoss(reduction="sum")

    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = config["num_epochs"]

    X_train, y_train = dataset.sliding_window(
        config["features"], dataset.previous_days, dataset.forecast_days - 1
    )

    train_hist = np.zeros(num_epochs)
    test_hist = np.zeros(num_epochs)

    for t in trange(num_epochs):
        model.reset_hidden_state()

        y_pred = model(X_train)
        loss = loss_fn(y_pred.float().flatten(), y_train.flatten())

        if test_data is not None:
            with torch.no_grad():
                y_test_pred = model(test_data)
                test_loss = loss_fn(y_test_pred.float(), test_labels)
                test_hist[t] = test_loss.item()

        train_hist[t] = loss.item()

        optimiser.zero_grad()

        loss.backward()

        optimiser.step()

    return model.eval(), train_hist, test_hist


def train_model_with_crossval(
    model: torch.nn.Module,
    dataset: Data,
    config: dict,
    tscv: TimeSeriesSplit,
):
    model = CoronaVirusPredictor(2, 28, 1)
    model = model.float()

    X, y = dataset.sliding_window(
        config["features"], dataset.previous_days, dataset.forecast_days - 1
    )
    print(X.shape, y.shape)
    X = X / 100000
    y = y / 100000

    r2_scores, test_error_mse, test_error_mae = [], [], []
    train, test = {}, {}
    train["feats"], train["pred"], train["true"], train["loss"] = {}, {}, {}, {}
    test["feats"], test["pred"], test["true"], test["loss"] = {}, {}, {}, {}
    lens = {}

    v = 0
    for train_index, test_index in tscv.split(X):
        model = CoronaVirusPredictor(2, 28, 1)
        model = model.float()
        v += 1
        #         print(train_index.shape, test_index.shape)
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        # lens[v] = X_train.shape[0]*X_train.shape[1] +

        # sc = MinMaxScaler()
        # sc.fit(X_train.numpy(), y_train.numpy())
        # X_train, y_train = sc.transform(X_train.numpy(), y_train.numpy())
        # X_test, y_test = sc.transform(X_test, y_test)
        print(X_train.shape)
        print(y_train.shape)
        print(X_test.shape)
        print(y_test.shape)
        train["feats"][v] = X_train.numpy()
        test["feats"][v] = X_test.numpy()

        loss_fn = torch.nn.L1Loss()
        loss_fn2 = torch.nn.MSELoss()

        optimiser = torch.optim.Adam(model.parameters(), lr=1e-5)
        num_epochs = config["num_epochs"]

        train_hist = np.zeros(num_epochs)
        test_hist = np.zeros(num_epochs)
        train["loss"][v] = []
        test["loss"][v] = []
        with trange(num_epochs) as gg:
            for t in gg:
                model.train()
                model.reset_hidden_state()
                y_pred = model(X_train)
                loss = loss_fn2(y_pred.float().flatten(), y_train.float().flatten())

                gg.set_postfix({"loss": loss.item()})
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                train_hist[t] = loss.item()
                train["loss"][v].append(loss.item())

        with torch.no_grad():
            model.eval()
            y_train_pred = model(X_train.float())
            # print(y_train_pred.shape)
            train["pred"][v] = y_train_pred.numpy().tolist()
            train["true"][v] = y_train.numpy().tolist()
            y_test_pred = model(X_test.float())
            # print(y_train.numpy().flatten().tolist())
            # print(y_train_pred.numpy().flatten().tolist())
            test["pred"][v] = y_test_pred.numpy().flatten().tolist()
            test["true"][v] = y_test.numpy().flatten().tolist()

            test_error_mae.append(
                loss_fn(y_test_pred.float().flatten(), y_test.flatten()).item()
            )
            test_error_mse.append(
                loss_fn2(y_test_pred.float().flatten(), y_test.flatten()).item()
            )
            # print(y_test.numpy().flatten())
            # print(y_test_pred.numpy().flatten())
            # print(loss.item())
            r2_scores.append(
                r2_score(y_test.numpy().flatten(), y_test_pred.numpy().flatten())
            )
    #     print(v)

    return r2_scores, test_error_mae, test_error_mse, train, test

    # Create a cross val dataset


if __name__ == "__main__":
    # df = pd.DataFrame({"new cases": np.arange(200)})
    print("I am starting!\n")
    df = pd.read_csv("../data/India_OWID_with_mobility_data.csv")
    dataset = Data(df, 28, 1)
    # dataset.preprocess()
    # dataset.smoothen_df()
    print(dataset.get_features(config["features"]).count())
    X, y = dataset.sliding_window(
        config["features"], dataset.previous_days, dataset.forecast_days - 1
    )
    # print(dataset.df)
    print(len(X))
    print(X.shape)
    print(y.shape)
    # exit(0)
    model = CoronaVirusPredictor(2, 28, 1)
    tscv = TimeSeriesSplit()
    r2_scores, test_error_mae, test_error_mse, train, test = train_model_with_crossval(
        model, dataset, config, tscv
    )
    for i in range(1, 6):
        plt.plot(train["loss"][i], label=i)
    plt.legend(loc="best")
    plt.show()
    # print(test["true"][1])
    for i in range(1, 6):
        #     fig, ax = plt.subplots(figsize=(6,6))
        plt.plot(
            list(range(0, len(train["pred"][i]))),
            train["pred"][i],
            label="Train Predicted",
        )
        plt.plot(
            list(range(0, len(train["true"][i]))), train["true"][i], label="Train True"
        )
        plt.plot(
            list(
                range(
                    len(train["pred"][i]), len(train["pred"][i]) + len(test["pred"][i])
                )
            ),
            test["pred"][i],
            label="Test Predicted",
            linestyle="dotted",
        )
        plt.plot(
            list(
                range(
                    len(train["true"][i]), len(train["true"][i]) + len(test["true"][i])
                )
            ),
            test["true"][i],
            label="Test True",
            linestyle="dotted",
        )
        plt.legend()
        plt.show()

    exit(0)
    gg, train_hist, test_hist = train_model(model, config, dataset)
    plt.plot(train_hist, label="train")
    plt.plot(test_hist, label="test")
    plt.legend(loc="best")
    plt.show()
