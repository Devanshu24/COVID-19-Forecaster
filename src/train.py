import numpy as np
import pandas as pd
import torch

from data import Data
from model import CoronaVirusPredictor

config = {
    "num_epochs": 60,
    "forecast_days": 7,
    "features": ["new cases"],
    "previous_days": 10,
}


def train_model(model, config, dataset: Data, test_data=None, test_labels=None):
    loss_fn = torch.nn.MSELoss(reduction="sum")

    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = config["num_epochs"]

    X_train, y_train = dataset.sliding_window(
        config["features"], config["previous_days"], config["forecast_days"]
    )

    train_hist = np.zeros(num_epochs)
    test_hist = np.zeros(num_epochs)

    for t in range(num_epochs):
        model.reset_hidden_state()

        y_pred = model(X_train)

        loss = loss_fn(y_pred.float().flatten(), y_train)

        if test_data is not None:
            with torch.no_grad():
                y_test_pred = model(test_data)
                test_loss = loss_fn(y_test_pred.float(), test_labels)
                test_hist[t] = test_loss.item()

            if t % 10 == 0:
                print(
                    f"Epoch {t} train loss: {loss.item()} test loss: {test_loss.item()}"
                )
        elif t % 10 == 0:
            print(f"Epoch {t} train loss: {loss.item()}")

        train_hist[t] = loss.item()

        optimiser.zero_grad()

        loss.backward()

        optimiser.step()

    return model.eval(), train_hist, test_hist


if __name__ == "__main__":
    df = pd.read_csv("../data/India_OWID_reduced.csv")
    # print(df)
    data = Data(df, 10, 1)
    data.preprocess()
    print(data.df.shape)
    X, y = data.sliding_window(["new_cases", "new_deaths"], 2)
    X = torch.tensor(X, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.float)
    print(X.shape, y.shape)
    model = CoronaVirusPredictor(2, 64, 2, 2)
    model = model.float()
    model.reset_hidden_state()
    model = model.float()
    wo = model(X)
