import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import trange

from data import Data
from model import CoronaVirusPredictor

config = {
    "num_epochs": 2000,
    "features": ["new cases"],
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


if __name__ == "__main__":
    df = pd.DataFrame({"new cases": np.arange(200)})
    data = Data(df, 10, 7)
    data.preprocess()
    model = CoronaVirusPredictor(1, 10, 7)
    model = model.float()
    gg, train_hist, test_hist = train_model(model, config, data)
    plt.plot(train_hist, label="train")
    plt.plot(test_hist, label="test")
    plt.legend(loc="best")
    plt.show()
