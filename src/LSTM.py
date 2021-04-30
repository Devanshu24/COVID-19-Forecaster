import time

import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from tqdm import trange

sns.set()
from sklearn.metrics import mean_absolute_percentage_error, r2_score

from data import get_pure_cases_df
from model import CoronaVirusPredictor
from utils import sliding_window

model = CoronaVirusPredictor(n_features=1, n_hidden=100, seq_len=28, n_layers=2)

model.train()


def train_model_with_crossval(
    model,
    df,
):

    model = model.float()

    r2_scores, test_error_mse, test_error_mae, test_error_mape = [], [], [], []
    train, test = {}, {}
    train["feats"], train["pred"], train["true"] = {}, {}, {}
    test["feats"], test["pred"], test["true"] = {}, {}, {}

    v = 0
    for split_idx in range(X.shape[0]):
        model = CoronaVirusPredictor(n_features=1, n_hidden=100, seq_len=28, n_layers=2)
        plotme = []
        plotmefake = []
        v += 1
        start = 0
        X_train, y_train = sliding_window(df, 28, 1, start, max_num=3 + split_idx)
        start += torch.numel(X_train) + torch.numel(y_train)
        X_test, y_test = sliding_window(df, 28, 14, start, max_num=2)
        start += torch.numel(X_test) + torch.numel(y_test)

        train["feats"][v] = X_train.numpy()
        test["feats"][v] = X_test.numpy()

        loss_fn = torch.nn.L1Loss(reduction="mean")
        loss_fn2 = torch.nn.MSELoss(reduction="mean")
        loss_fn3 = nn.PoissonNLLLoss(log_input=False)

        optimiser = torch.optim.Adam(model.parameters(), lr=1e-4)
        num_epochs = 1000

        train_hist = np.zeros(num_epochs)
        test_hist = np.zeros(num_epochs)
        plotme.append(
            (torch.cat((X_train, y_train.reshape(-1, 1, 1)), dim=1))
            .flatten()
            .numpy()
            .tolist()
        )
        with trange(num_epochs) as tr:
            for t in tr:
                for i in range(y_train.shape[1]):
                    model.train()
                    model.reset_hidden_state()
                    #             print(f"sadasda", X_train.shape)
                    y_pred = model(X_train.float()).squeeze()
                    if t > 9990:
                        print(X_train)
                        print(y_pred)
                        print(y_train)
                        time.sleep(1)
                    if t == num_epochs - 1:
                        plotmefake.append(
                            (torch.cat((X_train, y_pred.reshape(-1, 1, 1)), dim=1))
                            .detach()
                            .flatten()
                            .numpy()
                            .tolist()
                        )

                    loss = loss_fn2(y_pred.float().flatten(), y_train[:, i].float())
                    train_hist[t] = loss.item()

                    optimiser.zero_grad()

                    loss.backward()

                    optimiser.step()
                    tr.set_postfix({"loss": f"(  {loss.item()}  )"})

        with torch.no_grad():
            model.eval()
            y_train_pred = model(X_train.float()).squeeze()

            train["pred"][v] = y_train_pred.numpy().tolist()
            train["true"][v] = y_train.numpy().tolist()

            test["pred"][v] = []
            test["true"][v] = y_test.numpy().tolist()
            temae = []
            temse = []
            temape = []
            r2 = []
            y_test_pred_full = []
            for i in range(y_test.shape[1]):
                y_test_pred = model(X_test.float())
                X_test = torch.cat(
                    (X_test[:, 1:], y_test_pred[:, 0].reshape(-1, 1, 1)), dim=1
                )
                temae.append(
                    loss_fn(
                        y_test_pred.float().flatten(), y_test[:, i].flatten()
                    ).item()
                )
                temse.append(
                    loss_fn2(
                        y_test_pred.float().flatten(), y_test[:, i].flatten()
                    ).item()
                )
                r2.append(
                    r2_score(
                        (y_test[:, i]).numpy().flatten(), (y_test_pred).float().numpy()
                    )
                )
                temape.append(
                    mean_absolute_percentage_error(
                        (y_test[:, i]).numpy().flatten(), (y_test_pred).float().numpy()
                    )
                )
            print(np.array(temae).mean())
            test_error_mae.append(np.array(temae).mean())
            test_error_mse.append(np.array(temse).mean())
            r2_scores.append(np.array(r2).mean())
            test_error_mape.append(np.array(temape).mean())

    return (
        np.array(r2_scores),
        np.array(test_error_mae),
        np.array(test_error_mse),
        np.array(test_error_mape),
        train,
        test,
        plotme,
        plotmefake,
    )


if __name__ == "__main__":
    df = get_pure_cases_df()
    X, y = sliding_window(df, 28, 14)
    (
        r2_scores,
        test_error_mae,
        test_error_mse,
        test_error_mape,
        train,
        test,
        plotme,
        plotmefake,
    ) = train_model_with_crossval(
        model,
        df,
    )
    print(
        r2_scores[:-1].mean(),
        test_error_mae.mean(),
        np.sqrt(test_error_mse).mean(),
        test_error_mape.mean(),
    )
