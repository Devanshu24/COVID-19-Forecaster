import argparse
import datetime

import numpy as np
import pandas as pd
import pymc3 as pm

from arima_prophet import fb_prophet, run_arima
from data import get_pure_cases_df
from glm import get_glm_prediction
from LSTM import train_model_with_crossval
from model import CoronaVirusPredictor
from persistence import print_score
from SIR import get_model
from utils import plot_SIR, sliding_window

parser = argparse.ArgumentParser(description="Analyse various algorithms")
parser.add_argument(
    "model",
    choices=["lstm", "arima", "fbprophet", "nbeats", "sir", "glm", "persistence"],
    help="Selection of model",
)
args = parser.parse_args()

if args.model == "lstm":
    model = CoronaVirusPredictor(n_features=1, n_hidden=100, seq_len=28, n_layers=2)
    model.train()
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
    ) = train_model_with_crossval(model, df, 100)
    print(
        f"R2: {r2_scores[:-1].mean()},MAE: {test_error_mae.mean()},RMSE: {np.sqrt(test_error_mse.mean())}"
    )
elif args.model == "arima":
    df = get_pure_cases_df()
    arima_metrics = run_arima(df)
    print(f"Arima Metrics: {arima_metrics}")
elif args.model == "fbprophet":
    df = get_pure_cases_df()
    fb_metrics = fb_prophet(df)
    print(f"FbProphet Metrics: {fb_metrics}")
elif args.model == "sir":
    confirmed_cases_url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"  # noqa
    confirmed_cases = pd.read_csv(confirmed_cases_url, sep=",")

    data_end = "5/29/20"  # Take the data until yesterday
    data_begin = "5/1/20"
    num_days_to_predict = 14

    cases_obs = np.array(
        confirmed_cases.loc[
            confirmed_cases["Country/Region"] == "India", data_begin:data_end
        ]
    )[0]
    date_data_end = confirmed_cases.loc[
        confirmed_cases["Country/Region"] == "India", data_begin:data_end
    ].columns[-1]
    month, day, year = map(int, date_data_end.split("/"))
    date_data_end = datetime.date(year + 2000, month, day)
    date_today = date_data_end + datetime.timedelta(days=1)
    print(
        "Cases yesterday ({}): {} and day before yesterday: {}".format(
            date_data_end.isoformat(), *cases_obs[:-3:-1]
        )
    )
    num_days = len(cases_obs)

    np.random.seed(0)

    model = get_model()

    with model:
        trace = pm.sample(draws=500, target_accept=0.99, tune=2000)
    plot_SIR(trace, data_begin, confirmed_cases)
elif args.model == "nbeats":
    print("Refer to 'src/Meta_Learning.ipynb'")
elif args.model == "persistence":
    print_score()
elif args.model == "glm":
    get_glm_prediction()
