import datetime
import time

import pandas as pd
import pymc3 as pm
import seaborn as sns
import theano
import theano.tensor as tt

sns.set()
sns.set_context("paper")
# import matplotlib.pyplot as plt
import numpy as np

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

# ------------------------------------------------------------------------------ #
# model setup and training
# ------------------------------------------------------------------------------ #


def SIR_model(λ, μ, S_begin, I_begin, N):
    new_I_0 = tt.zeros_like(I_begin)

    def next_day(λ, S_t, I_t, _):
        new_I_t = λ / N * I_t * S_t
        S_t = S_t - new_I_t
        I_t = I_t + new_I_t - μ * I_t
        return S_t, I_t, new_I_t

    outputs, _ = theano.scan(
        fn=next_day, sequences=[λ], outputs_info=[S_begin, I_begin, new_I_0]
    )
    S_all, I_all, new_I_all = outputs
    return S_all, I_all, new_I_all


if __name__ == "__main__":

    with pm.Model() as model:
        # true cases at begin of loaded data but we do not know the real number

        I_begin = pm.Lognormal("I_begin", mu=np.log(cases_obs[0]), sigma=0.9)

        # fraction of people that are newly infected each day
        λ = pm.Lognormal("λ", mu=np.log(0.4), sigma=0.5)

        # fraction of people that recover each day, recovery rate mu
        μ = pm.Lognormal("μ", mu=np.log(1 / 8), sigma=0.2)

        # prior of the error of observed cases
        σ_obs = pm.HalfCauchy("σ_obs", beta=2)

        N_india = 136.7e7

        # -------------------------------------------------------------------------- #
        # training the model with loaded data
        # -------------------------------------------------------------------------- #

        S_begin = N_india - I_begin
        S_past, I_past, new_I_past = SIR_model(
            λ=λ * tt.ones(int(num_days) - 1),
            μ=μ,
            S_begin=S_begin,
            I_begin=I_begin,
            N=N_india,
        )
        #     new_infections_obs = np.diff(cases_obs)

        new_infections_obs = np.diff(cases_obs)

        # Approximates Poisson
        # calculate the likelihood of the model:
        # observed cases are distributed following studentT around the model

        obs = pm.StudentT(
            "obs",
            nu=4,
            mu=new_I_past,
            sigma=tt.switch(
                new_I_past ** 0.5 * σ_obs > 0, new_I_past ** 0.5 * σ_obs, 1e-6
            ),
            observed=new_infections_obs,
        )

        S_past = pm.Deterministic("S_past", S_past)
        I_past = pm.Deterministic("I_past", I_past)
        new_I_past = pm.Deterministic("new_I_past", new_I_past)

        # -------------------------------------------------------------------------- #
        # prediction, start with no changes in policy
        # -------------------------------------------------------------------------- #

        # delay in days between contracting the disease and being recorded
        delay = pm.Lognormal("delay", mu=np.log(8), sigma=0.1)

        S_begin = S_past[-1]
        I_begin = I_past[-1]
        forecast_no_change = SIR_model(
            λ=λ * tt.ones(num_days_to_predict),
            μ=μ,
            S_begin=S_begin,
            I_begin=I_begin,
            N=N_india,
        )
        S_no_change, I_no_change, new_I_no_change = forecast_no_change

        # saves the variables for later retrieval
        pm.Deterministic("S_no_change", S_no_change)
        pm.Deterministic("I_no_change", I_no_change)
        pm.Deterministic("new_I_no_change", new_I_no_change)

        # -------------------------------------------------------------------------- #
        # social distancing, m reduced by about 50 percent
        # -------------------------------------------------------------------------- #
        # For all following predictions:
        length_transient = 7  # days

        # λ is decreased by 50%
        reduc_factor_mild = 0.5
        days_offset = 0  # start the decrease in spreading rate after this

        time_arr = np.arange(num_days_to_predict)

        # change in m along time
        λ_correction = tt.clip(
            (time_arr - delay - days_offset + 1) / length_transient, 0, 1
        )
        λ_t_soc_dist = λ * (1 - λ_correction * reduc_factor_mild)

        S_begin = S_past[-1]
        I_begin = I_past[-1]
        forecast_soc_dist = SIR_model(
            λ=λ_t_soc_dist, μ=μ, S_begin=S_begin, I_begin=I_begin, N=N_india
        )
        S_soc_dist, I_soc_dist, new_I_soc_dist = forecast_soc_dist
        pm.Deterministic("S_soc_dist", S_soc_dist)
        pm.Deterministic("I_soc_dist", I_soc_dist)
        pm.Deterministic("new_I_soc_dist", new_I_soc_dist)

        # -------------------------------------------------------------------------- #
        # isolation, almost no new infections besides baseline after transient phase
        # -------------------------------------------------------------------------- #

        # λ is decreased by 90%
        reduc_factor_strong = 0.9
        days_offset = 0  # start the decrease in spreading rate after this

        # spreading of people who transmit although they are isolated
        time_arr = np.arange(num_days_to_predict)

        # change in λ along time
        λ_correction = tt.clip(
            (time_arr - delay - days_offset + 1) / length_transient, 0, 1
        )
        λ_t_isol = λ * (1 - λ_correction * reduc_factor_strong)

        S_begin = S_past[-1]
        I_begin = I_past[-1]
        forecast_isol = SIR_model(
            λ=λ_t_isol, μ=μ, S_begin=S_begin, I_begin=I_begin, N=N_india
        )
        S_isol, I_isol, new_I_isol = forecast_isol

        pm.Deterministic("S_isol", S_isol)
        pm.Deterministic("I_isol", I_isol)
        pm.Deterministic("new_I_isol", new_I_isol)

        # -------------------------------------------------------------------------- #
        # isolation 5 days later, almost no new infections besides baseline after transient phase
        # -------------------------------------------------------------------------- #

        # λ is decreased by 90%
        reduc_factor_strong = 0.9
        days_offset = 5  # start the decrease in spreading rate after this

        # spreading of people who transmit although they are isolated
        time_arr = np.arange(num_days_to_predict)

        # change in λ along time
        λ_correction = tt.clip(
            (time_arr - delay - days_offset + 1) / length_transient, 0, 1
        )
        λ_t_isol_later = λ * (1 - λ_correction * reduc_factor_strong)

        S_begin = S_past[-1]
        I_S_beginbegin = I_past[-1]
        forecast_isol_later = SIR_model(
            λ=λ_t_isol_later, μ=μ, S_begin=S_begin, I_begin=I_begin, N=N_india
        )
        S_isol_later, I_isol_later, new_I_isol_later = forecast_isol_later

        pm.Deterministic("S_isol_later", S_isol_later)
        pm.Deterministic("I_isol_later", I_isol_later)
        pm.Deterministic("new_I_isol_later", new_I_isol_later)

        # -------------------------------------------------------------------------- #
        # isolation 7 days earlyier, almost no new infections besides baseline after transient phase
        # -------------------------------------------------------------------------- #

        # λ is decreased by 90%
        reduc_factor_strong = 0.9
        days_offset = -5  # start the decrease in spreading rate after this

        # spreading of people who transmit although they are isolated
        time_arr = np.arange(num_days_to_predict)

        # change in λ along time
        λ_correction = tt.clip(
            (time_arr - delay - days_offset + 1) / length_transient, 0, 1
        )
        λ_t_isol_earlyier = λ * (1 - λ_correction * reduc_factor_strong)

        S_begin = S_past[-1]
        I_S_begin = I_past[-1]
        forecast_isol_earlyier = SIR_model(
            λ=λ_t_isol_earlyier, μ=μ, S_begin=S_begin, I_begin=I_begin, N=N_india
        )
        S_isol_earlyier, I_isol_earlyier, new_I_isol_earlyier = forecast_isol_earlyier

        pm.Deterministic("S_isol_earlyier", S_isol_earlyier)
        pm.Deterministic("I_isol_earlyier", I_isol_earlyier)
        pm.Deterministic("new_I_isol_earlyier", new_I_isol_earlyier)

        # -------------------------------------------------------------------------- #
        # run model, pm trains and predicts when calling this
        # -------------------------------------------------------------------------- #

        time_beg = time.time()
        #     prior_new = pm.sample_prior_predictive(samples=5000, random_seed=24)
        trace = pm.sample(draws=500, target_accept=0.99, tune=2000)
        print("Model run in {:.2f} s".format(time.time() - time_beg))
