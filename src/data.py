import random

import numpy as np
import pandas as pd
import tensorflow as tf

from src.utils import smoothen_data


class Data(object):
    def __init__(self, dataframe: pd.DataFrame, previous_days: int, forecast_days: int):
        self.df = dataframe
        self.previous_days = previous_days
        self.forecast_days = forecast_days
        self.scaling_params = None

    def preprocess(self):
        if "Date" in self.df.columns:
            df_no_date = self.df.drop("Date", axis=1)
            df_no_date = df_no_date[df_no_date > 0]
            self.df[df_no_date.columns] = df_no_date
        else:
            self.df = self.df[self.df > 0]

        self.df = self.df.dropna()

    def get_features(self, features: list):
        return self.df[features].copy()

    def sliding_window(self, features: list, seq_len: int, j=0):
        df = self.get_features(features)
        xs = []
        ys = []
        for i in range(0, len(df) - seq_len - 1 - j, seq_len + 1 + j):
            x = df[i : (i + seq_len)].to_numpy()
            y = (
                df[(i + seq_len) : i + seq_len + 1 + j]["new cases"]
                .to_numpy()
                .flatten()
            )  # [0]
            xs.append(x)
            ys.append(y)
        return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)

    def smoothen_df(self, cols=None):
        if cols is None:
            cols = [i for i in self.df.columns if self.df[i].dtype == float]

        for i in cols:
            self.df[i] = smoothen_data(self.df[i])

    def region_wise_normalized_df(self):
        """
        Normalizes Data Region (District) Wise and returns a dictionary with scaler params (min, max for each region)
        """
        # df = df.drop(['State', *colstoplot[:-1]], axis=1)

        df = self.smoothen_df(self.df)
        params = {}
        for region in df["District"].unique():
            cases = df[df["District"] == region]["new cases"]
            df.loc[df["District"] == region, "new cases"] = (cases - cases.min()) / (
                cases.max() - cases.min()
            )
            params[region] = [cases.min(), cases.max()]

        self.scaling_params = params
        return df, params

    def create_metaLarrays(self, days_before=None, days_after=None):
        xtrain, ytrain = [], []
        params = self.scaling_params
        for district in self.df["District"].unique():
            district_df = self.df[self.df["District"] == district].copy()
            if district_df.shape[0] < 200:  #
                del params[district]
                continue
            district_df = district_df.drop("District", axis=1)

            district_df["Date"] = pd.to_datetime(district_df["Date"])
            district_df["Date"] = (
                district_df["Date"].dt.year * 30 * 12
                + district_df["Date"].dt.month * 30
                + district_df["Date"].dt.day
            )
            district_df["Date"] -= district_df["Date"].min()

            if days_after is None and days_before is None:
                X, y = self.sliding_window(
                    district_df, self.previous_days, self.forecast_days - 1
                )
            elif days_after is None and days_before is not None:
                X, y = self.sliding_window(
                    district_df[district_df["Date"] < days_before],
                    self.previous_days,
                    self.forecast_days - 1,
                )
            elif days_after is not None and days_before is None:
                X, y = self.sliding_window(
                    district_df[district_df["Date"] >= days_after],
                    self.previous_days,
                    self.forecast_days - 1,
                )
            if 0 in X.shape or 0 in y.shape:  # or 0 in xt.shape or 0 in yt.shape:
                del params[district]
                continue

            xtrain.append(X)
            ytrain.append(y)

        return np.asarray(xtrain), np.asarray(ytrain)

    def get_mini_dataset(batchSize, iters, num_regions, datasets, test=False):
        # Shots is assumed to be all
        (
            xtrain,
            ytrain,
        ) = datasets  # Get these datasets from the `create_metaLarrays` function

        regions = random.choices(list(range(xtrain.shape[0])), k=num_regions)

        if test:
            xtests, ytests, xtrains, ytrains = [], [], [], []
            xtrain, ytrain = xtrain[[regions]], ytrain[[regions]]
            for i in range(num_regions):
                test_idx = random.choice(list(range(xtrain[i].shape[0])))
                xtests.append(xtrain[i][test_idx, :, :])
                ytests.append(ytrain[i][test_idx, :])
                xtrains.append(
                    xtrain[i][np.arange(0, xtrain[i].shape[0]) != test_idx, :, :]
                )
                ytrains.append(
                    ytrain[i][np.arange(0, xtrain[i].shape[0]) != test_idx, :]
                )

            xtrain, ytrain, xtest, ytest = (
                np.vstack(xtrains),
                np.vstack(ytrains),
                np.hstack(xtests),
                np.stack(ytests),
            )
        else:
            xtrain, ytrain = np.vstack(xtrain[[regions]]), np.vstack(ytrain[[regions]])

        dataset = tf.data.Dataset.from_tensor_slices(
            (xtrain.astype(np.float32), ytrain.astype(np.float32))
        )
        dataset = dataset.shuffle(100).batch(batchSize).repeat(iters)
        if test:
            return dataset, tf.expand_dims(xtest.T, -1), ytest, regions
        return dataset, regions

    def create_dataset(self, traindays, testdays):
        xtrain, ytrain, xtest, ytest = [], [], [], []
        params = self.scaling_params
        for district in self.df["District"].unique():
            district_df = self.df[self.df["District"] == district].copy()
            if district_df.shape[0] < 200:
                del params[district]
                continue
            district_df = district_df.drop("District", axis=1)

            district_df["Date"] = pd.to_datetime(district_df["Date"])
            district_df["Date"] = (
                district_df["Date"].dt.year * 30 * 12
                + district_df["Date"].dt.month * 30
                + district_df["Date"].dt.day
            )
            district_df["Date"] -= district_df["Date"].min()

            xtr, ytr = self.sliding_window(
                district_df[district_df["Date"] == traindays],
                self.previous_days,
                self.forecast_days - 1,
            )
            xte, yte = self.sliding_window(
                district_df[district_df["Date"] == testdays],
                self.previous_days,
                self.forecast_days - 1,
            )
            if 0 in xtr.shape or 0 in ytr.shape or 0 in xte.shape or 0 in yte.shape:
                del params[district]
                continue

            xtrain.append(xtr)
            ytrain.append(ytr)
            xtest.append(xte)
            ytest.append(yte)

        return (
            np.asarray(xtrain),
            np.asarray(ytrain),
            np.asarray(xtest),
            np.asarray(ytest),
        )
