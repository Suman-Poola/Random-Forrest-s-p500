"""Models for the improved experiment (Model_Comparison.py --experiment improved).

They train on Data_Frame_v2's 'target' and features through sklearn-style fit(X, y) and
predict(X). Models that are tuned or stopped early hold out the last VALIDATION_DAYS
trading days of each training window for it, so the test days never choose a setting.
Models that need each row's date get the data frame's 'date' column, matched by index.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRanker, XGBRegressor

VALIDATION_DAYS = 60


def recent_days(dates, days=VALIDATION_DAYS):
    """Mask of the rows dated within the last `days` trading days of `dates`."""
    cutoff = np.sort(dates.unique())[-days]
    return (dates >= cutoff).to_numpy()


def mean_daily_ic(dates, y, p):
    frame = pd.DataFrame({'date': np.asarray(dates), 'y': np.asarray(y), 'p': np.asarray(p)})
    return frame.groupby('date')[['y', 'p']].apply(lambda g: g['p'].rank().corr(g['y'].rank())).mean()


class Rule:
    """Ranks stocks by one feature, reversed when sign is -1. Nothing is fitted."""

    def __init__(self, feature, sign=1):
        self.feature, self.sign = feature, sign

    def fit(self, X, y):
        return self

    def predict(self, X):
        return self.sign * X[self.feature].to_numpy()


class Tuned:
    """Fits make(setting) for each setting on the training rows before the validation days,
    keeps the one with the highest mean daily rank IC on those days, and refits it on every row."""

    def __init__(self, make, settings, dates):
        self.make, self.settings, self.dates = make, settings, dates

    def fit(self, X, y):
        dates = self.dates.loc[X.index]
        val = recent_days(dates)
        ics = [mean_daily_ic(dates[val], y[val], self.make(s).fit(X[~val], y[~val]).predict(X[val]))
               for s in self.settings]
        self.best = self.settings[int(np.argmax(np.nan_to_num(ics, nan=-np.inf)))]
        self.model = self.make(self.best).fit(X, y)
        return self

    def predict(self, X):
        return np.ravel(self.model.predict(X))


class AboveMedianClassifier:
    """Logistic regression on whether the target is above 0, i.e. above the day's median.
    Predicts the probability minus 0.5."""

    def __init__(self, C):
        self.model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=300))

    def fit(self, X, y):
        self.model.fit(X, y > 0)
        return self

    def predict(self, X):
        return self.model.predict_proba(X)[:, 1] - 0.5


class EarlyStoppedXGB:
    """XGBoost regression whose number of trees is chosen by early stopping on the
    validation days; that many trees are then refit on every training row."""

    def __init__(self, dates, max_rounds=2000, patience=100, **params):
        self.dates, self.max_rounds, self.patience, self.params = dates, max_rounds, patience, params

    def fit(self, X, y):
        val = recent_days(self.dates.loc[X.index])
        probe = XGBRegressor(n_estimators=self.max_rounds, early_stopping_rounds=self.patience, **self.params)
        probe.fit(X[~val], y[~val], eval_set=[(X[val], y[val])], verbose=False)
        self.rounds = probe.best_iteration + 1
        self.model = XGBRegressor(n_estimators=self.rounds, **self.params).fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


class DailyRanker:
    """XGBoost learning to rank: each trading day is one query and each row's label is its
    target decile that day, so the trees learn which stocks beat the others. The number of
    trees is chosen by early stopping on the validation days (NDCG of the top 50), then refit."""

    def __init__(self, dates, max_rounds=1000, patience=50, **params):
        self.dates, self.max_rounds, self.patience, self.params = dates, max_rounds, patience, params

    def fit(self, X, y):
        dates = self.dates.loc[X.index]
        order = np.argsort(dates.to_numpy(), kind='stable')
        X, y, dates = X.iloc[order], y.iloc[order], dates.iloc[order]
        labels = (y.groupby(dates.to_numpy()).rank(pct=True) * 10).clip(upper=9.999).astype(int).to_numpy()
        qid = pd.factorize(dates)[0]
        val = recent_days(dates)
        probe = XGBRanker(n_estimators=self.max_rounds, early_stopping_rounds=self.patience,
                          eval_metric='ndcg@50', **self.params)
        probe.fit(X[~val], labels[~val], qid=qid[~val], eval_set=[(X[val], labels[val])], eval_qid=[qid[val]],
                  verbose=False)
        self.rounds = probe.best_iteration + 1
        self.model = XGBRanker(n_estimators=self.rounds, **self.params).fit(X, labels, qid=qid)
        return self

    def predict(self, X):
        return self.model.predict(X)


def train_early_stopped(net, forward, target, train_rows, val_rows, seed, lr, weight_decay, bs, max_epochs,
                        patience, samples_per_epoch=None):
    """Adam on squared error over `train_rows`, reshuffled each epoch (or a fresh sample of
    `samples_per_epoch` of them). Stops after `patience` epochs without a lower validation
    loss and returns the network with its best weights. forward(net, rows) gives predictions."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    best_loss, best_state, waited = np.inf, None, 0
    for epoch in range(max_epochs):
        net.train()
        rows = rng.permutation(train_rows)[:samples_per_epoch]
        for start in range(0, len(rows), bs):
            batch = rows[start:start + bs]
            if len(batch) < 2:  # batch norm needs two rows
                continue
            optimizer.zero_grad()
            loss = ((forward(net, batch) - target[batch]) ** 2).mean()
            loss.backward()
            optimizer.step()
        net.eval()
        with torch.no_grad():
            val_loss = sum(((forward(net, val_rows[s:s + 8192]) - target[val_rows[s:s + 8192]]) ** 2).sum().item()
                           for s in range(0, len(val_rows), 8192)) / len(val_rows)
        if val_loss < best_loss:
            best_loss, best_state, waited = val_loss, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            waited += 1
            if waited >= patience:
                break
    net.load_state_dict(best_state)
    return net.eval()


class MLP:
    """Feed-forward network: inputs z-scored with the training rows' mean and std, hidden
    layers with batch norm, ReLU and dropout, Adam with weight decay, and early stopping on
    the validation days. Predictions average `seeds` networks trained from different seeds."""

    def __init__(self, dates, layers=(32, 16, 8), dropout=0.1, weight_decay=1e-4, lr=1e-3, bs=4096,
                 max_epochs=20, patience=3, seeds=5):
        self.dates, self.layers, self.dropout = dates, layers, dropout
        self.train_args = dict(lr=lr, weight_decay=weight_decay, bs=bs, max_epochs=max_epochs, patience=patience)
        self.seeds = seeds

    def network(self, n_inputs):
        blocks, width = [], n_inputs
        for size in self.layers:
            blocks += [nn.Linear(width, size), nn.BatchNorm1d(size), nn.ReLU(), nn.Dropout(self.dropout)]
            width = size
        return nn.Sequential(*blocks, nn.Linear(width, 1))

    def inputs(self, X):
        return torch.from_numpy((X.to_numpy(np.float32) - self.mean) / self.std)

    def fit(self, X, y):
        val = recent_days(self.dates.loc[X.index])
        train = X.to_numpy(np.float32)[~val]
        self.mean, self.std = train.mean(0), train.std(0) + 1e-6
        x, target = self.inputs(X), torch.from_numpy(y.to_numpy(np.float32))
        forward = lambda net, rows: net(x[torch.from_numpy(rows)]).squeeze(1)
        self.nets = [train_early_stopped(self.network(x.shape[1]), forward, target, np.flatnonzero(~val),
                                         np.flatnonzero(val), seed, **self.train_args)
                     for seed in range(self.seeds)]
        return self

    def predict(self, X):
        x = self.inputs(X)
        with torch.no_grad():
            return np.mean([np.concatenate([net(x[s:s + 8192]).squeeze(1).numpy() for s in range(0, len(x), 8192)])
                            for net in self.nets], axis=0)


class LSTMNet(nn.Module):
    def __init__(self, n_sequence, n_static, hidden, dropout):
        super().__init__()
        self.rnn = nn.LSTM(n_sequence, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden + n_static, 1)

    def forward(self, sequence, static):
        out, _ = self.rnn(sequence)
        return self.fc(torch.cat([self.drop(out[:, -1]), static], dim=1)).squeeze(1)


class SequenceLSTM:
    """LSTM over each ticker's last `window` trading days of `sequence_features` (already
    ranked across tickers, so no further scaling), with `static_features` from the row's own
    day added at the output layer. The window repeats a ticker's first row when it has less
    history. Each epoch trains on a fresh sample of `samples_per_epoch` training rows, with
    early stopping on the validation days; predictions average `seeds` networks."""

    def __init__(self, df, sequence_features, static_features, window=60, hidden=32, dropout=0.1, lr=1e-3,
                 weight_decay=0.0, bs=512, samples_per_epoch=200_000, max_epochs=10, patience=2, seeds=3):
        df = df.sort_values(['ticker', 'date'])
        self.sequence = torch.from_numpy(df[sequence_features].to_numpy(np.float32))
        self.static = torch.from_numpy(df[static_features].to_numpy(np.float32))
        self.dates = df['date']
        self.position = pd.Series(np.arange(len(df)), index=df.index)
        self.first = np.arange(len(df)) - df.groupby('ticker').cumcount().to_numpy()
        self.offsets = np.arange(1 - window, 1)
        self.shape = dict(n_sequence=len(sequence_features), n_static=len(static_features), hidden=hidden,
                          dropout=dropout)
        self.train_args = dict(lr=lr, weight_decay=weight_decay, bs=bs, max_epochs=max_epochs, patience=patience,
                               samples_per_epoch=samples_per_epoch)
        self.seeds = seeds

    def forward(self, net, rows):
        window = torch.from_numpy(np.maximum(rows[:, None] + self.offsets, self.first[rows][:, None]))
        return net(self.sequence[window], self.static[torch.from_numpy(rows)])

    def fit(self, X, y):
        rows = self.position.loc[X.index].to_numpy()
        val = recent_days(self.dates.loc[X.index])
        target = torch.zeros(len(self.position))
        target[torch.from_numpy(rows)] = torch.from_numpy(y.to_numpy(np.float32))
        self.nets = [train_early_stopped(LSTMNet(**self.shape), self.forward, target, rows[~val], rows[val], seed,
                                         **self.train_args)
                     for seed in range(self.seeds)]
        return self

    def predict(self, X):
        rows = self.position.loc[X.index].to_numpy()
        with torch.no_grad():
            return np.mean([np.concatenate([self.forward(net, rows[s:s + 4096]).numpy()
                                            for s in range(0, len(rows), 4096)])
                            for net in self.nets], axis=0)
