"""Compares every model under random, chronological, and walk-forward validation.

Each model/validation pair adds a row to results/summary.csv, its per-day scores to
results/daily.csv, and its predictions to data/predictions/. Pairs already in
summary.csv are skipped, so an interrupted comparison picks up where it stopped.
Build the Excel table afterwards with Results_Workbook.py.

    python Model_Comparison.py
    python Model_Comparison.py --models "Linear Regression" XGBoost --methods Walk-forward
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from Data_Frame_10y import FEATURES, build_dataset, wikipedia_tickers
from Validation import chronological_split, daily_scores, holdout, random_split, refit_count, score, walk_forward

METHODS = ['Random', 'Chronological', 'Walk-forward']


class TabularMLP:
    """Neural_Network.ipynb's fastai tabular learner behind fit/predict."""

    def __init__(self, layers=(225, 105, 75, 32, 16, 1), epochs=5, bs=1024, seed=72):
        self.layers, self.epochs, self.bs, self.seed = layers, epochs, bs, seed

    def fit(self, X, y):
        from fastai.tabular.all import Categorify, FillMissing, Normalize, TabularDataLoaders, set_seed, tabular_learner

        set_seed(self.seed)
        data = X.assign(**{'future return': y.to_numpy()})
        # fastai needs a validation set; it is only used for its progress metrics here,
        # so a sample of training rows stands in and all training rows stay in training
        valid = np.random.default_rng(self.seed).choice(len(data), size=min(5000, len(data)), replace=False)
        self.dls = TabularDataLoaders.from_df(
            data, y_names='future return', cat_names=['id'], cont_names=[c for c in X.columns if c != 'id'],
            procs=[Categorify, FillMissing, Normalize], splits=(list(range(len(data))), list(valid)),
            bs=self.bs, num_workers=0)
        self.learn = tabular_learner(self.dls, layers=list(self.layers))
        with self.learn.no_bar(), self.learn.no_logging():
            self.learn.fit_one_cycle(self.epochs)
        return self

    def predict(self, X):
        with self.learn.no_bar():
            preds, _ = self.learn.get_preds(dl=self.dls.test_dl(X))
        return preds.numpy().ravel()


class StockLSTM(nn.Module):
    def __init__(self, n_cont_features, n_tickers, emb_dim=16, hidden_size=64,
                 num_layers=1, dropout=0.3, emb_dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(n_tickers, emb_dim)
        self.emb_drop = nn.Dropout(emb_dropout)
        self.rnn = nn.LSTM(
            input_size=n_cont_features + emb_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, ids):
        e = self.emb_drop(self.emb(ids))
        e = e.unsqueeze(1).expand(-1, x.size(1), -1)
        out, _ = self.rnn(torch.cat([x, e], dim=-1))
        return self.fc(self.drop(out[:, -1, :])).squeeze(1)


class SequenceLSTM:
    """RNN.ipynb's StockLSTM behind fit/predict. Each row is predicted from its ticker's
    last `window` rows ending on that row's date, so the target is the next day's return."""

    def __init__(self, df, features, window=10, epochs=5, bs=256, lr=1e-3, seed=72):
        df = df.sort_values(['ticker', 'date'])
        self.cont = [f for f in features if f != 'id']
        self.values = df[self.cont].to_numpy(np.float32)
        self.ids = df['id'].to_numpy(np.int64)
        self.n_tickers = int(self.ids.max()) + 1
        self.index = df.index
        self.epochs, self.bs, self.lr, self.seed = epochs, bs, lr, seed
        # row positions of each window, repeating a ticker's first row when it has less history
        pos = np.arange(len(df))
        first = pos - df.groupby('ticker').cumcount().to_numpy()
        self.windows = np.maximum(pos[:, None] + np.arange(1 - window, 1), first[:, None])

    def batch(self, rows):
        x = (self.values[self.windows[rows]] - self.mean) / self.std
        return torch.from_numpy(x), torch.from_numpy(self.ids[rows])

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        rows = self.index.get_indexer(X.index)
        # scale with training rows only
        self.mean = self.values[rows].mean(0)
        self.std = self.values[rows].std(0) + 1e-8
        target = torch.tensor(y.to_numpy(np.float32))
        self.model = StockLSTM(len(self.cont), self.n_tickers)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        self.model.train()
        for epoch in range(self.epochs):
            order = rng.permutation(len(rows))
            for start in range(0, len(order), self.bs):
                b = order[start:start + self.bs]
                xb, ib = self.batch(rows[b])
                optimizer.zero_grad()
                loss = criterion(self.model(xb, ib), target[b])
                loss.backward()
                optimizer.step()
        return self

    def predict(self, X):
        rows = self.index.get_indexer(X.index)
        self.model.eval()
        with torch.no_grad():
            return np.concatenate([self.model(*self.batch(rows[s:s + 4096])).numpy()
                                   for s in range(0, len(rows), 4096)])


def model_specs(df):
    """name -> (factory, walk-forward refit interval in trading days, settings for the workbook)."""
    return {
        'Zero forecast': (lambda: DummyRegressor(strategy='constant', constant=0.0), 1,
                          'Always predicts a 0% return'),
        'Historical mean': (lambda: DummyRegressor(strategy='mean'), 1,
                            'Predicts the mean training-set return'),
        'Linear Regression': (LinearRegression, 1, 'sklearn defaults'),
        'Decision Tree': (lambda: DecisionTreeRegressor(max_depth=5, random_state=72), 1,
                          'max_depth=5'),
        'Random Forest': (lambda: RandomForestRegressor(n_estimators=100, min_samples_leaf=200, max_features=0.33,
                                                        max_samples=0.5, n_jobs=-1, random_state=72), 5,
                          'n_estimators=100, min_samples_leaf=200, max_features=0.33, max_samples=0.5'),
        'XGBoost': (lambda: XGBRegressor(n_estimators=25, max_depth=10, learning_rate=0.05,
                                         n_jobs=-1, random_state=72), 1,
                    'n_estimators=25, max_depth=10, learning_rate=0.05'),
        'Neural Network': (lambda: TabularMLP(), 21,
                           'fastai tabular_learner, layers=[225,105,75,32,16,1], 5 epochs one-cycle, batch 1024, id embedded'),
        'LSTM': (lambda: SequenceLSTM(df, FEATURES), 21,
                 'StockLSTM, 10-day window, hidden 64, id embedding 16, dropout 0.3, Adam 1e-3, 5 epochs, batch 256'),
    }


def load_data(path, train_pct):
    if os.path.exists(path):
        return pd.read_parquet(path)
    df, *_ = build_dataset(train_pct=train_pct, tickers=wikipedia_tickers())
    df.to_parquet(path)
    return df


def append_csv(frame, path):
    frame.to_csv(path, mode='a', header=not os.path.exists(path), index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/dataset_3y.parquet')
    parser.add_argument('--out', default='results')
    parser.add_argument('--train-pct', type=float, default=0.8)
    parser.add_argument('--models', nargs='*')
    parser.add_argument('--methods', nargs='*', default=METHODS)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs('data/predictions', exist_ok=True)
    summary_path = os.path.join(args.out, 'summary.csv')
    done = set()
    if os.path.exists(summary_path):
        done = set(pd.read_csv(summary_path)[['model', 'validation']].itertuples(index=False, name=None))

    df = load_data(args.data, args.train_pct)
    specs = model_specs(df)
    for name in args.models or specs:
        make_model, refit_every, settings = specs[name]
        for method in args.methods:
            if (name, method) in done:
                continue
            start = time.time()
            if method == 'Random':
                results = holdout(df, make_model, FEATURES, split=random_split, train_pct=args.train_pct)
                every, refits = None, 1
            elif method == 'Chronological':
                results = holdout(df, make_model, FEATURES, split=chronological_split, train_pct=args.train_pct)
                every, refits = None, 1
            else:
                results = walk_forward(df, make_model, FEATURES, train_pct=args.train_pct, refit_every=refit_every)
                every, refits = refit_every, refit_count(df, args.train_pct, refit_every)
            seconds = time.time() - start

            row = {'model': name, 'validation': method, 'refit every (days)': every, 'refits': refits,
                   'train pct': args.train_pct, **score(results), 'seconds': round(seconds, 1), 'settings': settings}
            append_csv(pd.DataFrame([row]), summary_path)
            if method != 'Random':
                daily = daily_scores(results).reset_index().assign(model=name, validation=method)
                append_csv(daily, os.path.join(args.out, 'daily.csv'))
            results[['date', 'ticker', 'future return', 'prediction']].to_parquet(
                f"data/predictions/{name.replace(' ', '_')}_{method}.parquet")
            print(f"{name:18s} {method:13s} MSE {row['MSE']:.4f}  acc {row['directional accuracy']:.4f}  "
                  f"IC {row['IC mean']:.4f}  {seconds:.0f}s", flush=True)


if __name__ == '__main__':
    main()
