import argparse
import os
import time
from collections import namedtuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

import Data_Frame_v2
import Improved_Models as improved
from Data_Frame_10y import FEATURES, build_dataset, wikipedia_tickers
from Validation import chronological_split, daily_scores, holdout, random_split, refit_count, score, walk_forward

METHODS = ['Random', 'Chronological', 'Walk-forward']

Spec = namedtuple('Spec', ['make', 'refit', 'settings', 'family', 'normalization', 'ticker', 'inputs',
                           'units', 'original', 'members'], defaults=['return', None, None])


class TabularMLP:
    # from Neural_Network.ipynb
    def __init__(self, layers=(225, 105, 75, 32, 16, 1), epochs=5, bs=1024, seed=72):
        self.layers, self.epochs, self.bs, self.seed = layers, epochs, bs, seed

    def fit(self, X, y):
        from fastai.tabular.all import Categorify, FillMissing, Normalize, TabularDataLoaders, set_seed, tabular_learner

        set_seed(self.seed)
        data = X.assign(**{'future return': y.to_numpy()})
        # fastai needs a validation set
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
    # from RNN.ipynb
    def __init__(self, df, features, window=10, epochs=5, bs=256, lr=1e-3, seed=72):
        df = df.sort_values(['ticker', 'date'])
        self.cont = [f for f in features if f != 'id']
        self.values = df[self.cont].to_numpy(np.float32)
        self.ids = df['id'].to_numpy(np.int64)
        self.n_tickers = int(self.ids.max()) + 1
        self.index = df.index
        self.epochs, self.bs, self.lr, self.seed = epochs, bs, lr, seed
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


def original_specs(df):
    return {
        'Zero forecast': Spec(lambda: DummyRegressor(strategy='constant', constant=0.0), 1,
                              'Always predicts a 0% return', 'Constant baseline', 'n/a (no features)', 'Not used',
                              'No features'),
        'Historical mean': Spec(lambda: DummyRegressor(strategy='mean'), 1,
                                'Predicts the mean training-set return', 'Constant baseline', 'n/a (no features)',
                                'Not used', 'No features'),
        'Linear Regression': Spec(LinearRegression, 1, 'sklearn defaults', 'Linear', 'Raw features', 'Numeric code',
                                  'Day t'),
        'Decision Tree': Spec(lambda: DecisionTreeRegressor(max_depth=5, random_state=72), 1,
                              'max_depth=5', 'Single tree', 'Raw features', 'Numeric code', 'Day t'),
        'Random Forest': Spec(lambda: RandomForestRegressor(n_estimators=100, min_samples_leaf=200, max_features=0.33,
                                                            max_samples=0.5, n_jobs=-1, random_state=72), 5,
                              'n_estimators=100, min_samples_leaf=200, max_features=0.33, max_samples=0.5',
                              'Tree ensemble, bagged', 'Raw features', 'Numeric code', 'Day t'),
        'XGBoost': Spec(lambda: XGBRegressor(n_estimators=25, max_depth=10, learning_rate=0.05,
                                             n_jobs=-1, random_state=72), 1,
                        'n_estimators=25, max_depth=10, learning_rate=0.05',
                        'Tree ensemble, gradient boosted', 'Raw features', 'Numeric code', 'Day t'),
        'Neural Network': Spec(lambda: TabularMLP(), 21,
                               'fastai tabular_learner, layers=[225,105,75,32,16,1], 5 epochs one-cycle, batch 1024, '
                               'id embedded',
                               'Feed-forward neural network', 'Z-score, training rows (fastai Normalize)',
                               'Learned embedding', 'Day t'),
        'LSTM': Spec(lambda: SequenceLSTM(df, FEATURES), 21,
                     'StockLSTM, 10-day window, hidden 64, id embedding 16, dropout 0.3, Adam 1e-3, 5 epochs, batch 256',
                     'Recurrent neural network (LSTM)', 'Z-score, training rows', 'Learned embedding (16-d)',
                     'Days t-9 to t'),
    }


XGB_PARAMS = dict(learning_rate=0.03, max_depth=3, subsample=0.7, colsample_bytree=0.7, min_child_weight=500,
                  reg_lambda=10, tree_method='hist', n_jobs=-1, random_state=72)
RANKER_PARAMS = dict(objective='rank:pairwise', lambdarank_pair_method='mean', lambdarank_num_pair_per_sample=4,
                     learning_rate=0.05, max_depth=3, subsample=0.7, colsample_bytree=0.7, tree_method='hist',
                     n_jobs=-1, random_state=72)


def improved_specs(df):
    dates = df['date']
    ranked = 'Ranked across tickers each day'
    scaled = ranked + ', then z-scored on training rows'
    sector = 'Sector one-hot'
    learners = ['Ridge Regression', 'Logistic Regression', 'Decision Tree', 'Random Forest', 'XGBoost',
                'XGBoost Ranker', 'Neural Network', 'LSTM']
    return {
        'Zero forecast': Spec(lambda: DummyRegressor(strategy='constant', constant=0.0), 21,
                              "Always predicts that every stock matches the day's median", 'Constant baseline',
                              'n/a (no features)', 'Not used', 'No features', original='Zero forecast'),
        '1-day reversal': Spec(lambda: improved.Rule('return', -1), 21, "Ranks stocks by minus the day's return",
                               'Rule baseline', ranked, 'Not used', 'Day t', units='score'),
        '5-day reversal': Spec(lambda: improved.Rule('Lag 5'), 21, 'Ranks stocks by minus the 5-day return',
                               'Rule baseline', ranked, 'Not used', 'Day t', units='score'),
        'Ridge Regression': Spec(
            lambda: improved.Tuned(lambda a: make_pipeline(StandardScaler(), Ridge(alpha=a)), [1e3, 1e5, 1e7], dates),
            21, 'alpha in {1e3, 1e5, 1e7}, picked by validation IC', 'Linear, L2-penalized', scaled, sector, 'Day t',
            original='Linear Regression'),
        'Logistic Regression': Spec(
            lambda: improved.Tuned(improved.AboveMedianClassifier, [1e-4, 1e-3, 1e-2], dates),
            21, "Above or below the day's median; C in {1e-4, 1e-3, 1e-2}, picked by validation IC",
            'Linear classifier', scaled, sector, 'Day t', units='score', original='Linear Regression'),
        'Decision Tree': Spec(
            lambda: improved.Tuned(lambda leaf: DecisionTreeRegressor(max_depth=6, min_samples_leaf=leaf,
                                                                      random_state=72), [1000, 5000, 20000], dates),
            63, 'max_depth=6, min_samples_leaf in {1000, 5000, 20000}, picked by validation IC', 'Single tree',
            ranked, sector, 'Day t', original='Decision Tree'),
        'Random Forest': Spec(
            lambda: RandomForestRegressor(n_estimators=300, max_depth=6, min_samples_leaf=2000, max_features=0.33,
                                          max_samples=0.2, n_jobs=-1, random_state=72),
            21, 'n_estimators=300, max_depth=6, min_samples_leaf=2000, max_features=0.33, max_samples=0.2',
            'Tree ensemble, bagged', ranked, sector, 'Day t', original='Random Forest'),
        'XGBoost': Spec(
            lambda: improved.EarlyStoppedXGB(dates, **XGB_PARAMS),
            21, 'max_depth=3, learning_rate=0.03, up to 2000 trees by early stopping (patience 100), subsample=0.7, '
                'colsample_bytree=0.7, min_child_weight=500, reg_lambda=10',
            'Tree ensemble, gradient boosted', ranked, sector, 'Day t', original='XGBoost'),
        'XGBoost Ranker': Spec(
            lambda: improved.DailyRanker(dates, **RANKER_PARAMS),
            21, 'rank:pairwise on daily target deciles, max_depth=3, learning_rate=0.05, up to 1000 trees by early '
                'stopping on NDCG@50 (patience 50), subsample=0.7, colsample_bytree=0.7',
            'Tree ensemble, learning to rank', ranked, sector, 'Day t', units='score'),
        'Neural Network': Spec(
            lambda: improved.MLP(dates),
            63, 'PyTorch, layers 32-16-8 with batch norm, dropout 0.1, Adam 1e-3, weight decay 1e-4, batch 4096, up to '
                '20 epochs by early stopping (patience 3), average of 5 seeds',
            'Feed-forward neural network', scaled, sector, 'Day t', original='Neural Network'),
        'LSTM': Spec(
            lambda: improved.SequenceLSTM(df, Data_Frame_v2.SEQUENCE_FEATURES, Data_Frame_v2.SECTOR_FEATURES),
            63, '60-day window of 6 daily features, hidden 32, dropout 0.1, Adam 1e-3, batch 512, 200,000 sampled '
                'rows per epoch, up to 10 epochs by early stopping (patience 2), average of 3 seeds',
            'Recurrent neural network (LSTM)', ranked, sector, 'Days t-59 to t', original='LSTM'),
        'Ensemble': Spec(None, None, 'Equal-weight average of the daily ranks of ' + ', '.join(learners), 'Ensemble',
                         "n/a (averages the members' ranks)", 'n/a', 'n/a', units='score', members=learners),
    }


def build_improved():
    tickers = pd.read_parquet(EXPERIMENTS['original']['data'], columns=['ticker'])['ticker'].unique()
    return Data_Frame_v2.build_dataset(tickers=sorted(tickers))


EXPERIMENTS = {
    'original': dict(data='data/dataset_3y.parquet', out='results', predictions='data/predictions',
                     features=FEATURES, fit_target='future return', score_target='future return',
                     target='Next-day return', specs=original_specs,
                     build=lambda: build_dataset(months=36, tickers=wikipedia_tickers())[0]),  # 3 years
    'improved': dict(data='data/dataset_v2.parquet', out='results/improved', predictions='data/predictions_v2',
                     features=Data_Frame_v2.FEATURES, fit_target='target', score_target='excess return',
                     target="Next-day return minus the day's median", specs=improved_specs, build=build_improved),
}


def load_data(path, build):
    if os.path.exists(path):
        return pd.read_parquet(path)
    df = build()
    df.to_parquet(path)
    return df


def prediction_path(folder, model, method):
    return f"{folder}/{model.replace(' ', '_')}_{method}.parquet"


def append_csv(frame, path):
    frame.to_csv(path, mode='a', header=not os.path.exists(path), index=False)


def ensemble(members, method, folder, target):
    member_predictions = {m: pd.read_parquet(prediction_path(folder, m, method)).set_index(['date', 'ticker'])
                          for m in members}
    ranks = pd.concat({m: p['prediction'].groupby(level='date').rank(pct=True)
                       for m, p in member_predictions.items()}, axis=1)
    first = member_predictions[members[0]]
    return first[[target]].assign(prediction=2 * ranks.mean(axis=1) - 1).reset_index()


def summary_row(name, method, spec, results, experiment, train_pct, every, refits, seconds):
    row = {'model': name, 'validation': method, 'refit every (days)': every, 'refits': refits, 'train pct': train_pct,
           **score(results, experiment['score_target']), 'seconds': round(seconds, 1), 'settings': spec.settings,
           'family': spec.family, 'normalization': spec.normalization, 'ticker input': spec.ticker,
           'inputs': spec.inputs, 'units': spec.units, 'original model': spec.original, 'target': experiment['target']}
    if spec.units == 'score':
        row.update({'MSE': np.nan, 'MAE': np.nan, 'directional accuracy': np.nan})
    return row


def daily_rows(name, method, spec, results, experiment):
    daily = daily_scores(results, experiment['score_target']).reset_index().assign(model=name, validation=method)
    if spec.units == 'score':
        daily[['MSE', 'accuracy']] = np.nan
    return daily


def rescore(specs, experiment, out, predictions):
    summary_path = os.path.join(out, 'summary.csv')
    rows, daily = [], []
    for rec in pd.read_csv(summary_path).to_dict('records'):
        name, method = rec['model'], rec['validation']
        results = pd.read_parquet(prediction_path(predictions, name, method))
        rows.append(summary_row(name, method, specs[name], results, experiment, rec['train pct'],
                                rec['refit every (days)'], rec['refits'], rec['seconds']))
        if method != 'Random':
            daily.append(daily_rows(name, method, specs[name], results, experiment))
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    pd.concat(daily).to_csv(os.path.join(out, 'daily.csv'), index=False)
    print('rescored', len(rows), 'rows in', summary_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', choices=list(EXPERIMENTS), default='original')
    parser.add_argument('--data', help="defaults to the experiment's data file")
    parser.add_argument('--out', help="defaults to the experiment's results folder")
    parser.add_argument('--predictions', help="defaults to the experiment's predictions folder")
    parser.add_argument('--train-pct', type=float, default=0.8)
    parser.add_argument('--models', nargs='*')
    parser.add_argument('--methods', nargs='*', default=METHODS)
    parser.add_argument('--rescore', action='store_true', help='recompute every metric from the saved predictions')
    args = parser.parse_args()

    experiment = EXPERIMENTS[args.experiment]
    out = args.out or experiment['out']
    predictions = args.predictions or experiment['predictions']
    os.makedirs(out, exist_ok=True)
    os.makedirs(predictions, exist_ok=True)
    df = load_data(args.data or experiment['data'], experiment['build'])
    specs = experiment['specs'](df)
    if args.rescore:
        rescore(specs, experiment, out, predictions)
        return

    summary_path = os.path.join(out, 'summary.csv')
    done = set()
    if os.path.exists(summary_path):
        done = set(pd.read_csv(summary_path)[['model', 'validation']].itertuples(index=False, name=None))

    features, target = experiment['features'], experiment['fit_target']
    for name in args.models or specs:
        spec = specs[name]
        for method in args.methods:
            if (name, method) in done:
                continue
            start = time.time()
            if spec.members:
                results = ensemble(spec.members, method, predictions, experiment['score_target'])
                every, refits = None, None
            elif method == 'Random':
                results = holdout(df, spec.make, features, split=random_split, train_pct=args.train_pct, target=target)
                every, refits = None, 1
            elif method == 'Chronological':
                results = holdout(df, spec.make, features, split=chronological_split, train_pct=args.train_pct,
                                  target=target)
                every, refits = None, 1
            else:
                results = walk_forward(df, spec.make, features, train_pct=args.train_pct, refit_every=spec.refit,
                                       target=target)
                every, refits = spec.refit, refit_count(df, args.train_pct, spec.refit)
            seconds = time.time() - start

            row = summary_row(name, method, spec, results, experiment, args.train_pct, every, refits, seconds)
            append_csv(pd.DataFrame([row]), summary_path)
            if method != 'Random':
                append_csv(daily_rows(name, method, spec, results, experiment), os.path.join(out, 'daily.csv'))
            results[['date', 'ticker', experiment['score_target'], 'prediction']].to_parquet(
                prediction_path(predictions, name, method))
            print(f"{name:20s} {method:13s} cross-sectional acc {row['cross-sectional accuracy']:.4f}  "
                  f"acc {row['directional accuracy']:.4f}  IC {row['IC mean']:.4f}  {seconds:.0f}s", flush=True)


if __name__ == '__main__':
    main()
