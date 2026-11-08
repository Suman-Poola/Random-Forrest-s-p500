import numpy as np
import pandas as pd


def test_dates(df, train_pct=0.8):
    dates = np.sort(df['date'].unique())
    return dates[int(len(dates) * train_pct):]


def random_split(df, train_pct=0.8, seed=72):
    shuffled = df.sample(frac=1, random_state=seed)
    split = int(len(shuffled) * train_pct)
    return shuffled.iloc[:split], shuffled.iloc[split:]


def chronological_split(df, train_pct=0.8):
    cutoff = test_dates(df, train_pct)[0]
    return df[df['date'] < cutoff], df[df['date'] >= cutoff]


def holdout(df, make_model, features, split=chronological_split, train_pct=0.8, target='future return'):
    train, test = split(df, train_pct)
    model = make_model()
    model.fit(train[features], train[target])
    return test.assign(prediction=np.ravel(model.predict(test[features])))


def walk_forward(df, make_model, features, train_pct=0.8, refit_every=1, window=None, target='future return'):
    all_dates = np.sort(df['date'].unique())
    days = test_dates(df, train_pct)
    parts = []
    for start in range(0, len(days), refit_every):
        block = days[start:start + refit_every]
        train = df[df['date'] < block[0]]
        if window is not None:
            first = all_dates[max(0, np.searchsorted(all_dates, block[0]) - window)]
            train = train[train['date'] >= first]
        test = df[df['date'].isin(block)]
        model = make_model()
        model.fit(train[features], train[target])
        parts.append(test.assign(prediction=np.ravel(model.predict(test[features]))))
    return pd.concat(parts)


def refit_count(df, train_pct=0.8, refit_every=1):
    return -(-len(test_dates(df, train_pct)) // refit_every)


def daily_scores(results, target='future return', quantile=0.1):
    def one_day(g):
        y, p = g[target], g['prediction']
        scores = {'rows': len(g), 'MSE': ((p - y) ** 2).mean(), 'accuracy': ((p > 0) == (y > 0)).mean(),
                  'IC': np.nan, 'long-short': np.nan, 'cs accuracy': np.nan}
        if p.nunique() > 1:
            scores['cs accuracy'] = ((p > p.median()) == (y > y.median())).mean()
            scores['IC'] = p.rank().corr(y.rank())
            k = max(1, int(len(g) * quantile))
            ranked = y[p.sort_values(kind='stable').index]
            scores['long-short'] = ranked.iloc[-k:].mean() - ranked.iloc[:k].mean()
        return pd.Series(scores)

    return results.groupby('date')[[target, 'prediction']].apply(one_day)


def score(results, target='future return'):
    y, p = results[target], results['prediction']
    daily = daily_scores(results, target)
    return {
        'test rows': len(results),
        'test days': results['date'].nunique(),
        'first test day': pd.Timestamp(results['date'].min()).date(),
        'last test day': pd.Timestamp(results['date'].max()).date(),
        'MSE': ((p - y) ** 2).mean(),
        'MAE': (p - y).abs().mean(),
        'mean squared return': (y ** 2).mean(),
        'directional accuracy': ((p > 0) == (y > 0)).mean(),
        'share up': (y > 0).mean(),
        'cross-sectional accuracy': daily['cs accuracy'].mean(),
        'IC mean': daily['IC'].mean(),
        'IC std': daily['IC'].std(),
        'long-short mean': daily['long-short'].mean(),
        'long-short std': daily['long-short'].std(),
    }


def daily_edges(results, target='future return'):
    y, p = results[target], results['prediction']
    up = (y > 0).mean() >= 0.5
    by_day = results.groupby('date')
    correct_side = (p > by_day['prediction'].transform('median')) == (y > by_day[target].transform('median'))
    edges = pd.DataFrame({'date': results['date'], 'MSE gain': y ** 2 - (p - y) ** 2,
                          'accuracy edge': ((p > 0) == (y > 0)).astype(float) - ((y > 0) == up),
                          'cs edge': (correct_side - 0.5).where(by_day['prediction'].transform('nunique') > 1)})
    return edges.groupby('date').mean()


def t_stat(values):
    values = pd.Series(values).dropna()
    std = values.std()
    return values.mean() / std * np.sqrt(len(values)) if std > 1e-12 else np.nan
