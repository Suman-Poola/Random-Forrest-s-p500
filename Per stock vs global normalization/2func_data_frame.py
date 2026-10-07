import pandas as pd
import numpy as np
from dotenv import load_dotenv
import yfinance as yf

load_dotenv()
import kaggle


def build_dataset(time_period="36mo", normalization='none', df=None):
    if df is None:
        dataset_id = 'darkmatternet/s-and-p-500-stocks-25-years-of-data-updated-daily'
        kaggle.api.dataset_download_cli(dataset_id, path='ticker_list', unzip=True, force=True)

        tk = pd.read_csv('ticker_list/sp500_companies.csv')
        tickers = [t.replace('.', '-') for t in tk['symbol'].tolist()]

        data = yf.download(tickers, period=time_period)

        closes = data['Close']
        returns = closes.pct_change(fill_method=None) * 100
        volumes = data['Volume']

        cs = closes.stack().rename('close')
        rs = returns.stack().rename('return')
        vs = volumes.stack().rename('volume')

        df = pd.concat([cs, rs, vs], axis=1).reset_index()
        df.columns = ['date', 'ticker', 'close', 'return', 'volume']
        df = df.dropna(subset=['return'])
        df = df.sort_values(by=['ticker', 'date'])

        df['Lag 1'] = df.groupby('ticker')['close'].shift(1) / df['close']
        df['Lag 2'] = df.groupby('ticker')['close'].shift(2) / df['close']
        df['Lag 3'] = df.groupby('ticker')['close'].shift(3) / df['close']
        df['Lag 5'] = df.groupby('ticker')['close'].shift(5) / df['close']

        for w in [5, 10, 20, 30, 50]:
            df['SMA ' + str(w)] = df.groupby('ticker')['close'].transform(lambda x: x.rolling(window=w).mean()) / df['close']

        for w in [5, 10, 20, 30]:
            df['volatility ' + str(w)] = df.groupby('ticker')['return'].transform(lambda x: x.rolling(w).std())

        df['Price change per volume'] = 1000000 * (df['return'] / df['volume'])

        ticker_to_id = {t: i for i, t in enumerate(sorted(df['ticker'].unique()))}
        df['id'] = df['ticker'].map(ticker_to_id)

        df['mk cap'] = df['close'] * df['volume'] / 100000000

        df = df.replace([np.inf, -np.inf], np.nan).dropna()

        df['future return'] = df.groupby('ticker')['return'].shift(-1)
        df = df.dropna()

    encoding_table = df[['ticker', 'id']].drop_duplicates().sort_values('id').reset_index(drop=True)

    close_features = ['Lag 1', 'Lag 2', 'Lag 3', 'Lag 5', 'SMA 10', 'SMA 20', 'SMA 30', 'SMA 50']

    features = [
        'return',
        'Lag 1', 'Lag 2', 'Lag 3', 'Lag 5',
        'volatility 5', 'volatility 10', 'volatility 20', 'volatility 30',
        'SMA 10', 'SMA 20', 'SMA 30', 'SMA 50',
        'volume',
        'Price change per volume',
        'id',
        'mk cap'
    ]

    train_parts, test_parts = [], []

    for ticker, g in df.groupby('ticker'):
        split = int(len(g) * 0.8)
        train_parts.append(g.iloc[:split])
        test_parts.append(g.iloc[split:])

    train = pd.concat(train_parts)
    test = pd.concat(test_parts)

    if normalization == 'per_stock':
        mean = train.groupby('ticker')[close_features].mean()
        std = train.groupby('ticker')[close_features].std().replace(0, np.nan)
        for part in [train, test]:
            m = mean.reindex(part['ticker']).values
            s = std.reindex(part['ticker']).values
            part[close_features] = (part[close_features] - m) / s

    if normalization == 'cross_stock':
        for part in [train, test]:
            by_date = part.groupby('date')[close_features]
            part[close_features] = (part[close_features] - by_date.transform('mean')) / by_date.transform('std')

    train = train.dropna(subset=features)
    test = test.dropna(subset=features)

    x_train = train[features]
    y_train = train['future return']
    x_test = test[features]
    y_test = test['future return']

    return df, x_test, x_train, y_train, y_test, encoding_table

df, x_test, x_train, y_train, y_test, encoding_table = build_dataset("36mo", "none")

_, x_test_ps, x_train_ps, y_train_ps, y_test_ps, _ = build_dataset(normalization="per_stock", df=df)

_, x_test_cs, x_train_cs, y_train_cs, y_test_cs, _ = build_dataset(normalization="cross_stock", df=df)

from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
tree = RandomForestRegressor(max_depth = 5, random_state=72)

tree.fit(x_train_ps, y_train_ps)
predictions = tree.predict(x_test_ps)

score = mean_squared_error(y_test_ps, predictions)
print("score for the per stock normalization is: ", score)

tree.fit(x_train_cs, y_train_cs)
predictions = tree.predict(x_test_cs)

score = mean_squared_error(y_test_cs, predictions)
print("score for the overall normalizaiton is: ", score)

tree.fit(x_train, y_train)
predictions = tree.predict(x_test)

score = mean_squared_error(y_test, predictions)
print("score for the control grou with no normalization: ", score)

#score for the per stock normalization is:  6.683373054680258
#score for the overall normalizaiton is:  6.635398159743226
#score for the control grou with no normalization:  6.63165176420107