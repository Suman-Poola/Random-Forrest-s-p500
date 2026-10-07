from fastai import *
import pandas as pd
import os
import zipfile
import numpy as np
from dotenv import load_dotenv
import yfinance as yf
from sklearn.preprocessing import LabelEncoder

load_dotenv()
import kaggle

def build_dataset(time_period="36mo"):
    kaggle_username = os.getenv('KAGGLE_USERNAME')
    kaggle_key = os.getenv('KAGGLE_KEY')

    dataset_id = 'darkmatternet/s-and-p-500-stocks-25-years-of-data-updated-daily'
    path = 'ticker_list'
    kaggle.api.dataset_download_cli(
        dataset_id,
        path=str(path),
        unzip=True,
        force=True
    )

    tk = pd.read_csv('ticker_list/sp500_companies.csv')
    tickers = tk['symbol'].tolist()

    df = yf.download(tickers, period=time_period)

    closes = df["Close"]
    returns = closes.pct_change(fill_method=None) * 100
    volumes = df["Volume"]

    cs = closes.stack().rename("Close")
    rs = returns.stack().rename("Returns")
    vs = volumes.stack().rename("Volume")

    fdf = pd.concat([cs, rs, vs], axis=1).reset_index()
    fdf.columns = ['date', 'ticker', 'close', 'return', 'volume']
    fdf = fdf.dropna(subset=['return'])

    fdf.sort_values(by=['ticker', 'date'], inplace=True)
    fdf.sort_values(['ticker', 'date'], inplace=True)
    df = fdf

    df['Lag 1'] = df.groupby('ticker')['close'].shift(1) / df['close']
    df['Lag 2'] = df.groupby('ticker')['close'].shift(2) / df['close']
    df['Lag 3'] = df.groupby('ticker')['close'].shift(3) / df['close']
    df['Lag 5'] = df.groupby('ticker')['close'].shift(5) / df['close']

    df['SMA 5'] = df.groupby('ticker')['close'].transform(lambda x: x.rolling(window=5).mean()) / df['close']
    df['SMA 10'] = df.groupby('ticker')['close'].transform(lambda x: x.rolling(window=10).mean()) / df['close']
    df['SMA 20'] = df.groupby('ticker')['close'].transform(lambda x: x.rolling(window=20).mean()) / df['close']
    df['SMA 30'] = df.groupby('ticker')['close'].transform(lambda x: x.rolling(window=30).mean()) / df['close']
    df['SMA 50'] = df.groupby('ticker')['close'].transform(lambda x: x.rolling(window=50).mean()) / df['close']

    df['volatility 5'] = df.groupby('ticker')['return'].transform(lambda x: x.rolling(5).std())
    df['volatility 10'] = df.groupby('ticker')['return'].transform(lambda x: x.rolling(10).std())
    df['volatility 20'] = df.groupby('ticker')['return'].transform(lambda x: x.rolling(20).std())
    df['volatility 30'] = df.groupby('ticker')['return'].transform(lambda x: x.rolling(30).std())

    df['Price change per volume'] = 1000000 * (df['return'] / df['volume'])

    ticker_to_id = {ticker: i for i, ticker in enumerate(sorted(df['ticker'].unique()))}
    df['id'] = df['ticker'].map(ticker_to_id)
    encoding_table = pd.DataFrame(
        list(ticker_to_id.items()),
        columns=['ticker', 'id']
    )

    df['mk cap'] = df['close'] * df['volume'] / 100000000

    df = df.dropna()

    df['future return'] = df.groupby('ticker')['return'].shift(-1)
    df = df.dropna()

    features = [
        'return',
        'Lag 1',
        'Lag 2',
        'Lag 3',
        'Lag 5',
        'volatility 5',
        'volatility 10',
        'volatility 20',
        'volatility 30',
        'SMA 10',
        'SMA 20',
        'SMA 30',
        'SMA 50',
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

    x_train = train[features]
    y_train = train['future return']
    x_test = test[features]
    y_test = test['future return']

    return df, x_test, x_train, y_train, y_test, encoding_table