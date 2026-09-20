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


def build_dataset(time_period):
    kaggle_username = os.getenv('KAGGLE_USERNAME')
    kaggle_key = os.getenv('KAGGLE_KEY')

    dataset_id = 'darkmatternet/s-and-p-500-stocks-25-years-of-data-updated-daily'
    path = 'ticker_list'
    kaggle.api.dataset_download_cli(dataset_id, path = str(path), unzip=True, force=True)

    tk = pd.read_csv('ticker_list/sp500_companies.csv')

    tickers = tk['symbol'].tolist()
    tickers[0:25]
    len(tickers)
    tickers[499:503]
    df = yf.download(tickers, period=time_period)

    closes = df["Close"]
    returns = closes.pct_change() *100

    volumes = df['Volume']

    cs = closes.stack().rename("Close")
    rs = returns.stack().rename("Returns")
    vs = volumes.stack().rename("Volume")
    fdf = pd.concat([cs,rs,vs], axis = 1).reset_index()

    fdf.columns = ['date', 'ticker', 'close', 'return', 'volume']
    fdf = fdf.dropna(subset=['return'])

    fdf.sort_values(by=['ticker', 'date'], inplace=True)
    fdf.sort_values(['ticker', 'date'], inplace=True)
    df=fdf
    len(df)

    df['Lag 1'] = df.groupby('ticker')['close'].shift(1)
    df['Lag 2'] = df.groupby('ticker')['close'].shift(2)
    df['Lag 3'] = df.groupby('ticker')['close'].shift(3)
    df['Lag 5'] = df.groupby('ticker')['close'].shift(5)
    df['SMA 5'] = df.groupby('ticker')['close'].rolling(window=5).mean()
    df['SMA 10'] = df.groupby('ticker')['close'].rolling(window=10).mean()
    df['SMA 20'] = df.groupby('ticker')['close'].rolling(window=20).mean()
    df['SMA 30'] = df.groupby('ticker')['close'].rolling(window=30).mean()
    df['SMA 50'] = df.groupby('ticker')['close'].rolling(window=50).mean()

    df['volatility 5'] = df.groupby('ticker')['return'].rolling(5).std()
    df['volatility 10'] = df.groupby('ticker')['return'].rolling(10).std()
    df['volatility 20'] = df.groupby('ticker')['return'].rolling(20).std()
    df['volatility 30'] = df.groupby('ticker')['return'].rolling(30).std()

    df['Price change per volume'] = 1000000*(df['return']/df['volume'])

    df['id'] = LabelEncoder().fit_transform(df['ticker'])

    df['mk cap'] = df['close'] * df['volume'] / 100000000
    df = df.dropna()

    len(df)
    df['future return'] = df.groupby('ticker')['return'].shift(-1)
    df = df.dropna()
    

    return df