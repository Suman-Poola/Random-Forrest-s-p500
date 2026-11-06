import io
import pandas as pd
import os
from dotenv import load_dotenv
import requests
import yfinance as yf

from Validation import chronological_split

load_dotenv()

FEATURES = [
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

def kaggle_tickers():
    import kaggle

    dataset_id = 'darkmatternet/s-and-p-500-stocks-25-years-of-data-updated-daily'
    path = 'ticker_list'
    kaggle.api.dataset_download_cli(
        dataset_id,
        path=str(path),
        unzip=True,
        force=True
    )

    tk = pd.read_csv('ticker_list/sp500_companies.csv')
    return tk['symbol'].tolist()

def wikipedia_tickers():
    # current S&P 500 constituents, for machines without Kaggle credentials
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    html = requests.get(url, headers={'User-Agent': 'QF_Indigo research script'}, timeout=30).text
    table = pd.read_html(io.StringIO(html), attrs={'id': 'constituents'})[0]
    # yfinance writes share classes with a dash, e.g. BRK-B
    return table['Symbol'].str.replace('.', '-', regex=False).tolist()

def build_dataset(years=3, train_pct=0.8, tickers=None):
    if tickers is None:
        tickers = kaggle_tickers()

    df = yf.download(tickers, period="10y")

    closes = df["Close"]
    returns = closes.pct_change(fill_method=None) * 100
    volumes = df["Volume"]

    cs = closes.stack().rename("Close")
    rs = returns.stack().rename("Returns")
    vs = volumes.stack().rename("Volume")

    fdf = pd.concat([cs, rs, vs], axis=1).reset_index()
    fdf.columns = ['date', 'ticker', 'close', 'return', 'volume']
    fdf = fdf.dropna(subset=['return'])

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

    # features are computed on all 10 years, then only the last `years` are kept
    df = df[df['date'] > df['date'].max() - pd.DateOffset(years=years)].copy()

    # one cutoff date for every ticker, so no training row comes after a test row
    train, test = chronological_split(df, train_pct)

    x_train = train[FEATURES]
    y_train = train['future return']
    x_test = test[FEATURES]
    y_test = test['future return']

    return df, x_test, x_train, y_train, y_test, encoding_table
