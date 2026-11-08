import io
import os

import numpy as np
import pandas as pd
import requests
import yfinance as yf

RANKED_FEATURES = [
    'return', 'Lag 2', 'Lag 3', 'Lag 5',
    'volatility 5', 'volatility 10', 'volatility 20', 'volatility 30',
    'SMA 5', 'SMA 10', 'SMA 20', 'SMA 30', 'SMA 50',
    'volume', 'relative volume', 'dollar volume', 'Price change per volume',
    'overnight gap', 'intraday return', 'range', 'range 5',
    'sector-relative return', 'sector-relative return 5',
    'days since earnings', 'last EPS surprise',
]
FLAG_FEATURES = ['earnings next session']
MARKET_FEATURES = ['market return', 'market return 5', 'dispersion', 'VIX', 'VIX change']
SECTORS = ['Communication Services', 'Consumer Discretionary', 'Consumer Staples', 'Energy', 'Financials',
           'Health Care', 'Industrials', 'Information Technology', 'Materials', 'Real Estate', 'Utilities']
SECTOR_FEATURES = [f'sector: {s}' for s in SECTORS]
FEATURES = RANKED_FEATURES + FLAG_FEATURES + MARKET_FEATURES + SECTOR_FEATURES
SEQUENCE_FEATURES = ['return', 'overnight gap', 'intraday return', 'range', 'relative volume', 'sector-relative return']

MAX_DAYS_SINCE_EARNINGS = 63


def constituents():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    html = requests.get(url, headers={'User-Agent': 'QF_Indigo research script'}, timeout=30).text
    table = pd.read_html(io.StringIO(html), attrs={'id': 'constituents'})[0]
    return pd.Series(table['GICS Sector'].to_numpy(), index=table['Symbol'].str.replace('.', '-', regex=False))


def earnings_dates(tickers, path='data/earnings.parquet'):
    if os.path.exists(path):
        return pd.read_parquet(path)
    parts = []
    for ticker in tickers:
        try:
            dates = yf.Ticker(ticker).get_earnings_dates(limit=60)
        except Exception as error:
            print(f'no earnings dates for {ticker}: {error}', flush=True)
            continue
        if dates is not None and len(dates):
            parts.append(pd.DataFrame({'ticker': ticker, 'time': dates.index.tz_convert('UTC'),
                                       'surprise': dates['Surprise(%)'].to_numpy()}))
    earnings = pd.concat(parts, ignore_index=True)
    earnings.to_parquet(path)
    return earnings


def earnings_features(df, earnings, calendar):
    local = earnings['time'].dt.tz_convert('America/New_York')
    known = local.dt.tz_localize(None).dt.normalize() + pd.to_timedelta((local.dt.hour >= 16).astype(int), unit='D')
    position = np.searchsorted(calendar, known.to_numpy())
    announcements = earnings.assign(day=position)[position < len(calendar)]
    announcements = (announcements.sort_values('surprise', na_position='first')
                     .drop_duplicates(['ticker', 'day'], keep='last').sort_values('day'))

    rows = pd.DataFrame({'ticker': df['ticker'].to_numpy(), 'day': np.searchsorted(calendar, df['date'].to_numpy()),
                         'row': np.arange(len(df))}).sort_values('day')
    right = announcements[['ticker', 'day', 'surprise']].rename(columns={'day': 'announced'})
    last = pd.merge_asof(rows, right, left_on='day', right_on='announced', by='ticker').sort_values('row')
    upcoming = pd.merge_asof(rows, right[['ticker', 'announced']], left_on='day', right_on='announced', by='ticker',
                             direction='forward', allow_exact_matches=False).sort_values('row')
    return pd.DataFrame({
        'days since earnings': (last['day'] - last['announced']).clip(upper=MAX_DAYS_SINCE_EARNINGS)
                               .fillna(MAX_DAYS_SINCE_EARNINGS).to_numpy(),
        'last EPS surprise': last['surprise'].fillna(0).to_numpy(),
        'earnings next session': (upcoming['announced'] - upcoming['day'] == 1).astype(float).to_numpy(),
    }, index=df.index)


def build_dataset(tickers=None, start='2016-10-31', end='2026-10-31', earnings_path='data/earnings.parquet'):
    sectors = constituents()
    if tickers is None:
        tickers = sectors.index.tolist()

    prices = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    close, open_, high, low, volume = (prices[field] for field in ['Close', 'Open', 'High', 'Low', 'Volume'])
    vix = yf.download(['^VIX'], start=start, end=end, auto_adjust=True, progress=False)['Close']['^VIX']

    ret = close.pct_change(fill_method=None) * 100
    ret5 = (close / close.shift(5) - 1) * 100
    sector_of = sectors.reindex(close.columns).fillna('Unknown')
    sector_mean = lambda frame: frame.T.groupby(sector_of).transform('mean').T
    day_range = (high - low) / close
    wide = {
        'return': ret,
        **{f'Lag {k}': close.shift(k) / close for k in (2, 3, 5)},
        **{f'volatility {w}': ret.rolling(w).std() for w in (5, 10, 20, 30)},
        **{f'SMA {w}': close.rolling(w).mean() / close for w in (5, 10, 20, 30, 50)},
        'volume': volume,
        'relative volume': volume / volume.rolling(20).mean(),
        'dollar volume': close * volume,
        'Price change per volume': 1e6 * ret / volume,
        'overnight gap': open_ / close.shift(1) - 1,
        'intraday return': close / open_ - 1,
        'range': day_range,
        'range 5': day_range.rolling(5).mean(),
        'sector-relative return': ret - sector_mean(ret),
        'sector-relative return 5': ret5 - sector_mean(ret5),
        'future return': ret.shift(-1),
    }
    df = pd.concat({name: frame.stack() for name, frame in wide.items()}, axis=1)
    df.index.names = ['date', 'ticker']
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index().sort_values(['date', 'ticker'], ignore_index=True)

    market = pd.DataFrame({'market return': ret.mean(axis=1), 'market return 5': ret5.mean(axis=1),
                           'dispersion': ret.std(axis=1), 'VIX': vix.ffill(),
                           'VIX change': vix.ffill().pct_change(fill_method=None) * 100})
    df = df.join(market, on='date')
    for sector, column in zip(SECTORS, SECTOR_FEATURES):
        df[column] = (df['ticker'].map(sector_of) == sector).astype(float)
    calendar = close.index.to_numpy()
    df = pd.concat([df, earnings_features(df, earnings_dates(close.columns.tolist(), earnings_path), calendar)], axis=1)
    df = df.dropna(subset=MARKET_FEATURES).reset_index(drop=True)

    by_day = df.groupby('date')
    df['excess return'] = df['future return'] - by_day['future return'].transform('median')
    low_cut = df['date'].map(df.groupby('date')['excess return'].quantile(0.01))
    high_cut = df['date'].map(df.groupby('date')['excess return'].quantile(0.99))
    df['target'] = df['excess return'].clip(low_cut, high_cut)

    ranked = df.groupby('date')[RANKED_FEATURES]
    df[RANKED_FEATURES] = 2 * (ranked.rank() - 1) / (ranked.transform('count') - 1) - 1
    df[FEATURES] = df[FEATURES].astype('float32')
    return df
