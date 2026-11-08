import argparse
import os

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.formatting.rule import FormulaRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from Model_Comparison import EXPERIMENTS, METHODS, prediction_path
from Validation import daily_edges, t_stat

FONT = 'Arial'
BLUE = Font(name=FONT, size=10, color='0000FF')
BLACK = Font(name=FONT, size=10)
GREY = Font(name=FONT, size=10, color='808080')
BOLD = Font(name=FONT, size=10, bold=True)
TITLE = Font(name=FONT, size=14, bold=True)
SUBTITLE = Font(name=FONT, size=11, bold=True)
NOTE = Font(name=FONT, size=9, italic=True, color='595959')
HEADER = Font(name=FONT, size=10, bold=True, color='FFFFFF')
HEADER_FILL = PatternFill('solid', fgColor='1F3864')
BEST_FILL = PatternFill('solid', fgColor='E2EFDA')
THIN = Border(bottom=Side(style='thin', color='BFBFBF'))
WRAP = Alignment(wrap_text=True, vertical='top')

PCT = '0.00%'
PCT_SIGNED = '+0.00%;-0.00%;0.00%'
MSE_FMT = '0.0000'
IC_FMT = '0.0000'
RATIO = '0.00'

SUMMARY_COLUMNS = [
    ('Model', 'model', None, None),
    ('Validation', 'validation', None, None),
    ('Refit every (days)', 'refit every (days)', '0', None),
    ('Refits', 'refits', '#,##0', None),
    ('Test rows', 'test rows', '#,##0', None),
    ('Test days', 'test days', '#,##0', None),
    ('First test day', 'first test day', 'yyyy-mm-dd', None),
    ('Last test day', 'last test day', 'yyyy-mm-dd', None),
    ('MSE', 'MSE', MSE_FMT, None),
    ('RMSE', None, MSE_FMT, '=IF(ISNUMBER(I{r}),SQRT(I{r}),"n/a")'),
    ('MAE', 'MAE', MSE_FMT, None),
    ('Zero-forecast MSE', 'mean squared return', MSE_FMT, None),
    ('Out-of-sample R² vs zero', None, PCT_SIGNED, '=IF(ISNUMBER(I{r}),1-I{r}/L{r},"n/a")'),
    ('Directional accuracy', 'directional accuracy', PCT, None),
    ('Share of up rows', 'share up', PCT, None),
    ('Naive direction accuracy', None, PCT, '=MAX(O{r},1-O{r})'),
    ('Accuracy edge vs naive', None, PCT_SIGNED, '=IF(ISNUMBER(N{r}),N{r}-P{r},"n/a")'),
    ('Mean daily IC', 'IC mean', IC_FMT, None),
    ('Daily IC std', 'IC std', IC_FMT, None),
    ('IC t-stat', None, RATIO, '=IF(COUNT(R{r},S{r})=2,R{r}/S{r}*SQRT(F{r}),"n/a")'),
    ('Long-short mean daily return (%)', 'long-short mean', '0.000', None),
    ('Long-short daily std (%)', 'long-short std', '0.000', None),
    ('Long-short annualized return (%)', None, '0.0', '=IF(COUNT(U{r})=1,U{r}*252,"n/a")'),
    ('Long-short annualized Sharpe', None, RATIO, '=IF(COUNT(U{r},V{r})=2,U{r}/V{r}*SQRT(252),"n/a")'),
    ('Cross-sectional accuracy', 'cross-sectional accuracy', PCT, None),
    ('Run time (s)', 'seconds', '#,##0', None),
]
SUMMARY_HEADER_ROW = 4
COL = {header: get_column_letter(i + 1) for i, (header, *_) in enumerate(SUMMARY_COLUMNS)}

SPLITS = {
    'Random': ('Random 80/20, shuffled rows', 'Yes'),
    'Chronological': ('Chronological 80/20, one fit', 'No'),
    'Walk-forward': ('Walk-forward, expanding window', 'No'),
}
PAPER_FORMATS = {
    'Model fits': ('#,##0', 0),
    'Test rows': ('#,##0', 0),
    'Directional accuracy (%)': ('0.00', 2),
    'Naive accuracy (%)': ('0.00', 2),
    'Accuracy edge (pp)': ('+0.00;-0.00;0.00', 2),
    'Accuracy edge t-stat': ('0.00', 2),
    'Cross-sectional accuracy (%)': ('0.00', 2),
    'Cross-sectional accuracy t-stat': ('0.00', 2),
    'MSE': ('0.0000', 4),
    'R² OS vs zero (%)': ('+0.000;-0.000;0.000', 3),
    'DM t-stat vs zero': ('0.00', 2),
    'Mean daily rank IC': ('0.0000', 4),
    'IC t-stat': ('0.00', 2),
    'Long-short return, annualized (%)': ('0.0', 1),
    'Long-short Sharpe, annualized': ('0.00', 2),
    'Run time (s)': ('#,##0.0', 1),
}
COMPARISON_FORMATS = {
    'Days compared': ('#,##0', 0),
    'Original cross-sectional accuracy (%)': ('0.00', 2),
    'Improved cross-sectional accuracy (%)': ('0.00', 2),
    'Original rank IC': ('0.0000', 4),
    'Improved rank IC': ('0.0000', 4),
    'Original long-short Sharpe': ('0.00', 2),
    'Improved long-short Sharpe': ('0.00', 2),
    'Full-period days': ('#,##0', 0),
    'Full-period cross-sectional accuracy (%)': ('0.00', 2),
    'Full-period cross-sectional t-stat': ('0.00', 2),
    'Full-period rank IC': ('0.0000', 4),
    'Full-period IC t-stat': ('0.00', 2),
    'Full-period long-short Sharpe': ('0.00', 2),
}

SETUP_NOTES = {
    'original': (
        [('Ticker universe', '{tickers} current S&P 500 constituents, list taken from Wikipedia on the run date '
                             '(the Colab notebooks use the Kaggle sp500_companies.csv list)'),
         ('Prices', 'yfinance daily data, 10 years downloaded; features use all 10 years, models use the last 3'),
         ('Target', "'future return': the ticker's next-day return, in percent"),
         ('Feature processing', 'Raw values; the neural network and LSTM z-score them on their training rows.')],
        [('Survivorship', "Today's constituents are used for the whole sample, so stocks that left the index are missing."),
         ('Reduced training', 'To keep walk-forward runs practical, the neural network trains for 5 epochs (the notebook '
                              'uses 100), the LSTM for 5 (the notebook uses 20), and the random forest uses 100 regularized '
                              'trees instead of 500 fully grown ones. All three validation methods use the same settings.'),
         ('One seed', 'Each model was trained once (seed 72), so small differences between models may not be meaningful.'),
         ("'mk cap'", 'close × volume / 1e8, which is dollar volume rather than market capitalization.'),
         ("'id'", 'Ticker code: a plain number for linear and tree models, an embedding for the neural network and LSTM.')],
    ),
    'improved': (
        [('Ticker universe', "{tickers} tickers: the original experiment's list, so both cover the same stocks. GICS "
                             'sectors come from the same Wikipedia table.'),
         ('Prices', 'yfinance daily OHLCV (adjusted), the same 10 years the original downloads, all used for training '
                    'and testing; VIX (^VIX) from yfinance; earnings dates and EPS surprises from Yahoo via yfinance.'),
         ('Target', "Models train on 'target': the next-day return minus that day's median next-day return across "
                    "tickers, clipped to the day's 1st-99th percentiles. They are scored on the same difference "
                    "unclipped ('excess return'), in percent."),
         ('Feature processing', 'Stock-level features are ranked across tickers within each day and scaled to [-1, 1]. '
                                'Market-level features (market return, 5-day market return, cross-sectional dispersion, '
                                'VIX, VIX change) are the same for every ticker on a day and stay raw. Sector is one-hot. '
                                'Ridge, logistic regression and the neural network also z-score every input on their '
                                'training rows.')],
        [('Survivorship', "Today's constituents and sectors are used for the whole sample, so stocks that left the "
                          'index are missing.'),
         ('Validation days', 'Tuned and early-stopped models use the last 60 trading days of each training window as '
                             'validation. Ridge, logistic regression and the decision tree pick a setting by mean daily '
                             'rank IC there and are refit on the whole window; XGBoost and the ranker pick their number '
                             'of trees there and are refit; the neural network and LSTM keep the early-stopped fit, so '
                             'they never train on the last 60 days of a window. Test days are never used.'),
         ('Refits', 'Walk-forward models refit every 21 trading days (the decision tree, neural network and LSTM every '
                    '63) instead of daily; in the original experiment refitting daily and fitting once gave nearly '
                    'the same results.'),
         ('Seeds', 'The neural network averages 5 seeds and the LSTM 3; the other models are fit once (seed 72).'),
         ('Earnings dates', 'The next-session flag assumes each announcement date was known the day before (they are '
                            "usually scheduled weeks ahead). Yahoo's history can be revised, and tickers without "
                            'earnings data get 63 days since earnings and a 0% surprise.'),
         ('Score-only models', 'Logistic regression, the rules, the XGBoost ranker and the ensemble output scores, not '
                               'returns, so their MSE, R² and directional accuracy are n/a.'),
         ('Ensemble', "Combines the saved predictions of the eight learners; its run time excludes theirs.")],
    ),
}

PAPER_NOTES = {
    'original': [
        ('Feature normalization', 'Linear regression and tree predictions do not change when features are rescaled, so '
                                  'those models use raw features. The neural network and LSTM z-score each continuous '
                                  "feature with the mean and standard deviation of that fit's training rows, pooled "
                                  'across tickers and recomputed at every refit. The target is never scaled.'),
        ('Ticker input', "The ticker's numeric code ('id'): a plain number for linear and tree models, a learned "
                         'embedding for the neural networks.'),
    ],
    'improved': [
        ('Feature normalization', 'Stock-level features are ranked across tickers within each day and scaled to [-1, 1], '
                                  'so no stock or outlier dominates. Market-level features stay raw; ridge, logistic '
                                  'regression and the neural network also z-score every input on their training rows. '
                                  "Training clips the target to the day's 1st-99th percentiles; scoring uses it unclipped."),
        ('Ticker input', 'No ticker code: GICS sector enters as one-hot columns.'),
        ('Score-only models', 'Logistic regression, the rules, the XGBoost ranker and the ensemble output scores rather '
                              'than returns, so their MSE, R², DM t-stat and directional accuracy are n/a.'),
    ],
}


def header_row(ws, row, labels, start_col=1):
    for i, label in enumerate(labels):
        cell = ws.cell(row=row, column=start_col + i, value=label)
        cell.font, cell.fill = HEADER, HEADER_FILL
        cell.alignment = Alignment(wrap_text=True, vertical='center', horizontal='center')


def widths(ws, values):
    for col, width in values.items():
        ws.column_dimensions[col].width = width


def write_summary(ws, summary):
    ws['A1'], ws['A1'].font = 'Model results by validation method', TITLE
    ws['A2'] = ('Blue values come from summary.csv (Model_Comparison.py); black values are formulas. '
                'Returns are daily, in percent. See the Setup sheet for definitions.')
    ws['A2'].font = NOTE
    header_row(ws, SUMMARY_HEADER_ROW, [h for h, *_ in SUMMARY_COLUMNS])
    ws.row_dimensions[SUMMARY_HEADER_ROW].height = 42
    for i, rec in enumerate(summary.to_dict('records')):
        r = SUMMARY_HEADER_ROW + 1 + i
        for c, (header, key, fmt, formula) in enumerate(SUMMARY_COLUMNS, start=1):
            if formula:
                value, font = formula.format(r=r), BLACK
            else:
                value, font = rec[key], BLUE
                if pd.isna(value):
                    if key == 'refit every (days)':
                        value = 'single fit' if rec['refits'] == 1 else 'n/a'
                    else:
                        value = None
                elif key in ('first test day', 'last test day'):
                    value = pd.Timestamp(value).to_pydatetime()
            cell = ws.cell(row=r, column=c, value=value)
            cell.font, cell.border = font, THIN
            if fmt:
                cell.number_format = fmt
    ws.freeze_panes = ws.cell(row=SUMMARY_HEADER_ROW + 1, column=3)
    widths(ws, {'A': 19, 'B': 14})
    for c in range(3, len(SUMMARY_COLUMNS) + 1):
        ws.column_dimensions[get_column_letter(c)].width = 13
    return SUMMARY_HEADER_ROW + len(summary)


def lookup(column, last_row, model_cell, method):
    rng = lambda col: f"Summary!${col}${SUMMARY_HEADER_ROW + 1}:${col}${last_row}"
    return (f'IFERROR(AVERAGEIFS({rng(COL[column])},{rng("A")},{model_cell},{rng("B")},"{method}"),"n/a")')


def write_comparison(ws, models, last_row):
    ws['A1'], ws['A1'].font = 'Model comparison: random vs chronological vs walk-forward validation', TITLE
    ws['A2'] = ('Chronological and walk-forward score the same test days; the random split scores a shuffled 20% of all rows. '
                'Green marks the best walk-forward value in each table. Every cell is a formula on the Summary sheet.')
    ws['A2'].font = NOTE
    blocks = [
        ("Cross-sectional accuracy (share of tickers put on the correct side of the day's median; 50% is a coin flip)",
         'Cross-sectional accuracy', PCT,
         ('Walk-forward vs 50%', lambda r: f'=IFERROR(D{r}-0.5,"n/a")', PCT_SIGNED), 'max'),
        ('Out-of-sample R² vs always predicting 0% (higher is better)', 'Out-of-sample R² vs zero', PCT_SIGNED,
         ('Random − walk-forward', lambda r: f'=IFERROR(B{r}-D{r},"n/a")', PCT_SIGNED), 'max'),
        ('Test MSE (lower is better)', 'MSE', MSE_FMT,
         ('Walk-forward vs zero forecast', lambda r: f'=IFERROR(D{r}/D${{zero}}-1,"n/a")', PCT_SIGNED), 'min'),
        ('Directional accuracy (share of rows where the predicted sign matched the actual sign)',
         'Directional accuracy', PCT,
         ('Walk-forward edge vs naive', lambda r: '=' + lookup('Accuracy edge vs naive', last_row, f'$A{r}', 'Walk-forward'), PCT_SIGNED), 'max'),
        ('Mean daily rank IC (Spearman correlation of predicted vs actual returns across tickers)', 'Mean daily IC', IC_FMT,
         ('Walk-forward IC t-stat', lambda r: '=' + lookup('IC t-stat', last_row, f'$A{r}', 'Walk-forward'), RATIO), 'max'),
        ('Top-minus-bottom decile long-short, annualized Sharpe (before costs)', 'Long-short annualized Sharpe', RATIO,
         ('Walk-forward annualized return (%)', lambda r: '=' + lookup('Long-short annualized return (%)', last_row, f'$A{r}', 'Walk-forward'), '0.0'), 'max'),
    ]
    row = 4
    for title, column, fmt, (extra_label, extra_formula, extra_fmt), best in blocks:
        ws.cell(row=row, column=1, value=title).font = SUBTITLE
        header_row(ws, row + 1, ['Model', *METHODS, extra_label])
        ws.row_dimensions[row + 1].height = 30
        first = row + 2
        zero = first + models.index('Zero forecast') if 'Zero forecast' in models else first
        for i, model in enumerate(models):
            r = first + i
            ws.cell(row=r, column=1, value=model).font = BOLD
            for j, method in enumerate(METHODS, start=2):
                cell = ws.cell(row=r, column=j, value='=' + lookup(column, last_row, f'$A{r}', method))
                cell.number_format, cell.font = fmt, BLACK
            cell = ws.cell(row=r, column=5, value=extra_formula(r).replace('{zero}', str(zero)))
            cell.number_format, cell.font = extra_fmt, BLACK
            for c in range(1, 6):
                ws.cell(row=r, column=c).border = THIN
        last = first + len(models) - 1
        pick = 'MAX' if best == 'max' else 'MIN'
        ws.conditional_formatting.add(
            f'D{first}:D{last}',
            FormulaRule(formula=[f'AND(ISNUMBER(D{first}),D{first}={pick}($D${first}:$D${last}))'], fill=BEST_FILL))
        row = last + 3
    widths(ws, {'A': 22, 'B': 15, 'C': 15, 'D': 15, 'E': 22})


def write_daily(ws, daily):
    header_row(ws, 1, ['Date', 'Model', 'Validation', 'Month', 'Rows', 'MSE', 'Directional accuracy',
                       'Rank IC', 'Long-short return (%)', 'Cross-sectional accuracy'])
    blank = lambda value: None if pd.isna(value) else value
    for i, rec in enumerate(daily.to_dict('records'), start=2):
        values = [pd.Timestamp(rec['date']).to_pydatetime(), rec['model'], rec['validation'],
                  f'=YEAR(A{i})*100+MONTH(A{i})', rec['rows'], blank(rec['MSE']), blank(rec['accuracy']),
                  blank(rec['IC']), blank(rec['long-short']), blank(rec['cs accuracy'])]
        formats = ['yyyy-mm-dd', None, None, '0', '0', MSE_FMT, PCT, IC_FMT, '0.000', PCT]
        for c, (value, fmt) in enumerate(zip(values, formats), start=1):
            cell = ws.cell(row=i, column=c, value=value)
            cell.font = BLACK if c == 4 else BLUE
            if fmt:
                cell.number_format = fmt
    ws.freeze_panes = 'A2'
    ws.auto_filter.ref = f'A1:J{len(daily) + 1}'
    widths(ws, {'A': 12, 'B': 19, 'C': 14, 'D': 9, 'E': 7, 'F': 10, 'G': 12, 'H': 10, 'I': 13, 'J': 13})
    return len(daily) + 1


def write_monthly(ws, models, months, daily_last):
    ws['A1'], ws['A1'].font = 'Walk-forward results by month', TITLE
    ws['A2'] = ('Averages of the per-day values on the Daily sheet. Months are YYYYMM; '
                'the first and last month can be partial. Constant forecasts have no IC or cross-sectional accuracy.')
    ws['A2'].font = NOTE
    rng = lambda col: f'Daily!${col}$2:${col}${daily_last}'
    row = 4
    for title, col, fmt in [('Mean daily cross-sectional accuracy', 'J', PCT), ('Mean daily rank IC', 'H', IC_FMT),
                            ('Mean daily directional accuracy', 'G', PCT)]:
        ws.cell(row=row, column=1, value=title).font = SUBTITLE
        header_row(ws, row + 1, ['Month', 'Test days', *models])
        ws.row_dimensions[row + 1].height = 30
        first = row + 2
        for i, month in enumerate(months):
            r = first + i
            ws.cell(row=r, column=1, value=int(month)).font = BLUE
            days = f'=COUNTIFS({rng("B")},C${row + 1},{rng("C")},"Walk-forward",{rng("D")},$A{r})'
            ws.cell(row=r, column=2, value=days).font = BLACK
            for j in range(len(models)):
                c = 3 + j
                letter = get_column_letter(c)
                cell = ws.cell(row=r, column=c, value=(
                    f'=IFERROR(AVERAGEIFS({rng(col)},{rng("B")},{letter}${row + 1},'
                    f'{rng("C")},"Walk-forward",{rng("D")},$A{r}),"n/a")'))
                cell.font, cell.number_format = BLACK, fmt
        r = first + len(months)
        ws.cell(row=r, column=1, value='All months').font = BOLD
        ws.cell(row=r, column=2, value=f'=SUM(B{first}:B{r - 1})').font = BOLD
        for j in range(len(models)):
            letter = get_column_letter(3 + j)
            cell = ws.cell(row=r, column=3 + j, value=(
                f'=IFERROR(AVERAGEIFS({rng(col)},{rng("B")},{letter}${row + 1},{rng("C")},"Walk-forward"),"n/a")'))
            cell.font, cell.number_format = BOLD, fmt
        for rr in range(first, r + 1):
            for c in range(1, 3 + len(models)):
                ws.cell(row=rr, column=c).border = THIN
        row = r + 3
    widths(ws, {'A': 11, 'B': 10})
    for j in range(len(models)):
        ws.column_dimensions[get_column_letter(3 + j)].width = 14


def write_setup(ws, summary, df, experiment_name):
    ws['A1'], ws['A1'].font = 'Setup, definitions and caveats', TITLE
    chrono = summary[summary['validation'] == 'Chronological'].iloc[0]
    data_rows, caveats = SETUP_NOTES[experiment_name]
    data_rows = [(key, text.replace('{tickers}', str(df['ticker'].nunique()))) for key, text in data_rows]
    rows = [
        ('Data', None),
        *data_rows[:2],
        ('Sample', f"{pd.Timestamp(df['date'].min()).date()} to {pd.Timestamp(df['date'].max()).date()}, "
                   f"{df['date'].nunique()} trading days, {len(df):,} ticker-days"),
        *data_rows[2:3],
        ('Features', ', '.join(EXPERIMENTS[experiment_name]['features'])),
        *data_rows[3:],
        ('Train share', f"{chrono['train pct']:.0%} of trading days (train_pct); the test period is the remaining days"),
        ('Test period', f"{chrono['first test day']} to {chrono['last test day']}, {chrono['test days']} trading days, "
                        f"{chrono['test rows']:,} ticker-days"),
        (None, None),
        ('Validation methods', None),
        ('Random', 'Rows shuffled (seed 72), then split 80/20. Training contains days after the test days, '
                   'so this is only a baseline for how much shuffling flatters a model.'),
        ('Chronological', 'One fit on every day before the cutoff, then predicts every test day with that model.'),
        ('Walk-forward', 'Expanding window, one day ahead: each test day is predicted by a model fit only on earlier '
                         'days. The model is refit every N trading days (see Model settings below) and the latest fit '
                         'is reused in between, so it never sees the day it predicts.'),
        (None, None),
        ('Metrics', None),
        ('MSE / RMSE / MAE', 'Errors against the scored target, in percent points, over every test row.'),
        ('Out-of-sample R² vs zero', '1 − MSE / MSE of always predicting 0%. Above 0 means the model beats the zero forecast.'),
        ('Directional accuracy', 'Share of rows where the prediction and the target are both above 0, or both not.'),
        ('Naive direction accuracy', 'Accuracy of always guessing the more common direction in the test rows.'),
        ('Cross-sectional accuracy', "Each day, the share of tickers on the correct side of the median: predicted above "
                                     "the day's median prediction and actually above the median return, or both below. "
                                     'Averaged over test days. 50% is a coin flip, whichever way the market moves.'),
        ('Mean daily IC', 'Average over test days of the Spearman rank correlation between predicted and actual returns '
                          'across tickers. t-stat = mean / std × √days, treating days as independent.'),
        ('Long-short', 'Each day: average actual return of the top 10% of tickers by prediction minus the bottom 10%. '
                       'Equal weight, daily rebalance, no trading costs. Sharpe = mean / std × √252.'),
        (None, None),
        ('Caveats', None),
        *caveats,
    ]
    r = 3
    for key, text in rows:
        if key and text is None:
            ws.cell(row=r, column=1, value=key).font = SUBTITLE
        elif key:
            ws.cell(row=r, column=1, value=key).font = BOLD
            ws.cell(row=r, column=2, value=text).font = BLACK
            ws.cell(row=r, column=1).alignment = ws.cell(row=r, column=2).alignment = WRAP
        r += 1

    r += 1
    ws.cell(row=r, column=1, value='Model settings').font = SUBTITLE
    r += 1
    header_row(ws, r, ['Model', 'Settings', 'Walk-forward refit every (days)', 'Walk-forward refits'])
    ws.row_dimensions[r].height = 30
    wf = summary[summary['validation'] == 'Walk-forward'].set_index('model')
    for model, settings in summary.drop_duplicates('model')[['model', 'settings']].itertuples(index=False):
        r += 1
        ws.cell(row=r, column=1, value=model).font = BOLD
        ws.cell(row=r, column=2, value=settings).font = BLUE
        if model in wf.index and pd.notna(wf.loc[model, 'refits']):
            ws.cell(row=r, column=3, value=int(wf.loc[model, 'refit every (days)'])).font = BLUE
            ws.cell(row=r, column=4, value=int(wf.loc[model, 'refits'])).font = BLUE
        for c in range(1, 5):
            ws.cell(row=r, column=c).alignment = WRAP
            ws.cell(row=r, column=c).border = THIN

    r += 2
    ws.cell(row=r, column=1, value='Legend').font = SUBTITLE
    ws.cell(row=r + 1, column=1, value='Blue text').font = BLUE
    ws.cell(row=r + 1, column=2, value='Value written from the model runs (summary.csv, daily.csv)').font = BLACK
    ws.cell(row=r + 2, column=1, value='Black text').font = BLACK
    ws.cell(row=r + 2, column=2, value='Excel formula').font = BLACK
    widths(ws, {'A': 26, 'B': 110, 'C': 16, 'D': 12})


def paper_table(summary, df, predictions, target):
    days = df['date'].nunique()
    rows = []
    for rec in summary.to_dict('records'):
        model, method = rec['model'], rec['validation']
        split, future = SPLITS[method]
        every, train_days = rec['refit every (days)'], int(days * rec['train pct'])
        if pd.isna(rec['refits']):
            refit = trained_on = "As its members"
        elif method == 'Random':
            refit, trained_on = 'Once', f"{rec['train pct']:.0%} of rows from all {days} days"
        elif method == 'Chronological':
            refit, trained_on = 'Once', f'First {train_days}'
        else:
            refit = 'Every day' if every == 1 else f'Every {every:.0f} days'
            trained_on = f"First {train_days}, growing to {train_days + (rec['test days'] - 1) // every * every:.0f}"
        edges = daily_edges(pd.read_parquet(prediction_path(predictions, model, method)), target)
        returns = rec['units'] == 'return'
        naive = max(rec['share up'], 1 - rec['share up'])
        rows.append({
            'Model': model,
            'Model family': rec['family'],
            'Target': rec['target'],
            'Train/test split': split,
            'Future data in training': future,
            'Refit': refit,
            'Model fits': rec['refits'],
            'Training days': trained_on,
            'Test period': f"{rec['first test day']} to {rec['last test day']}",
            'Test rows': rec['test rows'],
            'Feature normalization': rec['normalization'],
            'Ticker input': rec['ticker input'],
            'Inputs per prediction': rec['inputs'],
            'Hyperparameters': rec['settings'],
            'Directional accuracy (%)': 100 * rec['directional accuracy'],
            'Naive accuracy (%)': 100 * naive,
            'Accuracy edge (pp)': 100 * (rec['directional accuracy'] - naive),
            'Accuracy edge t-stat': t_stat(edges['accuracy edge']) if returns else np.nan,
            'Cross-sectional accuracy (%)': 100 * rec['cross-sectional accuracy'],
            'Cross-sectional accuracy t-stat': t_stat(edges['cs edge']),
            'MSE': rec['MSE'],
            'R² OS vs zero (%)': 100 * (1 - rec['MSE'] / rec['mean squared return']),
            'DM t-stat vs zero': t_stat(edges['MSE gain']) if returns else np.nan,
            'Mean daily rank IC': rec['IC mean'],
            'IC t-stat': rec['IC mean'] / rec['IC std'] * np.sqrt(rec['test days']),
            'Long-short return, annualized (%)': 252 * rec['long-short mean'],
            'Long-short Sharpe, annualized': rec['long-short mean'] / rec['long-short std'] * np.sqrt(252),
            'Run time (s)': rec['seconds'],
        })
    return pd.DataFrame(rows)


def original_vs_improved(summary, daily, original_daily):
    walk = lambda frame: frame[frame['validation'] == 'Walk-forward']
    sharpe = lambda days: days['long-short'].mean() / days['long-short'].std() * np.sqrt(252)
    improved, original = walk(daily), walk(original_daily)
    rows = []
    for rec in walk(summary).to_dict('records'):
        mine = improved[improved['model'] == rec['model']]
        on_common = mine[mine['date'].isin(original['date'])]
        replaces = rec['original model'] if isinstance(rec['original model'], str) else None
        theirs = original[(original['model'] == replaces) & original['date'].isin(on_common['date'])]
        rows.append({
            'Model': rec['model'],
            'Replaces': replaces or 'New model',
            'Days compared': len(on_common),
            'Original cross-sectional accuracy (%)': 100 * theirs['cs accuracy'].mean(),
            'Improved cross-sectional accuracy (%)': 100 * on_common['cs accuracy'].mean(),
            'Original rank IC': theirs['IC'].mean(),
            'Improved rank IC': on_common['IC'].mean(),
            'Original long-short Sharpe': sharpe(theirs),
            'Improved long-short Sharpe': sharpe(on_common),
            'Full-period days': len(mine),
            'Full-period cross-sectional accuracy (%)': 100 * mine['cs accuracy'].mean(),
            'Full-period cross-sectional t-stat': t_stat(mine['cs accuracy'] - 0.5),
            'Full-period rank IC': mine['IC'].mean(),
            'Full-period IC t-stat': t_stat(mine['IC']),
            'Full-period long-short Sharpe': sharpe(mine),
        })
    return pd.DataFrame(rows)


def write_values(ws, table, formats, start_row, grey=lambda rec: False):
    header_row(ws, start_row, list(table.columns))
    ws.row_dimensions[start_row].height = 45
    for i, rec in enumerate(table.to_dict('records')):
        r = start_row + 1 + i
        font = GREY if grey(rec) else BLACK
        for c, (column, value) in enumerate(rec.items(), start=1):
            cell = ws.cell(row=r, column=c, value='n/a' if pd.isna(value) else value)
            cell.font, cell.border, cell.alignment = font, THIN, WRAP
            if column in formats:
                cell.number_format = formats[column][0]


def write_notes(ws, row, notes):
    ws.cell(row=row, column=1, value='Notes').font = SUBTITLE
    for key, text in notes:
        row += 1
        ws.cell(row=row, column=1, value=key).font = BOLD
        ws.cell(row=row, column=2, value=text).font = NOTE


def write_paper_table(ws, table, df, experiment_name, csv_name):
    ws['A1'], ws['A1'].font = 'Model specifications and test results', TITLE
    ws['A2'] = (f'Plain values, not formulas, for copying into a paper; the same table is in {csv_name}. '
                'Grey rows (random split) train on days after the days they test on and are only a leakage baseline.')
    ws['A2'].font = NOTE
    write_values(ws, table, PAPER_FORMATS, 4, grey=lambda rec: rec['Future data in training'] == 'Yes')
    features = EXPERIMENTS[experiment_name]['features']
    notes = [
        ('Sample', f"{df['ticker'].nunique()} S&P 500 tickers, {pd.Timestamp(df['date'].min()).date()} to "
                   f"{pd.Timestamp(df['date'].max()).date()} ({df['date'].nunique()} trading days). "
                   f"{len(features)} features, all known at the close of day t. See the Setup sheet."),
        *PAPER_NOTES[experiment_name],
        ('Naive accuracy', "Always guessing the test rows' more common direction."),
        ('Accuracy edge t-stat', "Each test day's directional accuracy minus the naive rule's, t-tested across days."),
        ('Cross-sectional accuracy', "Each day, the share of tickers on the correct side of the day's median (predicted "
                                     'and actual), averaged over days; 50% is a coin flip whatever the market does. The '
                                     't-stat tests the daily values against 50%.'),
        ('R² OS vs zero', '1 − MSE / MSE of always predicting 0%, as in Gu, Kelly and Xiu (2020, Review of Financial '
                          'Studies). Above 0 beats the zero forecast.'),
        ('DM t-stat vs zero', 'Diebold-Mariano test against the zero forecast, on the daily cross-sectional mean of the '
                              'squared-error difference. Positive favours the model.'),
        ('Rank IC / long-short', 'Daily Spearman correlation of predicted and actual returns across tickers, and the '
                                 'top-minus-bottom decile return before trading costs (definitions on the Setup sheet).'),
        ('t-stats', 'Treat test days as independent; |t| > 1.96 is about the 5% two-sided level. There is no correction '
                    'for comparing many models.'),
        ('Run time', 'Wall-clock seconds for all fits of that row, on CPU. Indicative only: it varies with machine load.'),
    ]
    write_notes(ws, 6 + len(table), notes)
    ws.freeze_panes = 'B5'
    widths(ws, {'A': 18, 'B': 22, 'C': 18, 'D': 22, 'E': 10, 'F': 12, 'G': 8, 'H': 18, 'I': 23, 'J': 9,
                'K': 22, 'L': 15, 'M': 12, 'N': 40})
    for c in range(15, len(table.columns) + 1):
        ws.column_dimensions[get_column_letter(c)].width = 11


def write_original_vs_improved(ws, table, csv_name):
    ws['A1'], ws['A1'].font = 'Original vs improved, walk-forward', TITLE
    ws['A2'] = (f'Plain values; the same table is in {csv_name}. The "original" and "improved" columns score both '
                "experiments on the original experiment's walk-forward test days; the full-period columns use the "
                "improved experiment's whole test period.")
    ws['A2'].font = NOTE
    write_values(ws, table, COMPARISON_FORMATS, 4)
    notes = [
        ('Cross-sectional accuracy', "Share of tickers on the correct side of the day's median, averaged over days. "
                                     'The original models predict raw returns, but this measure only uses their order '
                                     'within each day, so it compares both experiments on the same footing.'),
        ('t-stats', 'Daily values tested against 50% (accuracy) or 0 (IC), treating days as independent.'),
        ('n/a', 'Constant forecasts have no ordering, and new models have no original counterpart.'),
    ]
    write_notes(ws, 6 + len(table), notes)
    ws.freeze_panes = 'B5'
    widths(ws, {'A': 20, 'B': 18})
    for c in range(3, len(table.columns) + 1):
        ws.column_dimensions[get_column_letter(c)].width = 13


def save_csv(table, formats, path):
    table.round({column: decimals for column, (_, decimals) in formats.items()}).to_csv(
        path, index=False, encoding='utf-8-sig')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', choices=list(EXPERIMENTS), default='original')
    parser.add_argument('--results', help="defaults to the experiment's results folder")
    parser.add_argument('--data', help="defaults to the experiment's data file")
    parser.add_argument('--predictions', help="defaults to the experiment's predictions folder")
    args = parser.parse_args()
    experiment = EXPERIMENTS[args.experiment]
    results = args.results or experiment['out']

    summary = pd.read_csv(f'{results}/summary.csv')
    daily = pd.read_csv(f'{results}/daily.csv', parse_dates=['date'])
    df = pd.read_parquet(args.data or experiment['data'], columns=['date', 'ticker'])
    order = list(dict.fromkeys(summary['model']))
    summary = summary.assign(model=pd.Categorical(summary['model'], order),
                             validation=pd.Categorical(summary['validation'], METHODS)).sort_values(['model', 'validation'])
    summary['model'] = summary['model'].astype(str)
    summary['validation'] = summary['validation'].astype(str)
    daily = daily.sort_values(['validation', 'model', 'date'])

    wb = Workbook()
    comparison = wb.active
    comparison.title = 'Comparison'
    last_row = write_summary(wb.create_sheet('Summary'), summary)
    write_comparison(comparison, order, last_row)
    months = sorted((daily['date'].dt.year * 100 + daily['date'].dt.month).unique())
    write_monthly(wb.create_sheet('Monthly'), order, months, len(daily) + 1)
    write_daily(wb.create_sheet('Daily'), daily)
    write_setup(wb.create_sheet('Setup'), summary, df, args.experiment)
    table = paper_table(summary, df, args.predictions or experiment['predictions'], experiment['score_target'])
    write_paper_table(wb.create_sheet('Paper table', 1), table, df, args.experiment, 'paper_table.csv')
    save_csv(table, PAPER_FORMATS, f'{results}/paper_table.csv')

    original_daily = os.path.join(EXPERIMENTS['original']['out'], 'daily.csv')
    if args.experiment == 'improved' and os.path.exists(original_daily):
        versus = original_vs_improved(summary, daily, pd.read_csv(original_daily, parse_dates=['date']))
        write_original_vs_improved(wb.create_sheet('Original vs improved', 2), versus, 'original_vs_improved.csv')
        save_csv(versus, COMPARISON_FORMATS, f'{results}/original_vs_improved.csv')

    wb.calculation.fullCalcOnLoad = True
    path = f'{results}/Model_Comparison.xlsx'
    wb.save(path)
    print('wrote', path, 'and its CSVs')


if __name__ == '__main__':
    main()
