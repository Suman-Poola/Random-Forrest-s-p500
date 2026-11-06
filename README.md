# Random-Forrest-s-p500
Random forest model to predict stock returns 

This model imports the tickers from a public domain kaggle dataset
'S&P 500 Stocks: 25 Years of Data (Updated Daily)'

further financial data is donwloaded from Yahoo Finance (yfinance)

Inspiration and guidence for this model was from Jeremy Howard in Fast.ai

This random forest contains 100 trees, each is very complex and has a great chance to be overfit 
The last cell in this notebook is AI generated and plots every single tree in one image, I recomend not running that cell.
The last cell took me 2 hours to run.
My result from the random forest is in random forest.png

This model uses closing price, 1 day lag, two day lag, 3 day lag, 5 day lag, 5 day volatility, 10 day volatility, 20 day volatility, 10 day simple moving average, 20 day simple moving average, and simple moving average ratio to predict the next day's stock motion, either up (1) or down (0)

Furthur developments must be made to optimize this model. as it had only a 0.2% advantage. 

## Validation and model comparison

`Validation.py` has three ways to test a model on the `build_dataset()` frame, all taking `train_pct` (default 0.8):

- `random_split`: shuffled rows. Kept only as a baseline, since training then contains days after the test days.
- `chronological_split`: one cutoff date for every ticker; train before it, test from it onward. `build_dataset(train_pct=...)` now uses this.
- `walk_forward`: one-day-ahead validation. Before each test day the model is refit on every earlier day, predicts that day, and the day is added to training. `refit_every=N` refits every N days for slow models.

`Model_Comparison.py` runs every model under all three methods and writes `results/summary.csv` and `results/daily.csv`. `Results_Workbook.py` turns those into `results/Model_Comparison.xlsx`, and also writes `results/paper_table.csv` (the workbook's Paper table sheet): one row per model and split, with the model family, split, refit schedule, normalization, ticker handling, hyperparameters, accuracy and its edge over a naive guess, out-of-sample R², rank IC, long-short Sharpe and t-stats.

```
python Model_Comparison.py
python Results_Workbook.py
```

Without Kaggle credentials, pass `tickers=wikipedia_tickers()` to `build_dataset`; `Model_Comparison.py` does this.

This model was developed by Suman 

The latest version is here:
https://colab.research.google.com/drive/1D9mQE4snDr0UY7Iy5Ns3TpUgP4WwWHQF#scrollTo=iNilF3mVK4BI
