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

This model was developed by Suman 

The latest version is here:
https://colab.research.google.com/drive/1D9mQE4snDr0UY7Iy5Ns3TpUgP4WwWHQF#scrollTo=iNilF3mVK4BI
