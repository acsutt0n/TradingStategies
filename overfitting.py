# overfitting.py - from:
# https://www.quantopian.com/posts/quantopian-lecture-series-overfitting


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from statsmodels import regression
from scipy import poly1d
from pairs_trading import *
import datetime



def example1():
  x = np.arange(10)
  y = 2*np.random.randn(10) + x**2
  xs = np.linspace(-0.25, 9.25, 200)
  #
  lin = np.polyfit(x, y, 1)
  quad = np.polyfit(x, y, 2)
  many = np.polyfit(x, y, 9)
  #
  plt.scatter(x, y)
  plt.plot(xs, poly1d(lin)(xs))
  plt.plot(xs, poly1d(quad)(xs))
  plt.plot(xs, poly1d(many)(xs))
  plt.ylabel('Y')
  plt.xlabel('X')
  plt.legend(['Underfit', 'Good fit', 'Overfit'])
  return



def overfit_stocks():
  # Load one year's worth of pricing data for five different assets
  start = datetime.date(1,1,2013)
  end = datetime.datetime(1,1,2014)
  x1 = get_pricing('PEP', )
  x2 = get_pricing('MCD', fields='price', start_date=start, end_date=end)
  x3 = get_pricing('ATHN', fields='price', start_date=start, end_date=end)
  x4 = get_pricing('DOW', fields='price', start_date=start, end_date=end)
  y = get_pricing('PG', fields='price', start_date=start, end_date=end)
  #
  # Build a linear model using only x1 to explain y
  slr = regression.linear_model.OLS(y, sm.add_constant(x1)).fit()
  slr_prediction = slr.params[0] + slr.params[1]*x1
  #
  # Run multiple linear regression using x1, x2, x3, x4 to explain y
  mlr = regression.linear_model.OLS(y, sm.add_constant(np.column_stack((x1,x2,x3,x4)))).fit()
  mlr_prediction = mlr.params[0] + mlr.params[1]*x1 + mlr.params[2]*x2 + mlr.params[3]*x3 + mlr.params[4]*x4
  #
  # Compute adjusted R-squared for the two different models
  print('SLR R-squared: %.5f' %slr.rsquared_adj)
  print('SLR p-value: %.5f' %slr.f_pvalue)
  print('MLR R-squared: %.5f' %mlr.rsquared_adj)
  print('MLR p-value: %.5f'  %mlr.f_pvalue)
  #
  # Plot y along with the two different predictions
  y.plot()
  slr_prediction.plot()
  mlr_prediction.plot()
  plt.ylabel('Price')
  plt.xlabel('Date')
  plt.legend(['PG', 'SLR', 'MLR']);


def overfit_stocks_2():
  # Load the next of pricing data
  start = '2014-01-01'
  end = '2015-01-01'
  x1 = get_pricing('PEP', fields='price', start_date=start, end_date=end)
  x2 = get_pricing('MCD', fields='price', start_date=start, end_date=end)
  x3 = get_pricing('ATHN', fields='price', start_date=start, end_date=end)
  x4 = get_pricing('DOW', fields='price', start_date=start, end_date=end)
  y = get_pricing('PG', fields='price', start_date=start, end_date=end)
  #
  # Extend our model from before to the new time period
  slr_prediction2 = slr.params[0] + slr.params[1]*x1
  mlr_prediction2 = mlr.params[0] + mlr.params[1]*x1 + mlr.params[2]*x2 + mlr.params[3]*x3 + mlr.params[4]*x4
  #
  # Manually compute adjusted R-squared over the new time period
  adj = float(len(y) - 1)/(len(y) - 5) # Compute adjustment factor
  SST = sum((y - np.mean(y))**2)
  SSRs = sum((slr_prediction2 - y)**2)
  print('SLR R-squared: %.5f' %(1 - adj*SSRs/SST))
  SSRm = sum((mlr_prediction2 - y)**2)
  print('MLR R-squared: %.5f' %(1 - adj*SSRm/SST))
  #
  # Plot y along with the two different predictions
  y.plot()
  slr_prediction2.plot()
  mlr_prediction2.plot()
  plt.ylabel('Price')
  plt.xlabel('Date')
  plt.legend(['PG', 'SLR', 'MLR']);



def rolling_windows():
  # Load the pricing data for a stock
  start = '2011-01-01'
  end = '2013-01-01'
  pricing = get_pricing('MCD', fields='price', start_date=start, end_date=end)
  #
  # Compute rolling averages for various window lengths
  mu_30d = pd.rolling_mean(pricing, window=30)
  mu_60d = pd.rolling_mean(pricing, window=60)
  mu_100d = pd.rolling_mean(pricing, window=100)
  #
  #  Plot asset pricing data with rolling means from the 100th day, when all the means become available
  plt.plot(pricing[100:], label='Asset')
  plt.plot(mu_30d[100:], label='30d MA')
  plt.plot(mu_60d[100:], label='60d MA')
  plt.plot(mu_100d[100:], label='100d MA')
  plt.xlabel('Day')
  plt.ylabel('Price')
  plt.legend();



# Trade using a simple mean-reversion strategy
def trade(stock, length):
  # If window length is 0, algorithm doesn't make sense, so exit
  if length == 0:
    return 0
  # Compute rolling mean and rolling standard deviation
  mu = pd.rolling_mean(stock, window=length)
  std = pd.rolling_std(stock, window=length)
  # Compute the z-scores for each day using the historical data up to that day
  zscores = (stock - mu)/std
  # Simulate trading
  # Start with no money and no positions
  money = 0
  count = 0
  for i in range(len(stock)):
    # Sell short if the z-score is > 1
    if zscores[i] > 1:
      money += stock[i]
      count -= 1
    # Buy long if the z-score is < 1
    elif zscores[i] < -1:
      money -= stock[i]
      count += 1
    # Clear positions if the z-score between -.5 and .5
    elif abs(zscores[i]) < 0.5:
      money += count*stock[i]
      count = 0
  return money



def best_window_length():
  # Find the window length 0-254 that gives the highest returns using this strategy
  length_scores = [trade(pricing, l) for l in range(255)]
  best_length = np.argmax(length_scores)
  print('Best window length: %.5f' %best_length)
  return



def example_2():
  # Get pricing data for a different timeframe
  start2 = '2013-01-01'
  end2 = '2015-01-01'
  pricing2 = get_pricing('MCD', fields='price', start_date=start2, end_date=end2)
  # Find the returns during this period using what we think is the best window length
  length_scores2 = [trade(pricing2, l) for l in range(255)]
  print(best_length, 'day window: %.5f' length_scores2[best_length])
  # Find the best window length based on this dataset, and the returns using this window length
  best_length2 = np.argmax(length_scores2)
  print(best_length2, 'day window: %.5f' %length_scores2[best_length2])
  plt.plot(length_scores)
  plt.plot(length_scores2)
  plt.xlabel('Window length')
  plt.ylabel('Score')
  plt.legend(['2011-2013', '2013-2015']);
  return
























