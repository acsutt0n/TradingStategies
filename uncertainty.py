# uncertainty.py -- Based on the quantopian lecture
# "you don't know how wrong you are" 
# https://www.quantopian.com/posts/quantopian-lecture-series-you-dont-know-how-wrong-you-are


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime 
from pairs_trading import *



def example1():
  # Set a seed so we can play with the data without generating new random numbers every time
  np.random.seed(123)
  normal = np.random.randn(500)
  print(np.mean(normal[:10]))
  print(np.mean(normal[:100]))
  print(np.mean(normal[:250]))
  print(np.mean(normal))
  # Plot a stacked histogram of the data
  plt.hist([normal[:10], normal[10:100], normal[100:250], normal], normed=1, histtype='bar', stacked=True);
  plt.ylabel('Frequency')
  plt.xlabel('Value');
  print(np.std(normal[:10]))
  print(np.std(normal[:100]))
  print(np.std(normal[:250]))
  print(np.std(normal))
  plt.show()
  return



#Generate some data from a bi-modal distribution
def bimodal(n):
    X = np.zeros((n))
    for i in range(n):
        if np.random.binomial(1, 0.5) == 0:
            X[i] = np.random.normal(-5, 1)
        else:
            X[i] =  np.random.normal(5, 1)
    return X


def example2():
  #
  X = bimodal(1000)
  #Let's see how it looks
  plt.hist(X, bins=50)
  plt.ylabel('Frequency')
  plt.xlabel('Value')
  plt.title('Actual distribution')
  plt.show()
  print('mean:', np.mean(X))
  print('standard deviation:', np.std(X))
  mu = np.mean(X)
  sigma = np.std(X)
  N = np.random.normal(mu, sigma, 1000)
  plt.hist(N, bins=50)
  plt.ylabel('Frequency')
  plt.xlabel('Value');
  plt.title('Sample normal distribution')
  from statsmodels.stats.stattools import jarque_bera
  # Examine whether data are normally distributed using jarque-bera 
  # normality test
  jarque_bera(X)
  plt.show()
  return


# Sharpe ratio stuff
def sharpe_ratio(asset, riskfree):
  return np.mean(asset - riskfree)/np.std(asset - riskfree)


def sharpe1(show=True):
  start = datetime.datetime(2013, 1,1)
  end = datetime.datetime(2015, 1,1)
  # Use an ETF that tracks 3-month T-bills as our risk-free rate of return
  treasury_ret = pull_series('BIL', start, end).pct_change()[1:]
  pricing = pull_series('AMZN', start, end)
  returns = pricing.pct_change()[1:] # Get the returns on the asset
  #
  # Compute the running Sharpe ratio
  running_sharpe = [sharpe_ratio(returns[i-90:i], treasury_ret[i-90:i]) for i in range(90, len(returns))]
  #
  # Plot running Sharpe ratio up to 100 days before the end of the data set
  _, ax1 = plt.subplots()
  ax1.plot(range(90, len(returns)-100), running_sharpe[:-100]);
  ticks = ax1.get_xticks()
  ax1.set_xticklabels([pricing.index[i].date() for i in ticks[:-1]]) # Label x-axis with dates
  plt.xlabel('Date')
  plt.ylabel('Sharpe Ratio');
  plt.show()
  # Compute the mean and std of the running Sharpe ratios up to 100 days before the end
  mean_rs = np.mean(running_sharpe[:-100])
  std_rs = np.std(running_sharpe[:-100])
  # Plot running Sharpe ratio
  _, ax2 = plt.subplots()
  ticks = ax1.get_xticks()
  ax2.set_xticklabels([pricing.index[i].date() for i in ticks[:-1]]) # Label x-axis with dates
  ax2.plot(range(90, len(returns)), running_sharpe)
  # Plot its mean and the +/- 1 standard deviation lines
  ax2.axhline(mean_rs)
  ax2.axhline(mean_rs + std_rs, linestyle='--')
  ax2.axhline(mean_rs - std_rs, linestyle='--')
  # Indicate where we computed the mean and standard deviations
  # Everything after this is 'out of sample' which we are comparing with the estimated mean and std
  ax2.axvline(len(returns) - 100, color='pink');
  plt.xlabel('Date')
  plt.ylabel('Sharpe Ratio')
  plt.legend(['Sharpe Ratio', 'Mean', '+/- 1 Standard Deviation'])
  plt.show()
  print('Mean of running Sharpe ratio: %.5f' %mean_rs)
  print('std of running Sharpe ratio: %.5f' %std_rs)
  return


def moving_avg():
  """
  Create a moving avg with confidence limits.
  """
  # Example: Moving average
  # Load time series of prices
  start = datetime.datetime(2013, 1,1)
  end = datetime.datetime(2015, 1,1)
  pricing = pull_series('AMZN', start, end)
  # Compute the rolling mean for each day
  mu = pd.rolling_mean(pricing, window=90)
  # Plot pricing data
  _, ax1 = plt.subplots()
  ax1.plot(pricing) 
  ticks = ax1.get_xticks()
  ax1.set_xticklabels([pricing.index[i].date() for i in ticks[:-1]]) # Label x-axis with dates
  plt.ylabel('Price')
  plt.xlabel('Date')
  # Plot rolling mean
  ax1.plot(mu);
  plt.legend(['Price','Rolling Average']);
  print('Mean of rolling mean: %.5f' %np.mean(mu))
  print('std of rolling mean: %.5f' %np.std(mu))
  # Compute rolling standard deviation
  std = pd.rolling_std(pricing, window=90)
  # Plot rolling std
  _, ax2 = plt.subplots()
  ax2.plot(std)
  ax2.set_xticklabels([pricing.index[i].date() for i in ticks[:-1]]) # Label x-axis with dates
  plt.ylabel('Standard Deviation of Moving Average')
  plt.xlabel('Date')
  print('Mean of rolling std: %.5f' %np.mean(std))
  print('std of rolling std: %.5f' %np.std(std))
  # Plot original data
  _, ax3 = plt.subplots()
  ax3.plot(pricing)
  ax3.set_xticklabels([pricing.index[i].date() for i in ticks[:-1]]) # Label x-axis with dates
  # Plot Bollinger bands
  ax3.plot(mu)
  ax3.plot(mu + std)
  ax3.plot(mu - std);
  plt.ylabel('Price')
  plt.xlabel('Date')
  plt.legend(['Price', 'Moving Average', 'Moving Average +1 Std', 'Moving Average -1 Std'])
  plt.show()
  return













