import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pairs_trading import *
import datetime


def simple_correlation():
  # Simple shit
  X = np.random.rand(50)
  Y = 2 * X + np.random.normal(0, 0.1, 50)
  print('Baseline covariance: %.5f' %np.cov(X, Y)[0, 1])
  #
  X = np.random.rand(50)
  Y = 2 * X + 4
  print('Covariance of X and Y: \n' + str(np.cov(X, Y)))
  print('Correlation of X and Y: \n' + str(np.corrcoef(X, Y)))
  #
  cov_matrix = np.cov(X, Y)
  # We need to manually set the degrees of freedom on X to 1, as numpy defaults to 0 for variance
  # This is usually fine, but will result in a slight mismatch as np.cov defaults to 1
  error = cov_matrix[0, 0] - X.var(ddof=1)
  print('error: ' + str(error))
  return



def compare_correlations(X=None, Y=None):
  # Do a simple correlation comparison between custom, numpy as pandas
  if X is None:
    X = np.random.rand(50)
  if Y is None:
    Y = np.random.rand(50)
  fig1 = plt.figure()
  ax1 = fig1.add_subplot(221)
  ax1.scatter(X,Y)
  ax1.set_xlabel('X Value')
  ax1.set_ylabel('Y Value')
  # taking the relevant value from the matrix returned by np.cov
  print('Correlation: ' + str(np.cov(X,Y)[0,1]/(np.std(X)*np.std(Y))))
  # Let's also use the builtin correlation function
  print('Built-in Correlation: ' + str(np.corrcoef(X, Y)[0, 1]))
  #
  X = np.random.rand(50)
  Y = X + np.random.normal(0, 0.1, 50)
  ax11 = fig1.add_subplot(222)
  ax11.scatter(X,Y)
  ax11.set_xlabel('X Value')
  ax11.set_ylabel('Y Value')
  print('Correlation: ' + str(np.corrcoef(X, Y)[0, 1]))
  #
  X = np.random.rand(50)
  Y = X + np.random.normal(0, .2, 50)
  ax2 = fig1.add_subplot(223)
  ax2.scatter(X,Y)
  ax2.set_xlabel('X Value')
  ax2.set_ylabel('Y Value')
  print('Correlation: ' + str(np.corrcoef(X, Y)[0, 1]))
  #
  X = np.random.rand(50)
  Y = -X + np.random.normal(0, .1, 50)
  ax3 = fig1.add_subplot(224)
  ax3.scatter(X,Y)
  ax3.set_xlabel('X Value')
  ax3.set_ylabel('Y Value')
  print('Correlation: ' + str(np.corrcoef(X, Y)[0, 1]))
  plt.show()
  return



# 'How is this useful in finance?'
def correlation(sym1='LRCX', sym2='AAPL', bench='SPY'):
  # Pull the pricing data for our two stocks and S&P 500
  start = datetime.datetime(2013, 1, 1) # '2013-01-01'
  # end = '2015-01-01' # Fuck that I make my own end date
  bench = pull_series(bench, start)
  a1 = pull_series(sym1, start)
  a2 = pull_series(sym2, start)
  # Do a simple plot
  fig = plt.figure()
  ax1 = fig.add_subplot(211)
  ax1.scatter(a1,a2)
  ax1.set_xlabel(sym1)
  ax1.set_ylabel(sym2)
  ax1.set_title('Stock prices from %i-%i-%i to today' %(start.day,
                                                        start.month,
                                                        start.year))
  print("Correlation coefficients")
  print("%s and %s: %.5f" %(sym1, sym2, np.corrcoef(a1,a2)[0,1]))
  print("%s and %s: %.5f" %(sym1, bench, np.corrcoef(a1,bench)[0,1]))
  print("%s and %s: %.5f" %(sym2, bench, np.corrcoef(bench,a2)[0,1]))
  # Get rolling correlation from pandas
  rolling_correlation = pd.rolling_corr(a1, a2, 60)
  ax2 = fig.add_subplot(222)
  ax2.plot(rolling_correlation)
  ax2.set_xlabel('Day')
  ax2.set_ylabel('60-day Rolling Correlation')
  # Get raw correlation
  X = np.random.rand(100)
  Y = X + np.random.poisson(size=100)
  ax3 = fig.add_subplot(223)
  ax3.scatter(X, Y)
  print('X-Y correlation coefficient: %.5f' %np.corrcoef(X, Y)[0, 1])
  plt.show()
  return np.corrcoef(X, Y)[0,1]
  

