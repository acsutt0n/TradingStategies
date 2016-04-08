# pairs_trading.py -- Returns statistics on prospective pairs strategies
"""
Best if used in interactively in ipython, but can be called
from command line along with a text file of symbols <or> a string of
symbols. Will automatically include SPY if no other major index is
specified (DIA, SPY).
-- Symbol file must end in .txt.
-- All of this is inspired by (and some is even copied from) a Quantopian
IPython notebook on pairs trading. 
"""

import numpy as np
import pandas as pd
import pandas.io.data as web
import matplotlib.pyplot as plt
import seaborn
import statsmodels
from statsmodels.tsa.stattools import coint
import datetime
import sys



def simple_comparison(X,Y, show=True):
  """
  Compares correlation and cointegration for X and Y. Can be Y! dataframe
  or n x 1 arrays.
  """
  if type(X) is pd.DataFrame:
    x = X['Adj Close']
  else:
    x = X
  if type(Y) is pd.DataFrame:
    y = Y['Adj Close']
  else:
    y = Y
  #x = pd.Series(np.cumsum(x), name='x')
  #y = pd.Series(np.cumsum(y), name='y')
  x = pd.Series(x, name='x')
  y = pd.Series(y, name='y')
  x, y = return_timelocked(x,y)
  fig = plt.figure()
  ax1 = fig.add_subplot(121)
  ax1.plot(pd.concat([x,y], axis=1))
  ax2 = fig.add_subplot(122)
  ax2.plot(y-x)
  ax2.axhline((y-x).mean(), color='red', linestyle='--')
  score, pvalue, _ = coint(x,y)
  ax1.set_title('Correlation: %.5f' %x.corr(y))
  ax2.set_title('Cointegration P: %.5f' %pvalue)
  if show:
    plt.show()
    return
  return ax1, ax2



def pull_series(symbol_list, startdate=None, enddate=None):
  """
  Enter a symbol or a list of symbols to return a list of Adj Close data.
  Start date is format: datetime.datetime, or 20140101
  """
  def date(dat):
    if type(dat) is datetime.datetime:
      return dat
    elif type(startdate) is list:
      dat = datetime.datetime(dat[0], dat[1], dat[2])
      return dat
    elif type(startdate) is int:
      dat = datetime.datetime(int(str(dat)[:4]),
                                int(str(dat)[4:6]),
                                int(str(dat)[6:]))
      return dat
    else:
      print('do not recognize startdate format!')
      return None
  def pull_one(sym, startdate, enddate):
    if startdate is None:
      start = datetime.datetime(2014,1,1) # choose a default time
    else:
      start = date(startdate)
    if enddate is None:
      today = datetime.date.today()
      end = datetime.datetime(today.year, today.month, today.day)
    else:
      end = date(enddate)
    security = web.DataReader(sym, 'yahoo', start, end)
    beh = security['Adj Close']
    beh.name = sym
    return beh
  #
  if type(symbol_list) is str:
    return pull_one(symbol_list, startdate, enddate)
  if type(symbol_list) is list:
    SS = pd.Series([pull_one(i, startdate, enddate) for i in symbol_list],
                     index=symbol_list)
    for s in range(len(SS)):
      SS[s].name = symbol_list[s]
    return SS
  else:
    print('options for symbols are list of strings or string')
    return None
  



def gen_randdata(n=2):
  # Generate random data for correlation and cointegration tests.
  # First, cointegrated and correlated
  X_returns = np.random.normal(0, 1, 100) # Generate the daily returns
  # sum them and shift all the prices up into a reasonable range
  X = pd.Series(np.cumsum(X_returns), name='X') + 50
  some_noise = np.random.normal(0, 1, 100)
  Y = X + 5 + some_noise
  Y.name = 'Y'
  fig = plt.figure()
  ax1 = fig.add_subplot(321)
  ax1.plot(pd.concat([X, Y], axis=1))
  ax2 = fig.add_subplot(322)
  ax2.plot(Y-X)
  ax2.axhline((Y-X).mean(), color='red', linestyle='--')
  score, pvalue, _ = coint(X,Y)
  ax1.set_title('Correlation: %.5f' %X.corr(Y))
  ax2.set_title('Cointegration P: %.5f' %pvalue)
  # Next, correlated but not cointegrated
  X_returns = np.random.normal(1, 1, 100)
  Y_returns = np.random.normal(2, 1, 100)
  X_diverging = pd.Series(np.cumsum(X_returns), name='X_diverging')
  Y_diverging = pd.Series(np.cumsum(Y_returns), name='Y_diverging')
  ax3 = fig.add_subplot(323)
  ax4 = fig.add_subplot(324)
  ax3.plot(pd.concat([X_diverging, Y_diverging], axis=1))
  ax4.plot(Y_diverging-X_diverging)
  ax4.axhline((Y_diverging-X_diverging).mean(), color='red', linestyle='--')
  score, pvalue, _ = coint(X_diverging, Y_diverging)
  ax3.set_title('Correlation: %.5f' %X_diverging.corr(Y_diverging))
  ax4.set_title('Cointegration P: %.5f' %pvalue)
  # cointegration without correlation ("nominal convergence")
  X_coint = pd.Series(np.random.normal(0, 1, 1000), name='X_coint') + 20
  Y_coint = X_coint.copy()
  for i in range(10):
    if i % 2 == 0:
      Y_coint[i*100:(i+1)*100] = 10
    else:
      Y_coint[i*100:(i+1)*100] = 30
  ax5 = fig.add_subplot(325)
  ax5.plot(X_coint)
  ax5.plot(Y_coint)
  ax5.set_ylim([0,40])
  ax6 = fig.add_subplot(326)
  ax6.plot(Y_coint-X_coint)
  ax6.axhline((Y_coint-X_coint).mean(), color='red', linestyle='--')
  score, pvalue, _ = coint(X_coint, Y_coint)
  ax5.set_title('Correlation: %.5f' %X_coint.corr(Y_coint))
  ax6.set_title('Cointegration P: %.5f' %pvalue)
  plt.show()
  return


 
def return_timelocked(S1, S2):
  # Checks the dates of the pd.Series objects and returns common dates
  # First, make sure they end on the same date
  end1, end2 = S1.keys()[-1], S2.keys()[-1]
  end1 = datetime.datetime(end1.year, end1.month, end1.day)
  end2 = datetime.datetime(end2.year, end2.month, end2.day)
  # make sure the new start dates are the same
  def check_start(S1, S2):
    start1, start2 = S1.keys()[0], S2.keys()[0]
    start1 = datetime.datetime(start1.year, start1.month, start1.day)
    start2 = datetime.datetime(start2.year, start2.month, start2.day)
    if start1 != start2:
      return False
    else:
      return True
  if end1 != end2:
    print('Series 1 ends on %i %i, %i but Series 2 ends on %i %i, %i'
          %(end1.day, end1.month, end1.year,
            end2.day, end2.month, end2.year))
    return None, None
  if len(S1) == len(S2):
    return S1, S2
  elif len(S1) > len(S2):
    newlen = len(S2)
    S1 =  S1[-newlen:]
    if check_start(S1,S2):
      return S1, S2
    else:
      print('End dates fixed, but bad start dates somehow')
  elif len(S2) > len(S1):
    newlen = len(S1)
    S2 = S2[-newlen:]
    if check_start(S1,S2):
      return S1, S2
    else:
      print('End dates fixed, but bad start dates somehow')
  else:
    print('Something screwy going on')
  return None, None
  


def find_cointegrated_pairs(securities):
  # Quantopian function to compare pairs of securities; revised to use
  # pd.Series instead of pd.DataPanel
  if type(securities) is not pd.Series:
    print('type is %s but should be pd.Series' %type(securities))
    return
  n = len(securities.index)
  score_matrix = np.zeros((n, n))
  pvalue_matrix = np.ones((n, n))
  keys = securities.index
  pairs = []
  for i in range(n):
    for j in range(i+1, n):
      S1 = securities[i]
      S2 = securities[j]
      S1, S2 = return_timelocked(S1, S2)
      result = coint(S1, S2)
      score = result[0]
      pvalue = result[1]
      score_matrix[i, j] = score
      pvalue_matrix[i, j] = pvalue
      if pvalue < 0.05:
        pairs.append((keys[i], keys[j]))
  return score_matrix, pvalue_matrix, pairs



def bulk_analysis(symbol_list=None, startdate=None, show=True):
  # Runs a sample pairs comparison with some solar stocks if no input.
  if symbol_list is None:
    symbol_list = ['ABGB', 'ASTI', 'CSUN', 'DQ', 'FSLR','SPY']
  securities = pull_series(symbol_list, startdate)
  scores, pvalues, pairs = find_cointegrated_pairs(securities)
  seaborn.heatmap(pvalues, xticklabels=symbol_list,
                  yticklabels=symbol_list, cmap='RdYlGn_r',
                  mask=(pvalues>=0.95))
  if len(pairs) > 1:
    print(pairs)
    if show:
      plt.show()
      return [securities[symbol_list.index(i)] for i in pairs]
  if show:
    plt.show()
  return pairs



def zscore(S1, S2):
  # S1 and S2 should be series of only adjusted close values.
  
  diff_series = S1 - S2
  diff_series.name = 'diff'
  zseries = (diff_series - diff_series.mean()) / np.std(diff_series)
  fig = plt.figure()
  ax1 = fig.add_subplot(221)
  ax1.plot(zseries)
  ax1.axhline(zseries.mean(), color='black')
  ax1.axhline(1.0, color='red', linestyle='--')
  ax1.axhline(-1.0, color='green', linestyle='--')
  ax1.set_title('Z-scored difference %s-%s' %(S1.name, S2.name))
  """
  Spread = S1-S2. Strategy is 'long' the spread (buy S1, short S2) when
  Zseries is < -1.0 (indicates significantly lower S1 and greater S2 and
  assumes a correction). When Zseries is > 1.0, buy S2 and short S1. 
  """
  # Calculate some moving averages, not on z-scores but on raw data
  # 10- and 60-day moving average
  diff_mavg10 = pd.rolling_mean(diff_series, window=10)
  diff_mavg10.name = 'diff 10d mavg'
  diff_mavg60 = pd.rolling_mean(diff_series, window=60)
  diff_mavg60.name = 'diff 60d mavg'
  ax2 = fig.add_subplot(222)
  ax2.plot(pd.concat([diff_mavg60, diff_mavg10], axis=1))
  ax2.set_title('(Raw) Moving averages of difference %s-%s' %(S1.name, S2.name))
  # Now do moving averages with zscored mavg data
  ax3 = fig.add_subplot(223)
  std_60 = pd.rolling_std(diff_series, window=60)
  std_60.name = 'std 60d'
  zscore_60_10 = (diff_mavg10 - diff_mavg60) / std_60 # zscore for each day
  zscore_60_10.name = 'z-score'
  ax3.plot(zscore_60_10)
  ax3.axhline(0, color='black')
  ax3.axhline(1.0, color='red', linestyle='--')
  ax3.axhline(-1.0, color='green', linestyle='--')
  ax3.set_title('Daily z-score of moving averages')
  ax4 = fig.add_subplot(224)
  ax4.plot(pd.concat([S1, S2], axis=1))
  ax4.set_title('Raw stock values')
  plt.show()
  print('Current zscore for 60d/10d moving avg (%s-%s): %.5f' 
         %(S1.name, S2.name,zscore_60_10[-1]))
  if zscore_60_10[-1] > 1.0:
    print('Buy option %s, short option %s' %(S2.name, S1.name))
  elif zscore_60_10[-1] < -1.0:
    print('Buy option %s, short option %s' %(S1.name, S2.name))
  return



def run_pairwise(symbol_list, startdate=None):
  """
  By default, this function runs the bulk cointegration analysis and
  then the pairwise on significantly co-integrated pairs, if any.
  """
  pairs = bulk_analysis(symbol_list, startdate, show=False)
  if len(pairs) > 1:
    for i in range(len(pairs)):
      # print(pairs[i])
      S1, S2 = pull_series([pairs[i][0], pairs[i][1]])
      zscore(S1, S2)
  else:
    pairs = bulk_analysis(symbol_list, startdate, show=True)
    score, pvalue, _ = find_cointegrated_pairs(pull_series(symbol_list))
    print('pvalue matrix is:')
    print(pvalue)
  return



def parse_symbol_file(fil):
  # Reads \n-separated symbols into a list
  symbol_list = []
  with open(fil, 'r') as fIn:
    for line in fIn:
      try:
        splitline = line.split(None)
        symbol_list.append(splitline[0])
      except:
        pass
  run_pairwise(symbol_list)
  return
















##########################################################################
if __name__ == '__main__':
  args = sys.argv
  if len(args) == 2:
    if args[1].split('.')[-1] == 'txt':
      parse_symbol_file(args[1])
    else:
      print('Bad number of symbols! Must be 2!')
  elif len(args) > 2:
    symbol_list = [i for i in args[1:]]
    run_pairwise(symbol_list)
  else:
    print('Bad number of symbols! Must be 2 or a symbol file!')

