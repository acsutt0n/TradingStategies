# beta.py -- analyzes the beta of a portfolio stored as needed for
#            positions.py
# usage: python2 beta.py positions.txt
# currently only runs on python2

import numpy as np
from stockquote import *
from positions import *
from datetime import date
import sys


# global variable today
today = date.today()
d, m, y = str(today.day), str(today.month), str(today.year)
if len(d) != 2:
  d = '0'+ d
if len(m) != 2:
  m = '0' + m
today = ''.join([y,m,d])



def get_positions(filename):
  positions = load_positions(filename)
  positions = update_positions(positions)
  return positions


def individual_beta(sym, market='SPY', start='20100101'): # using 5 year backtest
  # get closing prices
  sym_closep, market_closep = [], []
  p, m = historical_quotes(sym,start, today), historical_quotes(market, start, today)
  for i in range(len(p)):
    sym_closep.append(float(p[i]['Adj Close']))
    market_closep.append(float(m[i]['Adj Close']))
    
  # actual beta calculation
  sym_ret, market_ret = [], []
  for cur in range(len(sym_closep)-1):               # for each datum
    # return = curr_price / prev_price - 1
    sym_ret.append((sym_closep[cur+1] / sym_closep[cur]) - 1)
    market_ret.append((market_closep[cur+1] / market_closep[cur]) - 1)
    # beta = covariance(sym_ret, market_ret) / var(market_ret)
  return np.cov(sym_ret, market_ret)[0,1]/np.var(market_ret)


def positions_beta(positions):
  vals, betas = [], []
  for s in positions.keys():
    positions[s]['beta'] = individual_beta(s)
    betas.append(positions[s]['beta'])
    vals.append(positions[s]['currvalue'])
    print('Beta for %s: %.4f' %(s,positions[s]['beta']))
  tot_val = sum(vals)
  total_beta = 0
  if len(betas) == len(vals):
    for i in range(len(betas)):
      total_beta = total_beta + vals[i]/tot_val*betas[i]
  print('Total beta for portfolio: %.4f' %total_beta)
  return total_beta



if __name__ == '__main__':
  positions = get_positions(sys.argv[1])
  tot_beta = positions_beta(positions)












