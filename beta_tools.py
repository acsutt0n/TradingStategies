# Simple beta tools from https://www.quantopian.com/research/notebooks/Cloned%20from%20%22Quantopian%20Lecture%20Series%3A%20The%20Art%20of%20Not%20Following%20the%20Market%22.ipynb

from pairs_trading import *


import numpy as np
from statsmodels import regression
import statsmodels.api as sm
import matplotlib.pyplot as plt
import math



def linreg(x,y):
  # Simple linear regression
  x = sm.add_constant(x)
  model = regression.linear_model.OLS(y,x).fit()
  x = x[:,1]
  return model.params[0], model.params[1]
  



def simple_beta(syms, bench='spy'):
  """
  Return simple alpha and beta based on lin regression
  """
  if type(syms) is not list:
    syms = [syms]
  bench = pull_series(bench)
  series = [pull_series(i) for i in syms]
  r_b = bench.pct_change()[1:]
  for s in range(len(series)):
    r_a = series[s].pct_change()[1:]
    alpha, beta = linreg(r_b.values, r_a.values)
    print('For %s: alpha: %.5f, beta: %.5f' %(syms[s], alpha, beta))
  return


  
  
