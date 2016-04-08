# momentum.py
# momentum-based strategies
# based on notebook from https://www.quantopian.com/posts/quantopian-lecture-series-momentum-strategies



# Import libraries to find linear trend and plot data
from statsmodels import regression
import statsmodels.api as sm
import scipy.stats as stats
import scipy.spatial.distance as distance
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pairs_trading import *
from statsmodels.tsa.stattools import adfuller



# Load pricing data for an asset, 1 year
start = datetime.datetime(2014,1,1)
end = datetime.datetime(2015,1,1)
asset = pull_series('XLY', startdate=start, enddate=end)
dates = asset.index
# Plot the price of the asset over time
_, ax = plt.subplots()
ax.plot(asset)
ticks = ax.get_xticks()
ax.set_xticklabels([dates[i].date() for i in ticks[:-1]]) # Label x-axis with dates
# Find the line of best fit to illustrate the trend
X = np.arange(len(asset))
x = sm.add_constant(X) # Add a column of ones so that line can have a y-intercept
model = regression.linear_model.OLS(asset, x).fit()
a = model.params[0] # Get coefficients of line
b = model.params[1]
Y_hat = X * b + a
plt.plot(X, Y_hat, 'r', alpha=0.9);
plt.ylabel('Price')
plt.legend(['XLY', 'Trendline']);
plt.show()



# Load pricing data for an asset, NEXT 5 months - how did it do?
# Only for one asset -- should check this against S&P500 or fortune500 companies
start = datetime.datetime(2015,1,1)
end = datetime.datetime(2015,6,1)
asset = pull_series('XLY', startdate=start, enddate=end)
dates = asset.index
# Plot the price of the asset over time
_, ax = plt.subplots()
ax.plot(asset)
ticks = ax.get_xticks()
ax.set_xticklabels([dates[i].date() for i in ticks[:-1]]) # Label x-axis with dates
# Find the line of best fit to illustrate the trend
X = np.arange(len(asset))
x = sm.add_constant(X) # Add a column of ones so that line can have a y-intercept
model = regression.linear_model.OLS(asset, x).fit()
a = model.params[0] # Get coefficients of line
b = model.params[1]
Y_hat = X * b + a
plt.plot(X, Y_hat, 'r', alpha=0.9);
plt.ylabel('Price')
plt.legend(['XLY', 'Trendline']);
plt.show()



# Noise auto-correlation -- no momentum
def generate_autocorrelated_values(N):
  X = np.zeros(N)
  for i in range(N-1):
    X[i+1] = X[i] + np.random.normal(0, 1)
  return X

for i in range(10):
  X = generate_autocorrelated_values(100)
  plt.plot(X)
plt.xlabel('$t$')
plt.ylabel('$X_t$');
plt.show()



# Incorporate some past return data as well (momentum)
def generate_autocorrelated_values(N):
  X = np.zeros(N)
  for i in range(1, N-1):
    # Do the past returns 'look good' to investors
    past_returns = X[i] - X[i-1]
    # Investors hypothesize that future returns will be equal to past returns and buy at that price
    X[i+1] = X[i] + past_returns + np.random.normal(0, 1)
  return X
for i in range(10):
  X = generate_autocorrelated_values(10)
  plt.plot(X)
plt.xlabel('$t$')
plt.ylabel('$X_t$');
plt.show()



X1 = generate_autocorrelated_values(100)
X2 = np.random.normal(0, 1, 100)

# Compute the p-value of the Dickey-Fuller statistic to test the null hypothesis that yw has a unit root
print('X1')
_, pvalue, _, _, _, _ = adfuller(X1)
if pvalue > 0.05:
  print('We cannot reject the null hypothesis that the series has a unit root.')
else:
  print('We reject the null hypothesis that the series has a unit root.')
print('X2')
_, pvalue, _, _, _, _ = adfuller(X2)
if pvalue > 0.05:
  print('We cannot reject the null hypothesis that the series has a unit root.')
else:
  print('We reject the null hypothesis that the series has a unit root.')



############################################################################
# Momentum vs mean reversion

# Time frame is important:

# Load pricing data for an asset
start = datetime.datetime(2014,1,1)
end = datetime.datetime(2015,1,1)
asset = pull_series('XLY', startdate=start, enddate=end)
dates = asset.index
# Plot the price of the asset over time
_, ax = plt.subplots()
ax.plot(asset)
ticks = ax.get_xticks()
ax.set_xticklabels([dates[i].date() for i in ticks[:-1]]) # Label x-axis with dates
# Find the line of best fit to illustrate the trend
X = np.arange(len(asset))
x = sm.add_constant(X) # Add a column of ones so that line can have a y-intercept
model = regression.linear_model.OLS(asset, x).fit()
a = model.params[0] # Get coefficients of line
b = model.params[1]
Y_hat = X * b + a
plt.plot(X, Y_hat, 'r', alpha=0.9);
plt.ylabel('Price')
plt.legend(['XLY', 'Trendline']);
plt.show()



plt.plot(asset - Y_hat)
plt.hlines(np.mean(asset - Y_hat), 0, 300, colors='r')
plt.hlines(np.std(asset - Y_hat), 0, 300, colors='r', linestyles='dashed')
plt.hlines(-np.std(asset - Y_hat), 0, 300, colors='r', linestyles='dashed')
plt.xlabel('Time')
plt.ylabel('Dollar Difference');
plt.show()



############################################################################

# Measuring momentum

# Load some asset data
start = '2013-01-01'
end = '2015-01-01'
assets = sorted(['STX', 'WDC', 'CBI', 'JEC', 'VMC', 'PG', 'AAPL', 'PEP', 'AON', 'DAL'])
data = get_pricing(assets, start_date=start, end_date=end).loc['price', :, :]

# Plot the prices just for fun
data.plot(figsize=(10,7), color=['r', 'g', 'b', 'k', 'c', 'm', 'orange',
                                  'chartreuse', 'slateblue', 'silver'])
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.ylabel('Price')
plt.xlabel('Time');



data.columns
asset = data.iloc[:, 8]
asset.plot()
plt.ylabel('Price')



short_mavg = pd.rolling_mean(asset, 30)
long_mavg = pd.rolling_mean(asset, 200)
asset.plot()
short_mavg.plot()
long_mavg.plot()
plt.ylabel('Price')



# Ribbon stuff! #########################
#
asset.plot(alpha = 0.5)
#
rolling_means = {}
#
for i in np.linspace(10, 100, 10):
    X = pd.rolling_mean(asset, i)
    rolling_means[i] = X
    X.plot(alpha = 0.7)
#    
rolling_means = pd.DataFrame(rolling_means).dropna()

# information about ribbon shape

# distance metrics; hamming distance
scores = pd.Series(index=asset.index)
for date in rolling_means.index:
    mavg_values = rolling_means.loc[date]
    ranking = stats.rankdata(mavg_values.values)
    d = distance.hamming(ranking, range(1, 11))
    scores[date] = d
    # Normalize the  score
(10 * scores).plot();
asset.plot()
plt.legend(['Signal', 'Asset Price']);



# correlation metric; spearman rank correlation
scores = pd.Series(index=asset.index)
for date in rolling_means.index:
    mavg_values = rolling_means.loc[date]
    ranking = stats.rankdata(mavg_values.values)
    _, d = stats.spearmanr(ranking, range(1, 11))
    scores[date] = d
# Normalize the  score
(10 * scores).plot();
asset.plot()
plt.legend(['Signal', 'Asset Price']);



# measure thickness
scores = pd.Series(index=asset.index)
for date in rolling_means.index:
    mavg_values = rolling_means.loc[date]
    d = np.max(mavg_values) - np.min(mavg_values)
    scores[date] = d
    # Normalize the  score
(10 * scores).plot();
asset.plot()
plt.legend(['Signal', 'Asset Price']);



#######################################################################
# Measuring momentum from physics

k = 30
start = '2014-01-01'
end = '2015-01-01'
pricing = get_pricing('PEP', fields='price', start_date=start, end_date=end)
fundamentals = init_fundamentals()
num_shares = get_fundamentals(query(fundamentals.earnings_report.basic_average_shares,)
                              .filter(fundamentals.company_reference.primary_symbol == 'PEP',), end)
x = np.log(pricing)
v = x.diff()
m = get_pricing('PEP', fields='volume', start_date=start, end_date=end)/num_shares.values[0,0]
#
p0 = pd.rolling_sum(v, k)
p1 = pd.rolling_sum(m*v, k)
p2 = p1/pd.rolling_sum(m, k)
p3 = pd.rolling_mean(v, k)/pd.rolling_std(v, k)



f, (ax1, ax2) = plt.subplots(2,1)
ax1.plot(p0)
ax1.plot(p1)
ax1.plot(p2)
ax1.plot(p3)
ax1.set_title('Momentum of PEP')
ax1.legend(['p(0)', 'p(1)', 'p(2)', 'p(3)'], bbox_to_anchor=(1.1, 1))
#
ax2.plot(p0)
ax2.plot(p1)
ax2.plot(p2)
ax2.plot(p3)
ax2.axis([0, 300, -0.005, 0.005])
ax2.set_xlabel('Time');



# Implementing physics measures
def get_p(prices, m, d, k):
  """ 
  Returns the dth-degree rolling momentum of data using lookback window length k.
  """
  x = np.log(prices)
  v = x.diff()
  m = np.array(m)
  #
  if d == 0:
    return pd.rolling_sum(v, k)
  elif d == 1:
    return pd.rolling_sum(m*v, k)
  elif d == 2:
    return pd.rolling_sum(m*v, k)/pd.rolling_sum(m, k)
  elif d == 3:
    return pd.rolling_mean(v, k)/pd.rolling_std(v, k)
  
def backtest_get_p(prices, m, d):
  """ Returns the dth-degree rolling momentum of data"""
  v = np.diff(np.log(prices))
  m = np.array(m)
  #
  if d == 0:
    return np.sum(v)
  elif d == 1:
    return np.sum(m*v)
  elif d == 2:
    return np.sum(m*v)/np.sum(m)
  elif d == 3:
    return np.mean(v)/np.std(v)
