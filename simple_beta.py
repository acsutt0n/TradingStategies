# simple_beta.py -- given a stock and a time frame (in days), this 
#                   calculates the beta
# usage: python simple_beta.py symbol(1) time_frame(2) exchange(2)
# Input:   1. Can either be a single symbol or a text file with one
#             symbol per line
#          2. Can either be a single int (i.e. 100 = 100 days) or
#             a text file with each line reflecting a time frame for
#             each corresponding line of symbols
#          3. Can either be a single exchange (i.e.: SPY, DIA) or
#             a file with each line reflecting each desired exchange
# Output:  Betas will be saved as a pickled dictionary or something

import numpy as np
from stockquote import *


