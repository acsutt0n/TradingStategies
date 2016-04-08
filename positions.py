# report portfolio progress to email
# usage: python positions.py positionFile emailAddrs (optional)
#        emailAddrs is a text file with a recipient email address
#        on each line (and nothing else)

import numpy as np
from stockquote import *


def load_positions(fileName):
  positions = {}
  L = ('symbol','purprice','purvalue','currprice','gainloss')
  
  with open(fileName, 'r') as fIn:
    lineNum = 0 
    for line in fIn:
      splitLine = line.split(None)
      if splitLine[0] == 'symbol':
        print('Found title line.')
      elif splitLine[0] is not None:
        positions[splitLine[0]] = dict.fromkeys(L,None)
        positions[splitLine[0]]['symbol'] = splitLine[0]
        for i in range(len(splitLine[1:])):
          if splitLine[i+1] != '_':
            positions[splitLine[0]][L[i+1]] = float(splitLine[i+1])
        print('Loaded stock %s.' %splitLine[0])
      lineNum = lineNum + 1
  
  return positions



def update_positions(positions):
  for s in positions.keys():
    Q = get(s)
    currprice = float(Q[1]['price_last'])
    positions[s]['currprice'] = currprice
    numshares = positions[s]['purvalue']/positions[s]['purprice']
    currvalue = numshares * currprice
    positions[s]['currvalue'] = currvalue
    gain = currvalue - positions[s]['purvalue'] 
    positions[s]['gainloss'] = gain
    print('Updated %s. ' %s)
  return positions
    
    



def simple_analysis(positions):
  market_val, original_val, tot_gain  = 0, 0, 0
  for s in positions.keys():
    currmarval = positions[s]['currvalue']
    if currmarval is not None:
      market_val = market_val + currmarval
    currorigval = positions[s]['purvalue']
    if currorigval is not None:
      original_val = original_val + currorigval
    currchange = positions[s]['gainloss']
    if currchange is not None:
      tot_gain = tot_gain + currchange
  # print report
  print('%i stocks owned. Total invested: %.2f. Market value: %.2f. Net gain: %.2f' 
        %(len(positions), original_val, market_val, market_val-original_val))
  for s in positions.keys():
    labels, vals = [], []
    for i in positions[s].keys():
      if positions[s][i] is not None:
        labels.append(i)
        vals.append(positions[s][i])
    labels = [x for (y,x) in sorted(zip(vals, labels))]
    vals.sort()
    vals.reverse()
    labels.reverse()
    for x,y in zip(labels, vals):
      try:
        print('%s: %.2f' %(x,y))
      except:
        print('%s: %s' %(x,y))
  
  return



def txt_analysis(positions):
  market_val, original_val, tot_gain  = 0, 0, 0
  for s in positions.keys():
    currmarval = positions[s]['currvalue']
    if currmarval is not None:
      market_val = market_val + currmarval
    currorigval = positions[s]['purvalue']
    if currorigval is not None:
      original_val = original_val + currorigval
    currchange = positions[s]['gainloss']
    if currchange is not None:
      tot_gain = tot_gain + currchange
  # print report
  print('Writing temporary file...')
  with open('temp_report.txt','w') as report:
    report.write('%i stocks owned. Total invested: %.2f. Market value: %.2f. Net gain: %.2f' 
          %(len(positions), original_val, market_val, market_val-original_val))
    for s in positions.keys():
      labels, vals = [], []
      for i in positions[s].keys():
        if positions[s][i] is not None:
          labels.append(i)
          vals.append(positions[s][i])
      labels = [x for (y,x) in sorted(zip(vals, labels))]
      vals.sort()
      vals.reverse()
      labels.reverse()
      for x,y in zip(labels, vals):
        try:
          report.write('%s: %.2f' %(x,y))
        except:
          report.write('%s: %s' %(x,y))
      
  return




def email_stockControl(portfolioFile, emailAddrsFile):
  positions = load_positions(portfolioFile)
  positions = update_positions(positions)
  txt_analysis(positions)
  




def simple_stockControl(fileName):
  positions = load_positions(fileName)
  positions = update_positions(positions)
  simple_analysis(positions)




if __name__ == "__main__":
  import sys, os
  arguments = sys.argv
  if len(arguments) < 2:
    print('Need portfolio folder')
  elif len(arguments) == 2:
    portFile = arguments[1]
    simple_stockControl(portFile)
  elif len(arguments) == 3:
    portFile = arguments[1]
    emailAddrs = arguments[2]
    email_stockControlstockControl(portFile, emailAddrs)
  
