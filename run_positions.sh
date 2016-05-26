#!/bin/bash
# run some positions stuff every day at close of trading

python2 ~/Documents/stocks/positions.py ~/Documents/stocks/positions.txt ~/Documents/stocks/email.txt ~/Documents/stocks/pfile.txt
echo "Stock report sent."
