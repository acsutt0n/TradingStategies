# send an email with contents from a txt file in format below with python
# called: python send_email.py email.txt pfile.txt
# email.txt is as below; pfile.txt is simply the email psw for fromaddr
"""
# Message must obey this template::
To: toaddress
From: fromaddress
Msg: Message as long as you want it
"""

import smtplib

def load_emailfile(emailFile):
  with open(emailFile, 'r') as fIn:
    for line in fIn:
      splitLine = line.split(None)
      if splitLine[0] == 'To:':
        if len(splitLine) > 2:
          print('To: line should be only: To: toaddress')
        toaddr = splitLine[1]
      elif splitLine[0] == 'From:':
        if len(splitLine) > 2:
          print('From: line should be only: From: fromaddress')
        fromaddr = splitLine[1]
      elif splitLine[0] == 'Msg:':
        msg = ' '.join(splitLine[1:])
  return toaddr, fromaddr, msg


def load_pfile(pfile):
  with open(pfile, 'r') as pIn:
    for line in pIn:
      splitLine = line.split(None)
      p = splitLine[0]
  return p


def send_email(toaddr, fromaddr, p, msg):
  print('Connecting to gmail server...')
  server = smtplib.SMTP('smtp.gmail.com:587')
  server.starttls()
  try:
    server.login(fromaddr, p)
  except:
    print('Bad password. Login name must be same as From: fromaddr')
  server.sendmail(fromaddr, toaddr, msg)
  print('Email sent.')
  server.quit()
  return



def emailControl(emailFile, pfile):
  toaddr, fromaddr, msg = load_emailfile(emailFile)
  p = load_pfile(pfile)
  send_email(toaddr, fromaddr, p, msg)
  return


####################################################################
if __name__ == "__main__":
  import sys, os
  args = sys.argv
  if len(args) != 3:
    print('Need 3 arguments: python send_email.py email.txt pfile.txt')
  else:
    emailFile = args[1]
    pfile = args[2]
    emailControl(emailFile, pfile)
