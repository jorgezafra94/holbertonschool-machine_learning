#!/usr/bin/env python3
"""
complete data features
"""

from datetime import date
import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')
df.rename(columns={'Timestamp': 'Date'}, inplace=True)
df.drop(['Weighted_Price'], axis=1, inplace=True)

df['Date'] = pd.to_datetime(df['Date'], unit='s')
df = df[df['Date'] >= '2017-01-01']

df = df.set_index('Date')

df['Close'].fillna(method='ffill', inplace=True)

df = df.fillna({'Open': df['Close'].shift(1, fill_value=0),
                'High': df['Close'].shift(1, fill_value=0),
                'Low': df['Close'].shift(1, fill_value=0)})

df = df.fillna({'Volume_(BTC)': 0,
                'Volume_(Currency)': 0})


df = df.resample('D').agg({'Open': 'first', 'High': 'max',
                           'Low': 'min', 'Close': 'last',
                           'Volume_(BTC)': 'sum',
                           'Volume_(Currency)': 'sum'})

df.fillna(method='ffill', inplace=True)
df.plot()
plt.show()
