#!/usr/bin/env python3
"""
filling the null values
"""

import pandas as pd

from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

cols = df.columns.to_list()
cols.remove('Weighted_Price')

df = df[cols]

df['Close'].fillna(method='ffill', inplace=True)

df = df.fillna({'Open': df['Close'],
                'High': df['Close'],
                'Low': df['Close']})

df = df.fillna({'Volume_(BTC)': 0,
                'Volume_(Currency)': 0})

print(df.head())
print(df.tail())
