#!/usr/bin/env python3
"""
setting index to a colummn of the dataframe
using DS4A knowledge
"""
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

df = df.set_index('Timestamp')

print(df.tail())
