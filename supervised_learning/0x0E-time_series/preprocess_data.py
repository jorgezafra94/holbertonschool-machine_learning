#!/usr/bin/env python3
"""
data preprocessing
"""
import pandas as pd


def preprocess():
    """
    Here we realize the entire preprocessing of the data
    data-cleaning
    data-resample
    """

    # coinbase = './coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv'
    bitstamp = './bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv'

    # df = pd.read_csv(coinbase)
    df = pd.read_csv(bitstamp)

    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
    df = df[df['Timestamp'] >= '2017']

    new_df = df.drop_duplicates(subset=['Timestamp'])
    new_df['Close'] = new_df['Close'].fillna(method='ffill')
    new_df['Weighted_Price'] = new_df['Weighted_Price'].fillna(method='ffill')

    new_df = new_df.fillna({'Open': new_df['Close'],
                            'High': new_df['Close'],
                            'Low': new_df['Close']})
    new_df = new_df.fillna({'Volume_(BTC)': 0,
                            'Volume_(Currency)': 0})

    new_df.reset_index(inplace=True, drop=True)

    last_df = new_df.set_index('Timestamp')
    last_df = last_df.resample('H').agg({'Open': 'first', 'High': 'max',
                                         'Low': 'min', 'Close': 'last',
                                         'Volume_(BTC)': 'sum',
                                         'Volume_(Currency)': 'sum',
                                         'Weighted_Price': 'sum'})
    last_df.fillna(method='ffill', inplace=True)
    # last_df.to_csv('data.csv')
    return last_df
