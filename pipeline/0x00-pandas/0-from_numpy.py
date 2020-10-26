#!/usr/bin/env python3
"""
Convert a Numpy to a Pandas
"""

import pandas as pd


def from_numpy(array):
    """
    * array: numpy array from which you should create the Dataframe
    """
    pd.set_option('display.max_columns', None)
    my_pandas = pd.DataFrame(array)

    # control more than 26 columns we restrict
    if my_pandas.shape[1] > 26:
        my_pandas = my_pandas.iloc[:, 0:26]

    # giving column names alphabetical order and capitalized
    cols_name = []
    for i in range(len(my_pandas.columns.tolist())):
        a = '{}'.format(chr(i + 65))
        cols_name.append(a)
    my_pandas.columns = cols_name
    return my_pandas
