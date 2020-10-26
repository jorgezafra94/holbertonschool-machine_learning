#!/usr/bin/env python3
"""
read a csv file with pandas
using DS4A knowledge
"""

import pandas as pd


def from_file(filename, delimiter):
    """
    * filename is the file to load from (csv file)
    * delimiter is the column separator
    Returns: the loaded pd.DataFrame
    """
    pd.set_option('display.max_columns', None)
    df = pd.read_csv(filename, delimiter=delimiter)
    return df
