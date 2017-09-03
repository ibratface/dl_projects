import os
import pickle
import pandas as pd
import numpy as np
import pandas_summary as pds


def to_pickle(obj, fname):
    pickle.dump(obj, open(fname, 'wb'))

def from_pickle(fname):
    f = None
    if os.path.exists(fname):
        f = pickle.load(open(fname, 'rb'))
    return f

# generator function for pulling out items a number at a time
def batch(s, n):
    for i in range(0, len(s), n):
        yield s[i:min(i+n, len(s))]

# test function
# display(props.columns)
# [c for c in batch(props.columns, 6)]

# for describing the data with DataFrameSummary
def summarize(df):
    for cols in batch(df.columns, 5):
        display(pds.DataFrameSummary(df[cols]).summary())

# test function
# describe(props)

# for showing data types, unique and missing values
def describe_values(df):
    stats = pd.DataFrame(columns=['column', 'dtype', 'sample', 'unique len', 'missing %'])
    for c in df.columns:
        uniques = df[c].unique()
        if uniques.dtype == np.float64:
            uniques = np.sort(uniques)
        stats.loc[-1] = [c, df[c].dtype, uniques[:3], len(uniques), df[c].isnull().sum(axis=0) / df[c].size * 100]
        stats.index += 1
    stats.set_index('column')
    return stats

# for onehot encoding categorical features
def onehot(df):
    for c in df.columns:
        if df[c].dtype.name == 'category':
            if len(df[c].unique()) <= 32:
                df = df.join(pd.get_dummies(df[c], prefix=c))
            df = df.drop(c, axis=1)
    return df
