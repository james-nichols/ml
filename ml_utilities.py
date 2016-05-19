#! /usr/bin/env python

import math
import numpy as np
import scipy as sp
import pandas as pd
import pdb

def split_by_nulls(df, col_null_ratio = 0.5, row_null_ratio = 0.99):
    """ Split a dataframe by rows when enough data in that row is NaN,
     and remove columns from the "null" dataframe that have too much 
     null information """

    null_row = []
    full_row = []

    # cut off is x% of the number of keys in the dataframe
    min_nulls = col_null_ratio * df.shape[1]

    for (row_name, row) in df.iterrows():    
        if row.isnull().sum() > min_nulls:
            null_row.append(row_name)
        else:
            full_row.append(row_name)
    
    df_null = df.ix[null_row, :]
    df_full = df.ix[full_row, :]
    
    for col_name, col in df_null.iteritems():
        # Check out how many nulls in each column
        #print(col_name, col.isnull().sum())
        if col.isnull().sum() > row_null_ratio * df_null.shape[0]:
            df_null = df_null.drop(col_name, axis=1)
    
    return (df_full, df_null, full_row, null_row)


def randomised_imputer(df, max_dummies=100, use_dummies=True, fill_value=-1, verbose=False):
    # First we impute missing values / NaN's
    df_keys_orig = df.keys()

    # Now we take care of categorical variables
    for (df_col_name, df_col) in df.iteritems():
        if df_col.dtype == 'object':
            n_uniques = len(df_col.unique())
            
            # First we take care of missing values
            df_tmp_len = len(df[df_col.isnull()])
            if df_tmp_len>0:
                if verbose:
                    print("Filling for", df_tmp_len, "NaNs in object data", df_col_name)
                # There's few enough to do a randomised imputation...
                df.loc[df_col.isnull(), df_col_name] = np.random.choice(df_col.value_counts().keys().ravel(), size=df_tmp_len,
                                 p = df_col.value_counts(normalize=True).ravel() / df_col.value_counts(normalize=True).ravel().sum() )
                # NB This only replaces NaNs in the ORIGINAL TABLE, df_col seems to be a COPY OF THE COLUMN

                # Most common imputation
                #df.loc[df_col.isnull(), df_col_name] = df_col.value_counts().keys()[0]
                # Mean
                #df.loc[df_col.isnull(), df_col_name] = df_col.mean()
            
            # Now fill with dummies or factorise
            if n_uniques < max_dummies and use_dummies:
                if verbose:
                    print("Adding", n_uniques, "dummies for", df_col_name)
                df_dum = pd.get_dummies(df_col, prefix=df_col_name)

                # There's not *too* many encodings, so we replace the 
                 # categorical variables with dummy columns
                df = df.drop([df_col_name], axis=1)
                df = pd.concat([df, df_dum], axis=1)
            else:
                if verbose:
                    print("Numerising", df_col_name)
                # If there's too many unique values, just make it numerical
                [df_fac, df_lab] = pd.factorize(df[df_col_name])
                df[df_col_name] = df_fac
        else:
            # Numeric type - dfy an imputation scheme - stolen from HBA        
            # NOTE I'D REALLY LIKE TO HAVE A DISTRIBUTION BASED RANDOMISED
            # ROUTINE - USE KERNEL DENSITY ESTIMATION FROM SCIKIT LEARN...
            tmp_len = len(df[df_col.isnull()])
            if verbose:
                print("Filling for", tmp_len, "NaNs in data", df_col_name)
            if tmp_len>0:
                #df.loc[df_col.isnull(), df_col_name] = df_col.mean()
                df.loc[df_col.isnull(), df_col_name] = fill_value

    return df


