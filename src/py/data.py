#!/usr/bin/python3
# Author: Anja Gumpinger

import pandas as pd


def load(file, positions=range(23)):
    """Loads the data file into a pandas DataFrame object. The dataframe will
    contain one column per requested position, with name Pos{xx}. Rows are
    ordered by strength, with strongest peptides at top of data frame.

    Args:
        file (string): filename containing data.
        positions (list): contains positions of amino acids of interest.
            Defaults to range(23).

    Return:
        pd.DataFrame: data with one column per position.
    """

    df = pd.read_csv(file, index_col=0)

    # create one column per position.
    for p in positions:
        if p < 10:
            pos = f'Pos0{p+1}'
        else:
            pos = f'Pos{p+1}'
        df[pos] = df['Sequence'].apply(lambda x: x[p])

    df = df.drop('Sequence', axis=1)
    df = df.drop('shrunken.log2.fold.change', axis=1)
    df = df.drop('ID', axis=1)

    return df


def aa_from_df(df):
    """Returns list of all amino acids in the data base.

    Args:
        df (pd.DataFrame): data (obtained with data.load).

    Returns:
        list: list of amino acids.
    """

    return sorted(list(set(df.values.flatten())))
