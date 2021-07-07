#!/usr/bin/python3
# Author: Anja Gumpinger

import os
import sys

import numpy as np
import pandas as pd

# Import local modules.
sys.path.append('../src/py')
import data
import enrichment


def main():

    # input/output paths, input filename.
    inpath = '~/Downloads'
    outpath = './figure_2b'
    filename = 'Supplementary Table 4_Ranking_sitesat.csv'

    data_df = data.load(f'{inpath}/{filename}', positions=[4, 17, 18, 19])
    positions = data_df.columns
    acids = data.aa_from_df(data_df)

    # arrays for AUC scores.
    array_auc = np.zeros((len(positions), len(acids)))

    for a_idx, aa in enumerate(acids):
        for p_idx, pos in enumerate(positions):

            print(f'at AA={aa}, pos={pos}\r', end='')

            df = data_df[pos]
            curve_ = enrichment.enrichment_curves(df, aa)
            array_auc[p_idx, a_idx] = enrichment.enrichment_auc(curve_)

    # store data in dataframe.
    df = pd.DataFrame(
        index=acids,
        columns=positions,
        data=array_auc.T,
    )

    os.makedirs(outpath, exist_ok=True)
    df.to_csv(f'{outpath}/auc_values.csv', float_format='%.9f')

    pass


if __name__ == '__main__':
    main()

