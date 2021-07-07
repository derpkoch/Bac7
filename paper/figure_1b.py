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
import utils


def main():

    # input/output paths, input filename.
    inpath = '~/Downloads'
    outpath = './figure_1b'
    filename = 'Supplementary Table 2_Ranking_random.csv'

    # number of permutations. Note that increasing this parameter will
    # significantly increase runtime of script. It will take a few minutes for
    # ten permutations, but multiple hours for 1'000.
    n_perm = 10

    # whether or not enrichment curves should be stored. This creates a very
    # large file and is only recommended if needed.
    store_enrichment_curves = False

    data_df = data.load(f'{inpath}/{filename}')
    positions = data_df.columns
    acids = data.aa_from_df(data_df)

    # get positions of stop-codons ('*').
    sc_index_raw, sc_pos_raw = np.where(data_df.isin(["*"]).values == 1)
    sc_index = [x + 1 for x in sc_index_raw]
    sc_pos = [x + 1 for x in sc_pos_raw]

    # arrays for AUC scores.
    metrics = ['auc', 'zscore', 'pval_raw', 'counts']
    out_dict = {x: np.zeros((len(positions), len(acids))) for x in metrics}
    perm_aucs = []
    enrichment_curves = []

    for a_idx, aa in enumerate(acids):
        for p_idx, pos in enumerate(positions):
            print(f'at AA={aa}, pos={pos}\r', end='')

            # remove non-relevant peptides from data, i.e. those that
            # previously had stop-codon.
            df = enrichment.remove_peptides_with_stopcodon(
                data_df, pos, sc_index, sc_pos,
            )

            # check how often an AA occurs at current position in the data set.
            # If zero, nothing has to be done and we continue with next
            # position.
            n_occurrences = df[df == aa].shape[0]
            out_dict['counts'][p_idx, a_idx] = n_occurrences
            if n_occurrences == 0:
                continue

            # compute the enrichment curves.
            curve_ = enrichment.enrichment_curves(df=df, acid=aa)

            # store enrichment curves.
            enrichment_curves.append([aa, pos] + list(curve_))

            # Compute the AUC (observed and null). Same analysis can be done
            # using standard enrichment scores, by changing to the following
            # snippet of code:
            # score_obs = enrichment.enrichment_score(curve_)
            # score_null = enrichment.enrichment_scores_null(
            #     df=df, acid=aa, n_perm=n_perm, mode='max'
            # )
            auc_obs = enrichment.enrichment_auc(curve_)
            auc_null = enrichment.enrichment_scores_null(
                df=df, acid=aa, n_perm=n_perm, mode='auc'
            )

            # store the permutations.
            permutations = [aa, pos, auc_obs] + auc_null
            perm_aucs.append(permutations)

            # compute the p-values.
            auc_pvalue = enrichment.twosided_pvalue(auc_obs, auc_null)

            # compute the z-score.
            auc_zscore = enrichment.zscore(
                auc_obs, np.mean(auc_null), np.std(auc_null)
            )

            # store values in array.
            out_dict['auc'][p_idx, a_idx] = auc_obs
            out_dict['zscore'][p_idx, a_idx] = auc_zscore
            out_dict['pval_raw'][p_idx, a_idx] = auc_pvalue

        print('')

    # compute the FDR correction.
    out_dict['pval_fdr'] = enrichment.fdr_correction(
        out_dict['pval_raw'], alpha=0.1
    )

    # save arrays to files.
    os.makedirs(outpath, exist_ok=True)
    prefix = f'{outpath}/{n_perm}perm_auc'

    # store metrics in files.
    for metric, values in out_dict.items():
        tmp_ = pd.DataFrame(
            index=acids,
            columns=positions,
            data=values.T,
        )
        tmp_.to_csv(f'{prefix}_{metric}.csv')

    # save the permutation values.
    with open(f'{prefix}_permutations.csv', 'w') as fout:
        for line in perm_aucs:
            str_ = ','.join([str(x) for x in line])
            fout.write(f'{str_}\n')

    if store_enrichment_curves:
        with open(f'{outpath}/enrichment_curves.csv', 'w') as fout:
            for line in enrichment_curves:
                str_ = ','.join([str(x) for x in line])
                fout.write(f'{str_}\n')


if __name__ == '__main__':

    with utils.timer('permutations'):
        main()
