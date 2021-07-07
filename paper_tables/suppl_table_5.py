#!/usr/bin/env python3
# Author: Bastian Rieck
# Modifications: Anja Gumpinger

import collections
import sys

import pandas as pd

# Import local modules.
sys.path.append('../src/py')
import data
import enrichment


def get_all_scores(df, mode='max'):
    """Computes all scores for a data set.

    Args:
        df (pd.DataFrame): columns correspond to positions, values to amino
            acids at that position.
        mode: whether to report the AUC, or the MAX of the enrichment curve.

    Returns:
        dict: enrichment scores for position, amino acid combinations.
    """
    scores = collections.defaultdict(dict)
    aa = data.aa_from_df(df)

    for position in df.columns:
        scores[position] = {}
        for a in aa:
            scores_ = enrichment.enrichment_curves(df[position], a)

            # Compute enrichment scores.
            if mode == 'max':
                scores[position][a] = enrichment.enrichment_score(scores_)
            # Using the AUC of the enrichment profile.
            elif mode == 'auc':
                scores[position][a] = enrichment.enrichment_auc(scores_)
            else:
                raise NotImplementedError

    return scores


def main():

    mode = 'auc'

    path = '/Users/anjagumpinger/Downloads'
    filename = 'Supplementary Table 4_Ranking_sitesat.csv'

    file_out = f'test.csv'

    # load data.
    df = data.load(f'{path}/{filename}', positions=[4, 17, 18, 19])

    # Get original scores for all position/amino acid combinations.
    original_scores = get_all_scores(df, mode=mode)

    # Prepare all rows for the data frame of the interactions. Each
    # entry of this list will contain tuples of the following form:
    #
    #   Amino Acid, Position, Score
    #
    # This is *repeated* to indicate that one acid has been fixed, and
    # followed by a delta value, indicating the increase or decrease.
    # Said delta is calculated based on the original scores of the
    # second acid.
    interactions = []

    for position1 in sorted(original_scores):
        for acid1 in sorted(original_scores[position1]):

            print(f'at {acid1}, Position {position1}', end='\r')

            score1 = original_scores[position1][acid1]

            # Fix the specified acid in the specified position and
            # re-calculate all the scores of the filtered data set
            # to learn about correlation effects.
            #
            # The position that is fixed is also dropped from the
            # data frame because we must not use it any more here
            # for further comparisons.
            df_filtered = df.loc[df[position1] == acid1]
            df_filtered = df_filtered.drop(position1, axis=1)

            filtered_scores = get_all_scores(df_filtered, mode=mode)

            for position2 in sorted(filtered_scores):
                for acid2 in sorted(filtered_scores[position2]):
                    score2 = filtered_scores[position2][acid2]

                    delta = score2 - original_scores[position2][acid2]

                    interactions.append({
                        'acid1': acid1,
                        'pos1': position1,
                        'score1': score1,
                        'acid2': acid2,
                        'pos2': position2,
                        'score2': score2,
                        'delta': delta,
                    })

    df_out = pd.DataFrame(interactions)
    df_out.to_csv(file_out)


if __name__ == '__main__':

    main()
