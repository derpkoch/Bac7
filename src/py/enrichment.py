#!/usr/bin/python3
# Author: Anja Gumpinger

import numpy as np


def remove_peptides_with_stopcodon(df, column, sc_index, sc_col):
    """Removes all rows (peptides) from data frame that have stop codon at
    any position preceding 'column'.

    Args:
         df (pandas.DataFrame): data. Indices correspond to ranked peptides,
            columns correspond to positions. The column names are supposed to
            be of the following format: ['Pos1', 'Pos2', ...].
         column (integer): current position.
         sc_index (list): contains all indices (int) of peptides with stop
            codon.
         sc_col (list): contains all columns (int, 1-based) of peptides with
            stop codon.

    Returns:
        pandas.Series: data Series containing amino acids at current position
            (column). Rows (peptides) with stop codons have been removed.
    """
    data = df[column]

    # find rows (peptides) that previously had a stop codon and should be
    # excluded from analyses.
    col_value = int(column.strip('Pos'))
    row_remove = [x for x, y in zip(sc_index, sc_col) if col_value > y]

    # remove the rows from data, if start codon detected before the current
    # position.
    if len(row_remove) > 0:
        data.drop(axis=0, index=row_remove, inplace=True)
        return data
    else:
        return data


def enrichment_curves(df, acid):
    """Creates the GSE representation as proposed in Subramanian et al. (
    2005) for the current position (implicitly encoded in data).

    Args:
        df (pandas.Series): amino acids for all peptides (indices) at current
            position (the Series is extracted from a data frame where each
            column corresponds to one position, and df is a column of this
            data frame). Rows are assumed to be ranked by activity.
        acid (string): amino acid of interest.

    Returns:
         np.array: 1 dimensional array with random walk representing ranking.

    Note:
        * This corresponds to the setting in Subramanian et al with p=0,
          as described in appendix. No normalisation with correlation (in our
          case, this would correspond to the log-fold change).
    """

    # number of all recorded values (required for normalisation below)
    n = len(df)

    counts = df.value_counts()

    # get number of hits for the specified acid if it exists
    if acid in counts:
        m = counts[acid]
    else:

        # Not sure whether this is the best score to report, but if the
        # acid does not exist, there's no way around it.
        return []

    curve = df.copy()

    mask = curve == acid

    # Normalisation. Required to make resulting curves comparable when the
    # number of sequences varies.
    curve.loc[~mask] = -1 / (n - m)
    curve.loc[mask] = 1 / m

    return curve.cumsum().values


def enrichment_scores_null(df, acid, n_perm=100, mode='auc'):
    """Computes the null distribution of enrichment scores or AUC using the GSE
    representation as proposed in Subramanian et al. (2005) by using
    permutations.

    Args:
        df (pandas.Series): amino acids for all peptides (indices) at current
            position (the Series is extracted from a data frame where each
            column corresponds to one position, and df is a column of this
            data frame). Rows are assumed to be ranked by activity.
        acid (string): amino acid of interest.
        n_perm (int): number of permutations. Note that the choice of this
            parameter significantly impacts runtime.
        mode (string): whether to compute enrichment scores or AUC.
            (options: 'auc' or 'max')

    Returns:
        list: null distribution of enrichment scores/AUCs.
    """

    # number of all recorded values (required for the normalisation
    # below)
    n = len(df)

    counts = df.value_counts()

    # get number of hits for the specified acid if it exists
    if acid in counts:
        m = counts[acid]
    else:

        # Not sure whether this is the best score to report, but if the
        # acid does not exist, there's no way around it.
        return []

    curve = df.copy()

    mask = curve == acid

    # Normalisation. Required to make resulting curves comparable when the
    # number of sequences varies.
    curve.loc[~mask] = -1 / (n - m)
    curve.loc[mask] = 1 / m

    curve = curve.values
    null_values = []
    for x in range(n_perm):
        curve = np.random.permutation(curve)
        tmp_ = np.cumsum(curve)

        if mode == 'auc':
            null_values.append(np.mean(tmp_))
        elif mode == 'max':
            null_values.append(enrichment_score(tmp_))
        else:
            print('Invalid mode')
            null_values = None

    return null_values


def enrichment_score(curve):
    """Computes the enrichment score as the most extreme value of
    the enrichment curve, as described in Subramanian et al. (2005).

    Args:
        curve (np.array): enrichment curve.

    Returns:
        float: enrichment score.

    """
    return curve[np.abs(curve).argmax()]


def zscore(x, mu, std):
    """Computes z-score of x.

    Args:
        x (float): x-value.
        mu (float): mean.
        std (float): standard deviation.

    Returns:
        float: z-score.
    """

    return (x-mu)/std


def twosided_pvalue(obs, null):
    """Computes the two-sided p-value from observed and permuted null
    hypothesis values.

    Args:
        obs (float): observed value.
        null (list): list of floats, contains null values.

    Returns:
        float: two-sided permutation p-value.
    """

    if len(null) == 0:
        return 1.0

    n_perm = len(null)
    numerator = len([1 for x in null if np.abs(x) >= np.abs(obs)])
    return numerator/n_perm


def fdr_correction(pvalues, alpha=0.1):
    """Computes Benjamini-Hochberg FDR correction.

    Args:
        pvalues (np.array): array of pvalues.
        alpha (float): FDR.

    Returns:
        np.array: boolean array, each entry indicates whether the pvalue of
            same row and column index in pvalues is significant.
    """

    n_rows, n_cols = np.shape(pvalues)
    flat_pvals = pvalues.reshape(-1)

    # compute FDR
    significant, q_vals = benjamini_hochberg(flat_pvals, alpha)

    # recreate original shape.
    significant_reshaped = significant.reshape([n_rows, n_cols])

    return significant_reshaped


def benjamini_hochberg(pvalues, alpha=0.05):
    """Computes the Benjamini-Hochberg q-values and returns whether a test
    is significant or not (binary).

    Args:
      pvalues (np.array): p-values.
      alpha (float): false discovery rate.

    Returns:
      np.array: binary array, indicating whether pvalue with same row/col
          index is significant.
      np.array: corresponding q-values.
    """

    pvals_arr = np.asarray(pvalues)

    # get the sorting indices of the list.
    sort_idx = sorted(range(pvals_arr.size), key=lambda x: pvals_arr[x])

    qvals_arr = np.ones(pvals_arr.size)

    qvals_arr[sort_idx] = np.arange(1, pvals_arr.size+1) / pvals_arr.size
    qvals_arr *= alpha

    is_sig = pvals_arr <= qvals_arr

    return is_sig, qvals_arr


def main():
    pass


if __name__ == '__main__':
    main()
