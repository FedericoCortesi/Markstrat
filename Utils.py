import pandas as pd
from scipy.stats import rankdata
import numpy as np


def compute_dataframe_conditional_probabilities(dataframe:pd.DataFrame)->pd.DataFrame:

    dataframe = dataframe.div(dataframe.sum(axis=1), axis=0)

    return dataframe


def cap_dataframe_values(dataframe, cutoff:int=2)->pd.DataFrame:
    result = dataframe.copy()

    for index, row in dataframe.iterrows():
        # Rank indices and above cutoff
        rank_indices = rankdata(row.values, method="min")
        rank_indices = np.where(rank_indices>cutoff,1,0)

        # Filter weights by multypling values with a boolean mask
        weights_clean = rank_indices * row.values

        # Assign values to output df
        result.loc[index] = weights_clean

    return result
