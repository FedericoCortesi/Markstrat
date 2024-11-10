from typing import Tuple
import pandas as pd
from scipy.stats import rankdata
import numpy as np
from sklearn.preprocessing import StandardScaler


def compute_dataframe_cond_prob(dataframe: pd.DataFrame, normalize_by: str = 'row') -> pd.DataFrame:
    """
    Normalize a DataFrame to compute conditional or marginal probabilities.

    Parameters:
    - dataframe (pd.DataFrame): The DataFrame to normalize.
    - normalize_by (str): Determines normalization type:
      - 'row' (default): Normalizes each row to sum to 1 (conditional probabilities per row).
      - 'column': Normalizes each column to sum to 1 (marginal probabilities per column).

    Returns:
    - pd.DataFrame: A normalized DataFrame where values sum to 1 along the specified axis.

    Raises:
    - ValueError: If `normalize_by` is not 'row' or 'column'.
    """
    if normalize_by == 'row':
        row_sums = dataframe.sum(axis=1)
        dataframe_normalized = dataframe.div(row_sums, axis=0)
    elif normalize_by == 'column':
        col_sums = dataframe.sum(axis=0)
        dataframe_normalized = dataframe.div(col_sums, axis=1)
    else:
        raise ValueError("Invalid normalize_by value. Use 'row' or 'column'.")

    return dataframe_normalized


def cap_dataframe_values(dataframe: pd.DataFrame, cutoff: int = 2) -> pd.DataFrame:
    """
    Cap values in a DataFrame based on a rank threshold.

    Parameters:
    - dataframe (pd.DataFrame): The input DataFrame with values to be filtered based on rank.
    - cutoff (int): The rank threshold above which values will be replaced with zero.

    Returns:
    - pd.DataFrame: A DataFrame where values are retained if their rank within each row is below or equal to 
      the cutoff; otherwise, they are set to zero.
    """
    result = dataframe.copy()

    for index, row in dataframe.iterrows():
        # Rank indices and apply cutoff
        rank_indices = rankdata(row.values, method="min")
        rank_indices = np.where(rank_indices > cutoff, 1, 0)

        # Apply the boolean mask to filter values
        weights_clean = rank_indices * row.values

        # Assign filtered values back to the result DataFrame
        result.loc[index] = weights_clean

    return result


def scaler_data_standard(data: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame, StandardScaler]:
    """
    Standardize features by removing the mean and scaling to unit variance.

    Parameters:
    - data (pd.DataFrame): The input DataFrame to be scaled.

    Returns:
    - np.ndarray: A NumPy array of standardized data with the same shape as the input DataFrame.
    - pd.DataFrame: A Pandas DataFrame of standardized data with the same shape, index, and columns as the input DataFrame.
    - StandardScaler: The fitted StandardScaler object used for transforming the data.
    """
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    return data_scaled,scaler

def inverse_scaler_data_standard(data_scaled: np.ndarray, scaler : StandardScaler) -> pd.DataFrame:
    """
    Revert the standardization of features to the original scale.

    Parameters:
    - data_scaled (np.ndarray): The standardized data to be inverted.
    - scaler (StandardScaler): The fitted StandardScaler object used for the original transformation.

    Returns:
    - pd.DataFrame: A DataFrame with the data reverted to the original scale.
    """
    # Use the scaler to perform the inverse transformation
    data_original = scaler.inverse_transform(data_scaled)
    
    # Return as a DataFrame, preserving the original DataFrame structure
    return pd.DataFrame(data_original, columns=scaler.feature_names_in_, index=data_scaled.index)

