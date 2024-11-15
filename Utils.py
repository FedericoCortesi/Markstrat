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

def weighted_distance(observation, benchmark, weights):
    # Weights refer to features importance
    if weights is not None:
        assert np.isclose(sum(weights), 1), "Weights must sum up to 1!"
    else:
        print("No weights provided, using simple average instead.")
        weights = np.ones(len(observation)) / len(observation)

    # Compute result
    result = np.sqrt(np.sum(weights * (benchmark - observation)**2))

    return result


def relevance_score(observation:list=None, benchmark:list=None, weights:list=None, max_distance_1D:int=6):
    """
    Compute relevance score based on a weighted distance and a maximum distance in 1D.
    """
    # Compute max distance, 7-1=6 when using the semantic scales
    max_distance = np.sqrt(np.sum(weights * (max_distance_1D**2)))

    distance = weighted_distance(observation, benchmark, weights)

    return 1 - (distance/max_distance)

def compute_distance_from_centroids(df_observations: pd.DataFrame, df_centroids: pd.DataFrame = None, 
                                    feature_weights: list = None, research:str="semantic", **kwargs):
    """
    Compute multiple distance metrics between observations and centroids.

    This function calculates several distance metrics, including Manhattan distance, relative distance,
    weighted average distance, and relevance scores, between each observation in `df_observations`
    and each centroid in `df_centroids`.

    Parameters:
        df_observations (pd.DataFrame): 
            DataFrame of observations, with each row representing an observation 
            and each column representing a feature.
        df_centroids (pd.DataFrame, optional): 
            DataFrame of centroids, with each row representing a centroid 
            and each column representing a feature. If `None`, a single centroid 
            is created from `df_centroids`.
        feature_weights (list, optional): 
            List of weights for features when calculating weighted distances 
            and relevance scores. Only used if `feature_weights` is not passed.
        **kwargs: 
            Additional keyword arguments passed to the relevance score calculation.

    Returns:
        tuple: 
            A tuple containing four dictionaries:
            - abs_res (dict): Euclidean distance for each observation and centroid.
            - rtv_res (dict): Relative distances (absolute distance divided by feature value) for each observation and centroid.
            - avg_res (dict): Weighted sum of manhattan distances for each observation and centroid.
            - rlv_scr (dict): Relevance scores for each observation and centroid.
            - man_res (dict): Manhattan distance for each feature and centroid.

    Each dictionary key represents an observation, and each value is a dictionary of distances or scores 
    calculated for that observation relative to each centroid.
    """
    # Discard unnecessary columns and obtain the values
    try:
        df_observations.set_index(["MARKET : Sonites"], inplace=True)
    except KeyError:
        pass

    # Define the columns to keep
    columns_to_keep_semantic = ["# Features", "Design Index", "Battery Life", "Display Size", "Proc. Power", "Price"]
    columns_to_keep_mds = ["Economy", "Performance", "Convenience"]

    # Set columns to keep
    if any(col in df_observations.columns for col in columns_to_keep_semantic) or df_observations.shape[1] == 6:
        columns_to_keep = columns_to_keep_semantic
    elif any(col in df_observations.columns for col in columns_to_keep_mds) or df_observations.shape[1] == 3:
        columns_to_keep = columns_to_keep_mds
    else:
        raise ValueError

    # Ensure centroids_df is a dataframe
    if type(df_centroids) is not pd.DataFrame:
        df_centroids = pd.DataFrame({
            "centroid" : df_centroids}).T
        df_centroids.columns = columns_to_keep
    else:
        pass

    # Filter columns to keep only those that exist in df_centroids
    df_centroids = df_centroids.reindex(columns=columns_to_keep)

    if isinstance(df_observations, pd.DataFrame):
        # Filter columns to keep only those that exist in df_observations
        df_observations = df_observations.reindex(columns=columns_to_keep)
    else:
        pass

    if feature_weights is None:
        feature_weights = np.ones(len(columns_to_keep)) / len(columns_to_keep)
    else:
        feature_weights = feature_weights

    # Assert feature weights sum up to one
    assert np.isclose(np.sum(feature_weights),1), "Weights must sum up to 1!"


    # Convert observations to a list
    observations_list = df_observations.values.tolist()

    # Initialize dictionaries to store results
    abs_res = {}
    rtv_res = {}
    avg_res = {}
    rlv_scr = {}
    man_res = {}

    for i, feat_values in enumerate(observations_list):
        # Initialize nested dicts for this observation
        abs_res[df_observations.index[i]] = {}
        rtv_res[df_observations.index[i]] = {}
        avg_res[df_observations.index[i]] = {}
        rlv_scr[df_observations.index[i]] = {}
        man_res[df_observations.index[i]] = {}

        # If `ideal_df` is provided, use each row as a centroid, else use the provided `centroid`
        centroids = df_centroids.iterrows() 

        for index, row in centroids:
            # Set the current centroid
            current_centroid = np.array(row.values)
            feat_values_array = np.array(feat_values)

            # Compute manhattan distance as a base
            manhattan_distance = np.abs(current_centroid - feat_values_array)
            man_res[df_observations.index[i]][index] = manhattan_distance 
            
            # Compute Euclidean distances
            absolute_distance = np.linalg.norm(current_centroid - feat_values_array)
            abs_res[df_observations.index[i]][index] = absolute_distance
            
            # Compute relative distance
            relative_distance = manhattan_distance / feat_values_array
            rtv_res[df_observations.index[i]][index] = relative_distance
            
            # Calculate weighted average distance 
            avg_distance = np.sum(feature_weights * manhattan_distance)
            avg_res[df_observations.index[i]][index] = avg_distance
            
            # Calculate relevance score (weighted by default)
            relevance = relevance_score(feat_values_array, current_centroid, feature_weights, **kwargs)
            rlv_scr[df_observations.index[i]][index] = relevance

    return abs_res, rtv_res, avg_res, rlv_scr, man_res


def combined_error(features, ideal_semantic, ideal_mds, semantic_weights, mds_weights, error_weights, model):
    """
    Calculate the combined error based on relevance scores for semantic and MDS data given the features.

    Parameters:
    - features (list): Observation values for the features.
    - ideal_semantic (list): Ideal values for the semantic features.
    - ideal_mds (list): Ideal values for the MDS features.
    - semantic_weights (list): Weights for the semantic features in the relevance score calculation.
    - mds_weights (list): Weights for the MDS features in the relevance score calculation.
    - semantic_scale (float): Scaling factor for the semantic score in the combined error calculation (default is 1).
    - model: Model object used to compute the semantic and MDS values.

    Returns:
    - float: The total combined error.

    The combined error is a weighted sum of the relevance scores of the semantic 
    and MDS inputs, indicating how closely the observation aligns with the ideal values.
    """
    
    # Obtain the predicted semantic and MDS values using the model
    predicted_semantic = model.regress_semantic(features)
    predicted_mds = model.regress_mds(features)

    # Compute relevance scores for semantic and MDS predictions
    semantic_relevance_score = relevance_score(predicted_semantic, ideal_semantic, semantic_weights, max_distance_1D=6)
    mds_relevance_score = relevance_score(predicted_mds, ideal_mds, mds_weights, max_distance_1D=40)


    # Calculate combined error as the weighted sum
    semantic_error = 1 - semantic_relevance_score
    mds_error = 1 - mds_relevance_score
    
    # Use an array to hold both errors for the dot product with weights.
    errors = np.array([semantic_error, mds_error])

    # Normalize errors to 2
    error_weights = (error_weights / sum(error_weights))*2

    # Now compute the total error as the dot product of the weights and the errors
    total_error_array = np.dot(error_weights, errors)

    total_error = np.sum(total_error_array)

    return total_error


