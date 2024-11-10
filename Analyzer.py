import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline
from DataLoader import DataLoader
from Brands import Sonites

from Utils import compute_distance_from_centroids

class Analyzer:
    def __init__(self, 
                 xlsx_path:str="./Exports/TeamExport_A46051_Alpha_M_Period 2.xlsx",
                 segment:str="Sonites", 
                 marketing_mix_segment_weights:dict=None,
                 last_period:int=None):
        
        # Define path
        self.xlsx_path = xlsx_path
        
        # Define last period
        self.last_period = last_period

        # Define weights and sort them 
        self.marketing_mix_segment_weights = marketing_mix_segment_weights
        if self.marketing_mix_segment_weights != None:
            assert np.isclose(sum(self.marketing_mix_segment_weights.values()), 1), "Weights must sum to 1."

            sorted_keys = sorted(self.marketing_mix_segment_weights.keys())
            self.marketing_mix_segment_weights = {key: self.marketing_mix_segment_weights[key] for key in sorted_keys}        

        else:
            print("No weights provided")


        # Instantiate DataLoader
        self.data_loader = DataLoader(xlsx_path=self.xlsx_path)

        if segment == "Sonites":
            # Instantiate Sonites
            self.sonites = Sonites()
            
            # Obtain the relative importance of features
            self.rel_importance_features = self.sonites.rel_importance_features
        
            # Obtain the df for the utilities (conjoint analysis)
            self.df_utility = self.sonites.df_utility
            
            # Obtain the df with the semantic ideal values
            self.df_segments_semantic = self.sonites.df_segments_semantic


        elif segment == "Vodites":
            pass

        else:
            raise ValueError        

    def _interpolate_and_predict(self, x_values, y_values, x_new=None, steps=20):
        if x_new == None:
            assert steps > len(x_values), "Steps must be greater than the current size of the input"
            x_new = np.linspace(start=min(x_values), stop=max(x_values), endpoint=True, num=steps)

        else:
            pass

        if len(x_values)>3:
            spline = UnivariateSpline(x_values, y_values, k=3, s=0)
        else:
            var_k = len(x_values) - 1
            spline = UnivariateSpline(x_values, y_values, k=var_k, s=0)

        y_interp = spline(x_new)

        return y_interp[-1]

    def compute_forecast_df(self, dataframe=None):
        # Define the last period
        period = max(dataframe["Period"].values) + 1

        # Build a list of periods
        x_new = list(range(1, period+1))

        if dataframe[dataframe["Period"]==period].values.shape[0] == 0:
            df_columns = list(dataframe.columns)

            col_to_remove = ["Segment", "Period"]

            for col in col_to_remove:
                try:
                    df_columns.remove(col)
                except ValueError:
                    pass
            
            # Define new dataframe to store the results
            df_new = dataframe.copy()
            df_new = df_new.sort_values(by="Period")

            # Get the x's (periods)    
            x_values = df_new["Period"].unique()
    
            # Iterate over columns and segments
            for segment in dataframe["Segment"].unique():
                # Define dictionary to later append it to the dataframe
                new_rows = pd.DataFrame({
                    "Period" : period,
                    "Segment" : segment 
                }, index=[0])

                for col in df_columns:
                    # Get the y values
                    y_values = df_new[df_new["Segment"]==segment][col].values
                    
                    # Interpolate
                    new_values = self._interpolate_and_predict(x_values, y_values, x_new)

                    # Add prediction to the dictionary
                    new_rows[col] = new_values                                 

                # Concat new rows
                df_new = pd.concat([new_rows, df_new], ignore_index=True)

            # Sort
            df_new = df_new.sort_values(by=["Period", "Segment"])
            df_new.reset_index(inplace=True, drop=True)

        return df_new

    def compute_centroid(self, dataframe:pd.DataFrame=None, weighted:str="marketing", weights=None, period:int=None)->np.ndarray:
        # Set period for which to comput the centroid
        if period == None:
            period = int(self.last_period) + 1 
        else:
            pass
        
        # Filter and clean dataframe
        df_last = dataframe[dataframe["Period"]==period]
        df_last.drop(columns=["Period", "Segment"], inplace=True)

        # Extract data points from the df
        data_points = df_last.values
        
        # Define weights
        if weighted == "marketing":
            weights = np.array(list(self.marketing_mix_segment_weights.values()))
        
        elif weighted == "eq":
            n_segments = len(dataframe["Segments"].unique())
            weights = np.ones(n_segments)/n_segments
        
        elif weighted == "other":
            weights = weights
        
        else:
            raise TypeError
 
        assert np.isclose(sum(weights), 1), "Weights must sum up to 1!"

        # Compute the weighted centroid
        weighted_centroid = np.sum(data_points.T * weights, axis=1) / np.sum(weights)

        return weighted_centroid

    def compute_distance_centroids(self, df_observations, df_centroids, weighted: str = "default", feature_weights: list = None):
        """
        Compute the distance between observations and centroids with optional feature weighting.

        Parameters:
            df_observations (pd.DataFrame): DataFrame containing the observations to compare.
            df_centroids (pd.DataFrame): DataFrame containing centroid values for comparison.
            weighted (str, optional): Method for applying feature weights:
                - "default": Uses `self.rel_importance_features` as weights.
                - "eq": Applies equal weighting across features.
                - "other": Uses the specified `feature_weights` list.
            feature_weights (list, optional): List of weights for features, used only if `weighted="other"`.

        Returns:
            dict: Computed distances between each observation and each centroid, calculated by `compute_distance_from_centroids`.
        """
        if weighted == "default":
            feature_weights = self.rel_importance_features
        elif weighted == "eq":
            feature_weights = None
        elif weighted == "other":
            feature_weights = feature_weights
            assert feature_weights is not None, '"other" selected, pass a list of feature weights!'
        else:
            raise ValueError
        return compute_distance_from_centroids(
            df_observations,
            df_centroids=df_centroids,
            feature_weights=feature_weights 
        )

    def compute_distance_from_centroids_2(self, df_observations: pd.DataFrame, df_centroids: pd.DataFrame = None, 
                                        weighted: str = "default", feature_weights: list = None, **kwargs):
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
            weighted (str, optional): 
                Method for applying feature weights. Accepts:
                - `"default"`: Uses `self.rel_importance_features` as feature weights.
                - `"eq"`: Assigns equal weight to all features.
                - `"other"`: Uses the provided `feature_weights` list. Defaults to `"default"`.
            feature_weights (list, optional): 
                List of weights for features when calculating weighted distances 
                and relevance scores. Only used if `weighted="other"`.
            **kwargs: 
                Additional keyword arguments passed to the relevance score calculation.

        Returns:
            tuple: 
                A tuple containing four dictionaries:
                - abs_res (dict): Euclidean distance for each observation and centroid.
                - rtv_res (dict): Relative distances (absolute distance divided by feature value) for each observation and centroid.
                - avg_res (dict): Weighted sum of manhattan distances for each observation and centroid.
                - rlv_scr (dict): Relevance scores for each observation and centroid.

        Each dictionary key represents an observation, and each value is a dictionary of distances or scores 
        calculated for that observation relative to each centroid.
        """
        # Discard unnecessary columns and obtain the values
        try:
            df_observations.set_index(["MARKET : Sonites"], inplace=True)
        except KeyError:
            pass

        # Define the columns to keep
        columns_to_keep = ["# Features", "Design Index", "Battery Life", "Display Size", "Proc. Power", "Price"]
     
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

        if weighted == "default":
            feature_weights = self.rel_importance_features
        elif weighted == "eq":
            feature_weights = np.ones(len(columns_to_keep)) / len(columns_to_keep)
        elif weighted == "other":
            feature_weights = feature_weights
        else:
            raise ValueError

        # Assert feature weights sum up to one
        assert np.isclose(np.sum(feature_weights),1), "Weights must sum up to 1!"


        # Convert observations to a list
        observations_list = df_observations.values.tolist()

        # Initialize dictionaries to store results
        abs_res, rtv_res, avg_res, rlv_scr = {}, {}, {}, {}

        for i, feat_values in enumerate(observations_list):
            # Initialize nested dicts for this observation
            abs_res[df_observations.index[i]] = {}
            rtv_res[df_observations.index[i]] = {}
            avg_res[df_observations.index[i]] = {}
            rlv_scr[df_observations.index[i]] = {}

            # If `ideal_df` is provided, use each row as a centroid, else use the provided `centroid`
            centroids = df_centroids.iterrows() 
    
            for index, row in centroids:
                # Set the current centroid
                current_centroid = np.array(row.values)
                feat_values_array = np.array(feat_values)

                # Compute manhattan distance as a base
                manhattan_distance = np.abs(current_centroid - feat_values_array) 
                
                # Compute Euclidean distances
                absolute_distance = np.linalg.norm(current_centroid - feat_values_array)
                abs_res[df_observations.index[i]][index] = absolute_distance
                
                # Compute relative distance
                relative_distance = manhattan_distance / feat_values_array
                rtv_res[df_observations.index[i]][index] = relative_distance
                
                # Calculate weighted average distance 
                avg_distance = np.sum(feature_weights * manhattan_distance)
                avg_res[df_observations.index[i]][index] = avg_distance
                
                # Calculate relevance score
                relevance = self._relevance_score(feat_values_array, current_centroid, feature_weights, **kwargs)
                rlv_scr[df_observations.index[i]][index] = relevance

        return abs_res, rtv_res, avg_res, rlv_scr


    def compute_performance(self, dict_distances:dict=None):
        dict_output = {}
        
        # Iterate over each product (first key)
        for key in dict_distances.keys():
            dict_inner_segments = dict_distances[key]

            dict_output[key] = np.average(list(dict_inner_segments.values()))

        return dict_output


