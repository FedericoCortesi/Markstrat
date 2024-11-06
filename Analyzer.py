import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline
from DataLoader import DataLoader

class Analyzer:
    def __init__(self, 
                 xlsx_path:str="./Exports/TeamExport_A46051_Alpha_M_Period 2.xlsx", 
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
        
        # Obtain the relative importance of features
        self.rel_importance_features = self.data_loader.relative_importance_features
     
        # Obtain the df for the utilities (conjoint analysis)
        self.df_utility =  self.data_loader.df_utility
        
        # Obtain the df with the semantic ideal values
        self.df_semantic_ideal = self.data_loader.load_segment_semantic_values()
        

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

    def compute_centroid(self, dataframe:pd.DataFrame=None, period:int=None, weights=None):
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
        if weights is None:
            weights = np.array(list(self.marketing_mix_segment_weights.values()))
        else:
            pass

        assert np.isclose(sum(weights), 1), "Weights must sum up to 1!"

        # Compute the weighted centroid
        weighted_centroid = np.sum(data_points.T * weights, axis=1) / np.sum(weights)

        return weighted_centroid


    def _weighted_distance(self, observation, benchmark, weights):
        # Weights refer to features importance
        if weights is not None:
            assert np.isclose(sum(weights), 1), "Weights must sum up to 1!"
        else:
            print("No weights provided, using simple average instead.")
            weights = np.ones(len(observation)) / len(observation)

        # Compute result
        result = np.sqrt(np.sum(weights * (benchmark - observation)**2))

        return result


    def _relevance_score(self, observation:list=None, benchmark:list=None, weights:list=None):
        # Compute max distance, 7-1=6 when using the semantic scales
        max_distance = np.sqrt(np.sum(weights * (6**2)))

        distance = self._weighted_distance(observation, benchmark, weights)

        return 1 - (distance/max_distance)


    def distance_from_centroids(self, df_observations: pd.DataFrame, df_centroids: pd.DataFrame = None, feature_weights: list = None):
        # Discard unnecessary columns and obtain the values
        try:
            df_observations.set_index(["MARKET : Sonites"], inplace=True)
        except KeyError:
            pass
       
        # Ensure centroids_df is a dataframe
        if type(df_centroids) is not pd.DataFrame:
            df_centroids = pd.DataFrame({
                "centroid" : df_centroids}).T
        else:
            pass

        # Clean centroids_df
        try:
            df_centroids = df_centroids[["# Features", "Design Index", "Battery Life", "Display Size", "Proc. Power", "Price"]]
        except KeyError:
            pass

        if feature_weights is None:
            feature_weights = self.rel_importance_features
        else:
            pass

        # Convert observations to a list
        observations_list = df_observations.values.tolist()

        # Initialize dictionaries to store results
        abs_res = {}
        rtv_res = {}
        rlv_res = {}
        avg_res = {}

        for i, feat_values in enumerate(observations_list):
            # Initialize nested dicts for this observation
            abs_res[df_observations.index[i]] = {}
            rtv_res[df_observations.index[i]] = {}
            avg_res[df_observations.index[i]] = {}
            rlv_res[df_observations.index[i]] = {}

            # If `ideal_df` is provided, use each row as a centroid, else use the provided `centroid`
            centroids = df_centroids.iterrows() 
    
            for index, row in centroids:
                # Set the current centroid
                current_centroid = np.array(row.values)
                feat_values_array = np.array(feat_values)
                
                # Compute Manhattan distances
                absolute_distance = np.abs(current_centroid - feat_values_array)
                
                abs_res[df_observations.index[i]][index] = absolute_distance
                
                relative_distance = absolute_distance / feat_values_array
                rtv_res[df_observations.index[i]][index] = relative_distance
                
                # Calculate average distance 
                avg_distance = np.sum(feature_weights * absolute_distance)
                
                avg_res[df_observations.index[i]][index] = avg_distance
                
                # Calculate relevance score
                relevance = self._relevance_score(feat_values_array, current_centroid, feature_weights)
                rlv_res[df_observations.index[i]][index] = relevance

        return abs_res, rtv_res, avg_res, rlv_res


    def compute_performance(self, dict_distances:dict=None):
        dict_output = {}
        
        # Iterate over each product (first key)
        for key in dict_distances.keys():
            dict_inner_segments = dict_distances[key]

            dict_output[key] = np.average(list(dict_inner_segments.values()))

        return dict_output


