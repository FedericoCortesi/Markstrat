import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline
from DataLoader import DataLoader
from Brands import Sonites

from Utils import compute_distance_from_centroids

class Analyzer:
    def __init__(self, 
                 xlsx_path:str=None,
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

        if segment == "Sonites":
            # Instantiate Sonites
            self.sonites = Sonites()
            
            # Obtain the relative importance of features
            self.rel_importance_features = self.sonites.rel_importance_features

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
            print("poly")
            var_k = len(x_values) - 1
            spline = UnivariateSpline(x_values, y_values, k=var_k, s=0)

        y_interp = spline(x_new)

        return y_interp

    def forecast_df(self, dataframe:pd.DataFrame=None, steps:int=1)->pd.DataFrame:
        # Define the last period
        last_period = max(dataframe["Period"].values)

        # Build a list of periods
        x_new = list(range(1, last_period+steps+1))

        if dataframe[dataframe["Period"]==(last_period+1)].values.shape[0] == 0:
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


            # Iterate over columns and segments
            for segment in dataframe["Segment"].unique():
                for period in list(range(last_period+1, last_period+steps+1)):
                    # Define dictionary to later append it to the dataframe
                    new_rows = pd.DataFrame({
                        "Period" : period,
                        "Segment" : segment 
                    }, index=[0])

                    for col in df_columns:
                        # Get the x's (periods)    
                        x_values = df_new["Period"].unique()
                        x_values = [x_val for x_val in x_values if x_val < period]

                        # Get the y values
                        y_values = df_new[df_new["Segment"]==segment][col].values
                        
                        # Interpolate
                        new_values = self._interpolate_and_predict(x_values, y_values, period)

                        new_rows[col] = new_values    

                    # Concat new rows
                    df_new = pd.concat([new_rows, df_new], ignore_index=True)                          
                    # Sort
                    df_new = df_new.sort_values(by=["Segment"])
                    df_new = df_new.sort_values(by=["Period"], ascending=True)

            # Sort
            df_new = df_new.sort_values(by=["Segment"])
            df_new = df_new.sort_values(by=["Period"], ascending=False)
            # New index
            df_new.reset_index(inplace=True, drop=True)
            index = df_new["Segment"]+"_"+ df_new["Period"].astype(str)
            df_new.index = index

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

    def compute_distance_centroids(self, df_observations:pd.DataFrame=None, df_centroids:pd.DataFrame=None, 
                                   weighted: str = "default", feature_weights: list = None, **kwargs)->tuple:
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
            tuple: Computed distances between each observation and each centroid, calculated by `compute_distance_from_centroids`.
                - abs_res (dict): Euclidean distance for each observation and centroid.
                - rtv_res (dict): Relative distances (absolute distance divided by feature value) for each observation and centroid.
                - avg_res (dict): Weighted sum of manhattan distances for each observation and centroid.
                - rlv_scr (dict): Relevance scores for each observation and centroid.
                - man_res (dict): Manhattan distance for each feature and centroid.
                - w_relative_res (dict): Weighted su of the relative distances (not abs) for each observation and centroid.
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
            feature_weights=feature_weights,
            **kwargs 
        )
    def get_n_closest(self, df_base: pd.DataFrame = None, df_performers: pd.DataFrame = None, num_top: int = 3, distance_metric: int = 3, **kwargs):
        # Extract 'weighted' from kwargs, or set it to a default if not provided
        weighted = kwargs.pop("weighted", "default")
        
        # Compute distances with extracted 'weighted' argument
        distance_semantic = self.compute_distance_centroids(
            df_observations=df_base,
            df_centroids=df_performers,
            weighted=weighted,
            **kwargs
        )

        # Initialize dict to store results
        dict_res = {}
        
        for seg in df_base.index:

            print("-"*10,seg,"-"*10)

            # Extract distances for the segment and convert to a list
            list_distances = list(distance_semantic[distance_metric][seg].values())

            # Get indices of the three smallest distances
            if distance_metric == 3:
                top_n_indices = np.argsort(list_distances)[-3:][::-1]
            else:
                top_n_indices = np.argsort(list_distances)[:3]

            # initialize list to store n best
            list_res = []

            # Loop over the top 3 closest segments
            for ind in top_n_indices:
                list_res.append(list_distances[ind])

                print("Segment:\t", df_performers.index[ind])
                print("Distance:\t", list_distances[ind])

            dict_res[seg] = list_res
            print()
        
        return dict_res


    def compute_performance(self, dict_distances:dict=None):
        dict_output = {}
        
        # Iterate over each product (first key)
        for key in dict_distances.keys():
            dict_inner_segments = dict_distances[key]

            dict_output[key] = np.average(list(dict_inner_segments.values()))

        return dict_output


    def interpolate_utilities(df_utility, feature, weights):
        level = f'{feature}_Level'
        util = f'{feature}_Utility'

        # Get unique sectors
        sectors = df_utility["Segment"].unique()
        x_values = df_utility[level].unique()
        # Initialize a DataFrame with x_values as the index
        result_df = pd.DataFrame(x_values, columns=[level])

        # Perform interpolation for each sector and add the results to the DataFrame
        for sector in sectors:
            # Filter the data for the current sector
            x_base = df_utility[df_utility["Segment"] == sector][level].values
            y_base = df_utility[df_utility["Segment"] == sector][util].values

            # Create the spline and interpolate
            spline = UnivariateSpline(x_base, y_base, k=3, s=0)
            y_interp = spline(x_values)
            y_interp = np.clip(y_interp, a_min=0, a_max=1)

            # Add the interpolated y values as a new column in result_df
            result_df[sector] = y_interp

        # Ensure weights add up to 1
        assert np.isclose(sum(weights.values()), 1), "Weights must sum to 1."

        # Compute the weighted average by multiplying each sector's column by its weight
        result_df["Weighted_Average"] = sum(result_df[sector] * weight for sector, weight in weights.items())

        return result_df


    def compute_contribution(self, price:int=None, transfer_cost:int=None, distribution_list:list=None):
        """
        Order: Specialty, Mass, Online (stores)
        """
        assert np.sum(distribution_list)==1, "Weights must sum up to 1!"
        
        # Compute promotions (discounts)
        promotions = [0, 0.1, 0.05]
        net_promotions = np.sum(np.dot(promotions, distribution_list))

        # COmpute retail prive
        retail_price = round(price*(1-net_promotions))
        
        # Compute margins
        distribution_margins = [0.40, 0.30, 0.30]
        net_distribution_margin = np.sum(np.dot(distribution_margins, distribution_list))

        selling_price = round(retail_price*(1-net_distribution_margin))

        unit_contribution = selling_price - transfer_cost

        return unit_contribution


