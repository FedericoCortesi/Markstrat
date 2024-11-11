import pandas as pd

from DataLoader import DataLoader
from Utils import compute_dataframe_cond_prob, cap_dataframe_values

class Sonites:
    """
    A class representing Sonites with physical and semantic characteristics.
    """
    def __init__(self, xlsx_path:str="./Exports/TeamExport_A46051_Alpha_M_Period 2.xlsx") -> None:
        """
        Initialize the Sonites class using a DataLoader instance.
        """
        # Initialize Data Loader
        self.data_loader = DataLoader(xlsx_path)
        
        # Load product data through the data loader
        self.df_sonites_phys_char = self.data_loader.load_sonites_physical_characteristics()
        self.df_sonites_semantic = self.data_loader.load_sonites_semantic_values()
        self.df_sonites_mds = self.data_loader.load_sonites_mds_values()
        
        # Load segment data studies through the data loader
        self.df_segments_semantic = self.data_loader.load_segment_semantic_values()
        self.df_segments_mds = self.data_loader.load_segment_mds_values()
        self.rel_importance_features = self.data_loader.relative_importance_features # semantic scales
        self.df_utility = self.data_loader.df_utility # conjoint analsys
                
        # Obtain features (Move and Most)
        self.move_features = self.get_features("MOVE")
        self.most_features = self.get_features("MOST")

        # Obtain semantic (Move and Most)
        self.move_semantic = self.get_semantic("MOVE")
        self.most_semantic = self.get_semantic("MOST")

        # Obtain MDS (Move and Most)
        self.move_mds = self.get_mds("MOVE")
        self.most_mds = self.get_mds("MOST")

    def get_features(self, brand:str=None):
        dataframe = self.df_sonites_phys_char.copy()
        result = dataframe[dataframe.index == brand]
        return result

    def get_semantic(self, brand:str=None):
        dataframe = self.df_sonites_semantic.copy()
        result = dataframe[dataframe.index == brand].copy()
        return result
    
    def get_mds(self, brand:str=None):
        dataframe = self.df_sonites_mds.copy()
        result = dataframe[dataframe.index == brand].copy()
        return result

    def get_marketing_mixes(self, capped:bool=False)->pd.DataFrame:
        # Obtain the marketing dataframe
        df_advertisement_expenditures_abs = self.data_loader.load_sonites_advertising_expenditures_absolute()
        
        # Compute the dataframe with conditional probabilities
        df_advertisement_expenditures_rel = compute_dataframe_cond_prob(df_advertisement_expenditures_abs)

        if capped:
            # Obtain the dataframe with capped values
            df_capped = cap_dataframe_values(df_advertisement_expenditures_rel)

            # Normalize to obtain values out of 100
            df_result = compute_dataframe_cond_prob(df_capped)
        
        else:
            df_result = df_advertisement_expenditures_rel

        return df_result


    def get_comprehensive_df_semantic(self, last_period:bool=True)->pd.DataFrame:
        df_brands = self.df_sonites_semantic.copy()
        df_segments = self.df_segments_semantic.copy()
        
        if last_period:
            df_segments = df_segments[df_segments["Period"] == max(df_segments["Period"])]
        else:
            pass

        df_result = pd.concat([df_brands, df_segments], join="inner")

        return df_result

    def get_comprehensive_df_mds(self, last_period:bool=True)->pd.DataFrame:
        df_brands = self.df_sonites_mds.copy()
        df_segments = self.df_segments_mds.copy()
        
        if last_period:
            df_segments = df_segments[df_segments["Period"] == max(df_segments["Period"])]
        else:
            pass

        df_result = pd.concat([df_brands, df_segments], join="inner")

        return df_result


    def get_comprehensive_df_features(self, **kwargs)->pd.DataFrame:
        df_sem = self.get_comprehensive_df_semantic(**kwargs)
        df_mds = self.get_comprehensive_df_mds(**kwargs)
        
        df_result = pd.concat([df_sem, df_mds], axis=1)

        return df_result
    
