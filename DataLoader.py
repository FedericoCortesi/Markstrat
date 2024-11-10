import pandas as pd

class DataLoader:
    def __init__(self, xlsx_path: str = None):
        """
        Initialize the DataLoader, loading all necessary sheets from the Excel file.
        
        Parameters:
            xlsx_path (str): Path to the Excel file.
        """
        self.xlsx_path = xlsx_path
        # Load all required data upon initialization
        self.relative_importance_features = self._load_relative_importance_features()
        self.df_utility = self._load_utility_dataframe()
        self.df_semantic_ideal = self.load_segment_semantic_values()

    def _load_sonites_market_studies(self, sheet_name: str, usecols: str, skiprows: int, nrows: int = None):
        """
        Generic loader for extracting data from specific sections of the Studies - Sonites Market sheet.
        
        Parameters:
            sheet_name (str): Name of the sheet to load.
            usecols (str): Columns to read.
            skiprows (int): Rows to skip at the start.
            nrows (int): Number of rows to read. Reads all if None.
        
        Returns:
            pd.DataFrame: Loaded data frame with the specified parameters.
        """
        return pd.read_excel(self.xlsx_path, sheet_name=sheet_name, usecols=usecols, skiprows=skiprows, nrows=nrows)

    def _load_relative_importance_features(self):
        """
        Load relative importance of features from Studies - Sonites Market sheet.
        
        Returns:
            pd.Series: Series with relative importance feature values.
        """
        data = self._load_sonites_market_studies("Studies - Sonites Market", "F:K", 288)
        data = data.iloc[0].values
        return data / data.sum()  # Normalize to make the sum equal to 1

    def _load_utility_dataframe(self):
        """
        Load utility data for conjoint analysis from Studies - Sonites Market sheet.
        
        Returns:
            pd.DataFrame: Utility data frame.
        """
        return self._load_sonites_market_studies("Studies - Sonites Market", "E:M", 570)

    def load_segment_semantic_values(self):
        """
        Load semantic ideal values from Studies - Sonites Market sheet.
        
        Returns:
            pd.DataFrame: Data frame with ideal semantic values.
        """
        data = self._load_sonites_market_studies("Studies - Sonites Market", "D:K", 255)
        
        # Define new index
        data["Index"] = data["Segment"] + "_" + data["Period"].astype("str")
        
        # Set the new index
        data.set_index("Index", inplace=True) 

        return data.iloc[:32].dropna()

    def load_segment_mds_values(self):
        """
        Load semantic ideal values from Studies - Sonites Market sheet.
        
        Returns:
            pd.DataFrame: Data frame with ideal semantic values.
        """
        data = self._load_sonites_market_studies("Studies - Sonites Market", "D:H", 328)
        
        # Define new index
        data["Index"] = data["Segment"] + "_" + data["Period"].astype("str")
        
        # Set the new index
        data.set_index("Index", inplace=True) 

        return data.iloc[:32].dropna()



    def load_sonites_physical_characteristics(self) -> pd.DataFrame:
        """
        Load and clean the physical characteristics data for Sonites.

        Returns:
            pd.DataFrame: Cleaned data frame of Sonites' physical characteristics.
        """
        data = pd.read_excel(self.xlsx_path, 
                             sheet_name="Sonites",
                             usecols="D:L",
                             skiprows=17)
        return data.dropna()

    def load_sonites_semantic_values(self) -> pd.DataFrame:
        """
        Load and clean the semantic values data for Sonites.

        Returns:
            pd.DataFrame: Cleaned data frame of Sonites' semantic values.
        """
        data = pd.read_excel(self.xlsx_path, 
                             sheet_name="Studies - Sonites Market",
                             usecols="E:K",
                             skiprows=222)
        
        data.set_index("MARKET : Sonites", inplace=True)
        
        return data.iloc[:31].dropna()

    def load_sonites_mds_values(self) -> pd.DataFrame:
        """
        Load and clean the multidimensional scaling values data for Sonites.

        Returns:
            pd.DataFrame: Cleaned data frame of Sonites' multidimensional scaling values.
        """
        data = pd.read_excel(self.xlsx_path, 
                             sheet_name="Studies - Sonites Market",
                             usecols="E:H",
                             skiprows=295)
        
        data.set_index("MARKET : Sonites", inplace=True)
        
        return data.iloc[:31].dropna()



    def load_sonites_advertising_expenditures_absolute(self) -> pd.DataFrame:
        """
        Load and clean the advertising expenditures values data for Sonites.

        Returns:
            pd.DataFrame: Cleaned data frame of Sonites' advertising expenditures.
        """
        data = pd.read_excel(self.xlsx_path, 
                             sheet_name="Studies - Sonites Market",
                             usecols="E:K",
                             skiprows=382)

        # Drop the total column
        data.drop(columns=["Total"], inplace=True)

        # Set the new index
        data.set_index("MARKET : Sonites", inplace=True)
        
        return data.iloc[:30].dropna()
