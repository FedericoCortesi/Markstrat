import pandas as pd
import os

class DataLoader:
    def __init__(self, xlsx_path: str = None, sector : str = None):
        """
        Initialize the DataLoader, loading all necessary sheets from the Excel file.
        
        Parameters:
            xlsx_path (str): Path to the Excel file.
            sector (str): Sonites or Vodites
        """
        if xlsx_path is None:
            files = os.listdir("./Exports")
            files.sort()
            file = files[-1]
            self.xlsx_path = f"./Exports/{file}"
        else:
            self.xlsx_path = xlsx_path

        print(f"{self.xlsx_path} loaded")

        if sector == "Sonites":
            self.sector = sector
        elif sector == "Vodites":
            self.sector = sector
        else:
            print(f"{sector}: Invalid Sector!")
            raise ValueError            

        # Load all required data upon initialization
        self.relative_importance_features = self._load_relative_importance_features()
        self.df_utility = self._load_utility_dataframe()
        self.df_semantic_ideal = self.load_segment_semantic_values()


    def _load_market_studies(self, sheet_name: str, usecols: str, skiprows: int, nrows: int = None):
        """
        Generic loader for extracting data from specific sections of the Studies - Sonites or Vodites Market sheet.
        
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
        data = self._load_market_studies(f"Studies - {self.sector} Market", "F:K", 288)
        data = data.iloc[0].values
        return data / data.sum()  # Normalize to make the sum equal to 1

    def _load_utility_dataframe(self):
        """
        Load utility data for conjoint analysis from Studies - Sonites Market sheet.
        
        Returns:
            pd.DataFrame: Utility data frame.
        """
        return self._load_market_studies(f"Studies - {self.sector} Market", "E:M", 570)

    def load_segment_semantic_values(self):
        """
        Load semantic ideal values from Studies - Sonites Market sheet.
        
        Returns:
            pd.DataFrame: Data frame with ideal semantic values.
        """
        data = self._load_market_studies(f"Studies - {self.sector} Market", "D:K", 255)
        
        # Define new index
        data["Index"] = data["Segment"] + "_" + data["Period"].astype("str")

        if self.sector == "Vodites":
            data = data.replace({"Segment":{"Early Adopters":"Adopters"}})
        else:
            pass

        # Set the new index
        data.set_index("Index", inplace=True) 

        return data.iloc[:32].dropna()

    def load_segment_mds_values(self):
        """
        Load semantic ideal values from Studies - Sonites Market sheet.
        
        Returns:
            pd.DataFrame: Data frame with ideal semantic values.
        """
        data = self._load_market_studies(f"Studies - {self.sector} Market", "D:H", 328)
        
        # Define new index
        data["Index"] = data["Segment"] + "_" + data["Period"].astype("str")
        
        if self.sector == "Vodites":
            data = data.replace({"Segment":{"Early Adopters":"Adopters"}})
        else:
            pass

        # Set the new index
        data.set_index("Index", inplace=True) 

        return data.iloc[:32].dropna()

    def load_physical_characteristics(self) -> pd.DataFrame:
        """
        Load and clean the physical characteristics data for Sonites.

        Returns:
            pd.DataFrame: Cleaned data frame of Sonites' physical characteristics.
        """
        data = pd.read_excel(self.xlsx_path, 
                             sheet_name=self.sector,
                             usecols="D:L",
                             skiprows=17,
                             nrows=10)
        
        data.set_index(f"MARKET : {self.sector}", inplace=True)
        
        # Drop the total column
        data.drop(columns=["Launched in Period"], inplace=True)
        
        return data.dropna()

    def load_semantic_values(self) -> pd.DataFrame:
        """
        Load and clean the semantic values data for Sonites.

        Returns:
            pd.DataFrame: Cleaned data frame of Sonites' semantic values.
        """
        data = pd.read_excel(self.xlsx_path, 
                             sheet_name=f"Studies - {self.sector} Market",
                             usecols="D:K",
                             skiprows=222,
                             nrows=31)
        
        # Delete columns form merged cells in excel 
        data = data.loc[:, ~data.columns.str.contains('Unnamed', case=False)]
        
        # Set index
        data.set_index(f"MARKET : {self.sector}", inplace=True)
        
        return data

    def load_mds_values(self) -> pd.DataFrame:
        """
        Load and clean the multidimensional scaling values data for Sonites.

        Returns:
            pd.DataFrame: Cleaned data frame of Sonites' multidimensional scaling values.
        """
        data = pd.read_excel(self.xlsx_path, 
                             sheet_name=f"Studies - {self.sector} Market",
                             usecols="D:H",
                             skiprows=295,
                             nrows=31)

        # Delete columns form merged cells in excel 
        data = data.loc[:, ~data.columns.str.contains('Unnamed', case=False)]


        data.set_index(f"MARKET : {self.sector}", inplace=True)
        
        return data



    def load_advertising_expenditures_absolute(self) -> pd.DataFrame:
        """
        Load and clean the advertising expenditures values data for Sonites.

        Returns:
            pd.DataFrame: Cleaned data frame of Sonites' advertising expenditures.
        """
        data = pd.read_excel(self.xlsx_path, 
                             sheet_name=f"Studies - {self.sector} Market",
                             usecols="E:K",
                             skiprows=382,
                             nrows=30)

        # Drop the total column
        data.drop(columns=["Total"], inplace=True)

        # Delete columns form merged cells in excel 
        data = data.loc[:, ~data.columns.str.contains('Unnamed', case=False)]

        # Set index
        #data.set_index(f"MARKET : {self.sector}", inplace=True)
        
        return data

    def load_all_info(self) -> pd.DataFrame:
        df1 = self.load_advertising_expenditures_absolute()

        df2 = self.load_mds_values()

        df3 = self.load_physical_characteristics()

        df4 = self.load_semantic_values()
        # Concatenate all the data frames along the rows (axis=0)
        result = pd.concat([df1, df2, df3, df4], axis=1)

        # Rename columns to add an indicator if they appear more than once
        counts = result.columns.value_counts()
        for column, count in counts.items():
            if count > 1:
                result.rename(columns={column: f"{column}_{counts[column] - i}" for i, _ in enumerate(range(count))}, inplace=True)
        return result
        