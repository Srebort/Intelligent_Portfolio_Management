"""
Module for downloading macroeconomic data from FRED API.
"""

import os
import pandas as pd
import requests
from pathlib import Path

class FredLoader:
    """
    Class to load macroeconomic data (e.g., interest rates, inflation) from FRED.
    """
    
    def __init__(self):
        """
        Initializes the FredLoader, reading the API key from environment variables.
        """
        self.api_key = os.getenv("FRED_API_KEY")
        self.base_url = "https://api.stlouisfed.org/fred/series/observations"
        if not self.api_key:
            raise ValueError("FRED_API_KEY is not set in the environment variables.")
            
    def download_macro_data(self, series_id: str, start_date: str) -> pd.DataFrame:
        """
        Downloads macroeconomic data for a given FRED series ID.
        
        Args:
            series_id (str): The FRED series ID (e.g., 'FEDFUNDS', 'CPIAUCSL').
            start_date (str): The start date for the data (YYYY-MM-DD).
            
        Returns:
            pd.DataFrame: DataFrame containing the macroeconomic data.
        """
        print(f"Downloading FRED data for series {series_id} starting from {start_date}...")
        
        save_dir = Path("data/external")
        save_dir.mkdir(parents=True, exist_ok=True)
        file_path = save_dir / f"{series_id}.csv"
        
        # Return empty DataFrame as placeholder
        return pd.DataFrame()
