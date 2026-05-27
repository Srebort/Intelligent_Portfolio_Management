"""
Module for downloading intraday data from Tiingo API.

This module provides the TiingoLoader class which connects to the Tiingo
IEX endpoint to retrieve historical intraday price data for a given ticker.
The API key is loaded securely from the .env file via python-dotenv.
"""

import os
import pandas as pd
import requests
from pathlib import Path


class TiingoLoader:
    """
    Class to load intraday historical prices using Tiingo 'iex' API.

    The API key is read from the TIINGO_API_KEY environment variable,
    which must be defined in the local .env file (never committed to git).
    """

    def __init__(self):
        """
        Initializes the TiingoLoader, reading the API key from environment variables.

        Raises:
            ValueError: If TIINGO_API_KEY is not set in the environment.
        """
        # Load API key from environment (set in .env, never hardcoded)
        self.api_key = os.getenv("TIINGO_API_KEY")
        # Base URL for the Tiingo IEX intraday endpoint
        self.base_url = "https://api.tiingo.com/iex"

        if not self.api_key:
            raise ValueError(
                "TIINGO_API_KEY is not set in the environment variables. "
                "Please add it to your .env file."
            )

    def download_historical_data(
        self, ticker: str, start_date: str, resample_freq: str = "1Hour"
    ) -> pd.DataFrame:
        """
        Downloads historical intraday data for a given ticker and saves it to a CSV file.

        Args:
            ticker (str): The stock ticker symbol (e.g., 'AAPL', 'SPY').
            start_date (str): The start date for data download in YYYY-MM-DD format.
            resample_freq (str): Resampling frequency for the data (e.g., '1Hour', '4Hour').

        Returns:
            pd.DataFrame: DataFrame containing the historical OHLCV prices.
        """
        print(
            f"Downloading Tiingo data for {ticker} "
            f"from {start_date} at {resample_freq} frequency..."
        )

        # Build the output directory and file path under data/raw/
        save_dir = Path("data/raw")
        save_dir.mkdir(parents=True, exist_ok=True)
        file_path = save_dir / f"{ticker}_{resample_freq}.csv"

        # --- TODO: Implement actual API call ---
        # The request should hit self.base_url with headers:
        #   {"Content-Type": "application/json", "Authorization": f"Token {self.api_key}"}
        # and query params: startDate, resampleFreq, columns=open,high,low,close,volume
        # Save the response JSON as a DataFrame and export to file_path as CSV.

        # Return empty DataFrame as placeholder until API call is implemented
        return pd.DataFrame()
