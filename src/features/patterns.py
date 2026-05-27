"""
Module for detecting chart patterns.
"""

import pandas as pd

def detect_bill_williams_fractals(df: pd.DataFrame, high_col: str = 'high', low_col: str = 'low') -> pd.DataFrame:
    """
    Detects Bill Williams Fractals (Up and Down) in the price series.
    
    Args:
        df (pd.DataFrame): Input dataframe containing high and low prices.
        high_col (str): Column name for high prices.
        low_col (str): Column name for low prices.
        
    Returns:
        pd.DataFrame: DataFrame with boolean columns indicating Up and Down fractals.
    """
    pass

def detect_wick_reclaims(df: pd.DataFrame, open_col: str = 'open', high_col: str = 'high', low_col: str = 'low', close_col: str = 'close') -> pd.Series:
    """
    Detects wick reclaims/rejections.
    
    Args:
        df (pd.DataFrame): Input dataframe containing OHLC data.
        open_col, high_col, low_col, close_col (str): Names of OHLC columns.
        
    Returns:
        pd.Series: A series indicating occurrences of wick reclaims.
    """
    pass
