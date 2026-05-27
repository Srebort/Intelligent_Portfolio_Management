"""
Module for calculating financial performance metrics.
"""

import pandas as pd
import numpy as np

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Calculates the Sharpe Ratio of the strategy returns.
    
    Args:
        returns (pd.Series): Series of periodic returns.
        risk_free_rate (float): The baseline risk-free rate.
        
    Returns:
        float: The Sharpe Ratio.
    """
    pass

def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """
    Calculates the Maximum Drawdown from the equity curve.
    
    Args:
        equity_curve (pd.Series): Series representing portfolio value over time.
        
    Returns:
        float: The Maximum Drawdown as a percentage.
    """
    pass

def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0, target_return: float = 0.0) -> float:
    """
    Calculates the Sortino Ratio focusing on downside deviation.
    
    Args:
        returns (pd.Series): Series of periodic returns.
        risk_free_rate (float): The baseline risk-free rate.
        target_return (float): The minimum acceptable return.
        
    Returns:
        float: The Sortino Ratio.
    """
    pass
