"""
Module for backtesting simulation.
"""

import pandas as pd
from src.environment.portfolio import Portfolio
from src.environment.risk_manager import RiskManager
from src.models.agent_logic import PortfolioAgent

class Backtester:
    """
    Iterates through historical data row by row, simulating an event-driven backtest.
    """
    
    def __init__(self, portfolio: Portfolio, risk_manager: RiskManager, agent: PortfolioAgent):
        """
        Initializes the Backtester.
        
        Args:
            portfolio (Portfolio): Instance of the portfolio.
            risk_manager (RiskManager): Instance of the risk manager.
            agent (PortfolioAgent): Instance of the trading agent.
        """
        self.portfolio = portfolio
        self.risk_manager = risk_manager
        self.agent = agent
        self.equity_curve = []

    def run(self, df_data: pd.DataFrame, model):
        """
        Runs the backtest simulation over the provided data dataframe.
        
        Args:
            df_data (pd.DataFrame): The merged dataframe of historical market and macro data.
            model: The trained predictive model (e.g., TradeSelectorXGB).
        """
        print("Starting backtest iteration...")
        pass
        
    def get_equity_curve(self) -> pd.Series:
        """
        Returns the recorded equity curve.
        """
        return pd.Series(self.equity_curve)
