"""
Module for generating performance plots.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_equity_curve(strategy_equity: pd.Series, benchmark_equity: pd.Series, save_name: str = "equity_curve.png"):
    """
    Generates a plot comparing the strategy's equity curve against a benchmark (e.g., SPY)
    and saves it to the results/figures/ folder.
    
    Args:
        strategy_equity (pd.Series): Equity curve of the portfolio.
        benchmark_equity (pd.Series): Equity curve of the benchmark.
        save_name (str): The filename for the saved plot.
    """
    save_dir = Path("results/figures")
    save_dir.mkdir(parents=True, exist_ok=True)
    file_path = save_dir / save_name
    
    plt.figure(figsize=(10, 6))
    # plt.plot(strategy_equity, label='Strategy')
    # plt.plot(benchmark_equity, label='Benchmark')
    plt.title("Strategy vs Benchmark Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid(True)
    
    plt.savefig(file_path)
    plt.close()
