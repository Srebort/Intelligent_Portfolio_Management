import pandas as pd
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M",
)
logger = logging.getLogger("Metrics")

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods: int = 252) -> float:
    """
    Calcula el Sharpe Ratio anualizado.
    returns: Serie de retornos diarios (ej. 0.01 para 1%)
    risk_free_rate: Tasa libre de riesgo anual (ej. 0.04 para 4%)
    """
    if returns.empty or returns.std() == 0:
        return 0.0
        
    daily_rf = risk_free_rate / periods
    excess_returns = returns - daily_rf
    
    sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(periods)
    return float(sharpe)

def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods: int = 252) -> float:
    """
    Calcula el Sortino Ratio anualizado (penaliza solo volatilidad negativa).
    """
    if returns.empty:
        return 0.0
        
    daily_rf = risk_free_rate / periods
    excess_returns = returns - daily_rf
    negative_returns = excess_returns[excess_returns < 0]
    
    if negative_returns.empty or negative_returns.std() == 0:
        return np.inf if excess_returns.mean() > 0 else 0.0
        
    downside_std = np.sqrt((negative_returns**2).mean())
    sortino = (excess_returns.mean() / downside_std) * np.sqrt(periods)
    return float(sortino)

def calculate_max_drawdown(equity_series: pd.Series) -> float:
    """
    Calcula el Maximum Drawdown real de una curva de equity.
    Devuelve porcentaje negativo (ej. -0.15 para -15%).
    """
    if equity_series.empty:
        return 0.0
        
    peak = equity_series.cummax()
    drawdown = (equity_series - peak) / peak
    return float(drawdown.min())

def calculate_beta(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """
    Calcula la exposición (Beta) de la cartera respecto al mercado.
    """
    if portfolio_returns.empty or benchmark_returns.empty:
        return 1.0
        
    # Alinear series
    df = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    if df.empty:
        return 1.0
        
    cov = df.iloc[:, 0].cov(df.iloc[:, 1])
    var = df.iloc[:, 1].var()
    
    if var == 0:
        return 1.0
        
    return float(cov / var)

def calculate_alpha(portfolio_returns: pd.Series, benchmark_returns: pd.Series, beta: float, risk_free_rate: float = 0.0, periods: int = 252) -> float:
    """
    Calcula el Alpha de Jensen anualizado.
    """
    if portfolio_returns.empty or benchmark_returns.empty:
        return 0.0
        
    ann_port_return = (1 + portfolio_returns.mean())**periods - 1
    ann_bench_return = (1 + benchmark_returns.mean())**periods - 1
    
    alpha = ann_port_return - (risk_free_rate + beta * (ann_bench_return - risk_free_rate))
    return float(alpha)

def evaluate_portfolio(equity_path: str = "results/logs/equity_curve.csv", benchmark_path: str = "results/logs/benchmarks.csv", benchmark_ticker: str = "SPY"):
    """
    Carga los CSVs y genera el reporte institucional de métricas.
    """
    try:
        equity_df = pd.read_csv(equity_path, parse_dates=["date"])
        bench_df = pd.read_csv(benchmark_path, parse_dates=["date"])
    except FileNotFoundError as e:
        logger.error(f"Falta archivo: {e}")
        return None
        
    # Convertir a datetime timezone-aware si no lo son
    if equity_df['date'].dt.tz is None:
        equity_df['date'] = pd.to_datetime(equity_df['date'], utc=True)
    if bench_df['date'].dt.tz is None:
        bench_df['date'] = pd.to_datetime(bench_df['date'], utc=True)
        
    # Merge en fecha
    df = pd.merge(equity_df, bench_df, on="date", how="inner")
    if df.empty:
        logger.error("No hay fechas solapadas entre la simulación y el benchmark.")
        return None
        
    df.set_index("date", inplace=True)
    
    # Calcular retornos diarios
    df['port_ret'] = df['total_equity'].pct_change().fillna(0)
    df['bench_ret'] = df[benchmark_ticker].pct_change().fillna(0)
    
    # Calcular métricas
    sharpe = calculate_sharpe_ratio(df['port_ret'])
    sortino = calculate_sortino_ratio(df['port_ret'])
    max_dd = calculate_max_drawdown(df['total_equity'])
    beta = calculate_beta(df['port_ret'], df['bench_ret'])
    alpha = calculate_alpha(df['port_ret'], df['bench_ret'], beta)
    
    # Rentabilidad acumulada
    port_cum = (df['total_equity'].iloc[-1] / df['total_equity'].iloc[0]) - 1
    bench_cum = (df[benchmark_ticker].iloc[-1] / df[benchmark_ticker].iloc[0]) - 1
    
    metrics = {
        "Total Return": f"{port_cum*100:.2f}%",
        "Benchmark Return": f"{bench_cum*100:.2f}%",
        "Alpha (Anualizado)": f"{alpha*100:.2f}%",
        "Beta": f"{beta:.2f}",
        "Sharpe Ratio": f"{sharpe:.2f}",
        "Sortino Ratio": f"{sortino:.2f}",
        "Max Drawdown": f"{max_dd*100:.2f}%"
    }
    
    logger.info("=" * 50)
    logger.info("REPORTE INSTITUCIONAL DE RENDIMIENTO")
    logger.info("=" * 50)
    for k, v in metrics.items():
        logger.info(f"{k:<20}: {v:>10}")
    logger.info("=" * 50)
    
    return metrics

if __name__ == "__main__":
    evaluate_portfolio()
