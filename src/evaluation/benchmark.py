import yfinance as yf
import pandas as pd
import logging
import os
from datetime import datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M",
)
logger = logging.getLogger("Benchmark")

# Benchmarks definidos en settings.yaml
BENCHMARKS = {
    "SPY": "S&P 500",
    "QQQ": "Nasdaq 100",
    "DIA": "Dow Jones",
    "IWM": "Russell 2000"
}

# Periodo Out-Of-Sample
START_DATE = "2025-01-01"

def download_benchmarks():
    """
    Descarga los datos históricos de los benchmarks para el periodo OOS
    y los guarda en results/logs/benchmarks.csv
    """
    logger.info(f"Descargando benchmarks desde {START_DATE}...")
    
    # Descargar datos
    tickers_list = list(BENCHMARKS.keys())
    data = yf.download(tickers_list, start=START_DATE, progress=False)
    
    if data.empty:
        logger.error("No se descargaron datos de los benchmarks.")
        return
        
    # Extraer solo los precios de cierre (Close)
    close_prices = data['Close'].copy()
    
    # Reset index para tener la fecha como columna
    close_prices = close_prices.reset_index()
    
    # Asegurar que la columna de fecha se llame 'date' y esté en formato UTC
    close_prices = close_prices.rename(columns={'Date': 'date'})
    
    # Convertir a datetime timezone-aware en UTC
    if close_prices['date'].dt.tz is None:
        close_prices['date'] = close_prices['date'].dt.tz_localize('UTC')
    else:
        close_prices['date'] = close_prices['date'].dt.tz_convert('UTC')
        
    # Normalizar la fecha (quitar horas)
    close_prices['date'] = close_prices['date'].dt.normalize()
    
    # Guardar a CSV
    os.makedirs("results/logs", exist_ok=True)
    out_path = "results/logs/benchmarks.csv"
    close_prices.to_csv(out_path, index=False)
    
    logger.info(f"Benchmarks guardados correctamente en {out_path}")
    logger.info(f"Dimensión del dataset: {close_prices.shape}")
    logger.info("-" * 40)
    for ticker, name in BENCHMARKS.items():
        if ticker in close_prices.columns:
            start_price = close_prices[ticker].iloc[0]
            end_price = close_prices[ticker].dropna().iloc[-1]
            pct_change = ((end_price / start_price) - 1) * 100
            logger.info(f"{name} ({ticker}): {pct_change:+.2f}% desde OOS")

if __name__ == "__main__":
    download_benchmarks()
