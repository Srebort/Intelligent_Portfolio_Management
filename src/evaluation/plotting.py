import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M",
)
logger = logging.getLogger("Plotting")

# Configurar estilo de Seaborn
sns.set_theme(style="darkgrid", context="talk")

def plot_equity_vs_benchmarks(
    equity_path: str = "results/logs/equity_curve.csv",
    benchmark_path: str = "results/logs/benchmarks.csv",
    output_dir: str = "results/figures"
):
    """
    Genera un gráfico comparativo de la curva de equity del bot frente a los benchmarks,
    más un subplot inferior con el histórico de drawdowns.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        equity_df = pd.read_csv(equity_path, parse_dates=["date"])
        bench_df = pd.read_csv(benchmark_path, parse_dates=["date"])
    except FileNotFoundError as e:
        logger.error(f"Falta archivo: {e}")
        return
        
    # Convertir a datetime timezone-aware (UTC)
    if equity_df['date'].dt.tz is None:
        equity_df['date'] = pd.to_datetime(equity_df['date'], utc=True)
    if bench_df['date'].dt.tz is None:
        bench_df['date'] = pd.to_datetime(bench_df['date'], utc=True)
        
    # Merge en fecha
    df = pd.merge(equity_df, bench_df, on="date", how="inner")
    if df.empty:
        logger.error("No hay fechas solapadas entre la simulación y los benchmarks.")
        return
        
    df.set_index("date", inplace=True)
    
    # Extraer los tickers de los benchmarks que existen en el CSV
    # Asumimos que todas las columnas en bench_df menos 'date' son benchmarks
    bench_tickers = [col for col in bench_df.columns if col != "date"]
    
    # -----------------------------------------------------------------
    # NORMALIZACIÓN
    # Para poder compararlos, normalizamos todas las series para que
    # empiecen en el Capital Inicial del Bot (ej. $100,000)
    # -----------------------------------------------------------------
    start_capital = df['total_equity'].iloc[0]
    
    # La del bot ya está en formato dinero, pero la normalizamos por si acaso
    df['Bot_Norm'] = (df['total_equity'] / start_capital) * start_capital
    
    # Normalizar benchmarks
    for ticker in bench_tickers:
        start_price = df[ticker].iloc[0]
        df[f'{ticker}_Norm'] = (df[ticker] / start_price) * start_capital

    # -----------------------------------------------------------------
    # CREACIÓN DE LA FIGURA
    # -----------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(
        2, 1, 
        figsize=(14, 10), 
        gridspec_kw={'height_ratios': [3, 1]},
        sharex=True
    )
    fig.suptitle('Rendimiento del Portfolio Algorítmico vs Mercado', fontsize=18, fontweight='bold', y=0.95)
    
    # --- SUBPLOT 1: Curvas de Crecimiento ---
    # Dibujar benchmarks con líneas más finas y transparentes
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for i, ticker in enumerate(bench_tickers):
        color = colors[i % len(colors)]
        ax1.plot(df.index, df[f'{ticker}_Norm'], label=f"Benchmark: {ticker}", linewidth=1.5, alpha=0.6, color=color)
        
    # Dibujar el Bot con una línea más gruesa y destacada
    ax1.plot(df.index, df['Bot_Norm'], label="Portfolio Algorítmico", linewidth=3, color='#9467bd')
    
    ax1.set_ylabel('Valor del Portfolio ($)')
    ax1.legend(loc="upper left")
    
    # Formatear el eje Y en miles (K)
    current_values = ax1.get_yticks()
    ax1.set_yticklabels(['${:,.0f}K'.format(x/1000) for x in current_values])
    
    # --- SUBPLOT 2: Drawdowns ---
    ax2.fill_between(df.index, df['drawdown_pct'], 0, color='#9467bd', alpha=0.3)
    ax2.plot(df.index, df['drawdown_pct'], color='#9467bd', linewidth=2, label="Portfolio DD")
    
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_xlabel('Fecha')
    ax2.legend(loc="lower left", fontsize=10)
    
    # Limitar el eje Y de drawdowns (ej. desde el peor DD hasta 0)
    min_dd = df['drawdown_pct'].min()
    ax2.set_ylim(min_dd * 1.1, 0)
    
    plt.tight_layout()
    
    # Guardar
    output_path = os.path.join(output_dir, "equity_vs_benchmarks.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Gráfico comparativo guardado exitosamente en: {output_path}")

def plot_drawdown_comparison(
    equity_path: str = "results/logs/equity_curve.csv",
    benchmark_path: str = "results/logs/benchmarks.csv",
    output_dir: str = "results/figures"
):
    """
    Genera un gráfico dedicado exclusivamente a comparar los drawdowns.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        equity_df = pd.read_csv(equity_path, parse_dates=["date"])
        bench_df = pd.read_csv(benchmark_path, parse_dates=["date"])
    except FileNotFoundError as e:
        logger.error(f"Falta archivo: {e}")
        return
        
    if equity_df['date'].dt.tz is None:
        equity_df['date'] = pd.to_datetime(equity_df['date'], utc=True)
    if bench_df['date'].dt.tz is None:
        bench_df['date'] = pd.to_datetime(bench_df['date'], utc=True)
        
    df = pd.merge(equity_df, bench_df, on="date", how="inner")
    if df.empty:
        return
        
    df.set_index("date", inplace=True)
    bench_tickers = [col for col in bench_df.columns if col != "date"]
    
    plt.figure(figsize=(14, 6))
    plt.title('Comparativa de Drawdowns: Portfolio Algorítmico vs Mercado', fontsize=16, fontweight='bold')
    
    # Dibujar drawdowns de benchmarks
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for i, ticker in enumerate(bench_tickers):
        color = colors[i % len(colors)]
        peak = df[ticker].cummax()
        bench_dd = (df[ticker] - peak) / peak * 100
        plt.plot(df.index, bench_dd, color=color, linewidth=1.5, alpha=0.7, label=f"Benchmark: {ticker}")
        
    # Dibujar drawdown del Portfolio
    plt.fill_between(df.index, df['drawdown_pct'], 0, color='#9467bd', alpha=0.3)
    plt.plot(df.index, df['drawdown_pct'], color='#9467bd', linewidth=3, label="Portfolio Algorítmico")
    
    plt.ylabel('Drawdown (%)')
    plt.xlabel('Fecha')
    
    # Limitar el eje Y
    all_dds = [df['drawdown_pct'].min()]
    for ticker in bench_tickers:
        peak = df[ticker].cummax()
        bench_dd = (df[ticker] - peak) / peak * 100
        all_dds.append(bench_dd.min())
        
    plt.ylim(min(all_dds) * 1.1, 0)
    plt.legend(loc="lower left", fontsize=12, ncol=2)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "drawdown_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Gráfico de drawdowns guardado exitosamente en: {output_path}")

if __name__ == "__main__":
    plot_equity_vs_benchmarks()
    plot_drawdown_comparison()

