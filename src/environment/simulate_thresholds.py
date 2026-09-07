import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm

# Asegurar path para imports
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
os.chdir(root_dir)
sys.path.append(root_dir)

from src.environment.portfolio import Portfolio
from src.environment.risk_manager import RiskManager
from src.environment.exit_manager import ExitManager
from src.models.agent_logic import PortfolioAgent
from src.environment.equity_tracker import EquityTracker
from src.environment.run_simulation import load_dataset, load_price_panel

logging.basicConfig(level=logging.WARNING)

def simulate_thresholds():
    # Parámetros base
    DATASET_PATH = "data/processed/MULTI_LABELED_DATASET.csv"
    MODEL_PATH = "models/best_model.pkl"
    SCALER_PATH = "models/scaler.pkl"
    INITIAL_CAPITAL = 100_000.0
    
    print("Cargando datos para simulación de umbrales...")
    df = load_dataset(DATASET_PATH)
    price_panel = load_price_panel("data/raw")
    
    # Filtro 2025
    split_date = pd.to_datetime("2025-01-01", utc=True)
    df = df[df["fecha_entrada"] >= split_date].copy()
    fechas_unicas = sorted(df["fecha_entrada"].dt.normalize().unique())
    
    features_cols = [
        'RSI_14', 'ATR_14', 'NATR_14', 
        'SMA_9', 'EMA_9', 'SMA_21', 'EMA_21', 'SMA_50', 'EMA_50', 'SMA_100', 'EMA_100', 
        'SMA_200', 'EMA_200', 'dist_SMA_50', 'slope_SMA_50', 'dist_SMA_200', 'slope_SMA_200', 
        'atr_squeeze', 'is_bullish_divergence', 'SMA_200_1D', 'dist_SMA_200_1D', 
        'slope_SMA_200_1D', 'RSI_14_1D', 'is_bullish_divergence_1D', 'SMA_50_1D',
        'SMA_200_1W', 'slope_SMA_200_1W', 'dist_SMA_200_1W', 'is_resistance_fractal', 
        'is_support_fractal', 'is_hammer', 'is_inverted_hammer', 'is_bullish_wick_reclaim', 
        'is_bearish_wick_reclaim', 'dist_fib_retr_382', 'dist_fib_retr_618', 'dist_fib_ext_up_382', 
        'dist_fib_ext_up_618', 'dist_fib_ext_dn_382', 'dist_fib_ext_dn_618', 'impulso',
        'Tier_A', 'Tier_B', 'Tier_C'
    ]

    thresholds = np.arange(0.05, 0.95, 0.05)
    
    results_win_rate = []
    results_drawdown = []
    results_return = []
    
    print("Ejecutando simulaciones...")
    for th in tqdm(thresholds):
        # Reiniciar entorno
        portfolio = Portfolio(initial_capital=INITIAL_CAPITAL)
        agent = PortfolioAgent(model_path=MODEL_PATH, scaler_path=SCALER_PATH, prob_threshold=th)
        exit_mgr = ExitManager(portfolio=portfolio, time_stop_days=30)
        risk_mgr = RiskManager(capital_inicial=INITIAL_CAPITAL)
        tracker = EquityTracker(initial_capital=INITIAL_CAPITAL)
        
        open_positions_meta = {}
        latest_known_prices = {}
        
        for idx_d, current_date in enumerate(fechas_unicas):
            day_df = df[df["fecha_entrada"].dt.normalize() == current_date].copy()
            day_df = day_df.rename(columns={"Ticker": "ticker"})
            
            # Actualizar precios conocidos
            for tck in day_df["ticker"].unique():
                row_tck = day_df[day_df["ticker"] == tck]
                if not row_tck.empty:
                    latest_known_prices[tck] = row_tck.iloc[0]["close"]
                    
            current_prices = {}
            for pos_ticker in open_positions_meta.keys():
                if pos_ticker in latest_known_prices:
                    current_prices[pos_ticker] = latest_known_prices[pos_ticker]
                    
            if open_positions_meta:
                closed_today = exit_mgr.evaluate_exits(current_date, current_prices, open_positions_meta)
                for closed in closed_today:
                    tracker.log_sell(
                        date=current_date,
                        ticker=closed["ticker"],
                        quantity=closed["quantity"],
                        entry_price=closed["entry_price"],
                        exit_price=closed["exit_price"],
                        reason=closed["reason"],
                    )
                    res_enum = "WIN" if closed["pnl"] >= 0 else "LOSS"
                    if closed["reason"] == "TIME_STOP": res_enum = "TIME_STOP"
                    agent.register_trade_result(closed["ticker"], res_enum, current_date)
                    
            vix_value = 20.0
            if "VIXCLS" in day_df.columns:
                vix_value = day_df["VIXCLS"].iloc[0]
                
            if "Tier" in day_df.columns:
                tier_dummies = pd.get_dummies(day_df["Tier"], prefix="Tier")
                for col in ["Tier_A", "Tier_B", "Tier_C"]:
                    if col in tier_dummies.columns:
                        day_df[col] = tier_dummies[col].astype(int)
                    else:
                        day_df[col] = 0
            
            features_available = [c for c in features_cols if c in day_df.columns]
                
            current_equity = portfolio.get_portfolio_value(current_prices)
            approved_orders = agent.process_signals(day_df, features_available, current_equity, current_date, vix_value)
            
            for order in approved_orders:
                ticker = order["ticker"]
                price = order["price"]
                tier = order.get("tier")
                
                row_ticker = day_df[day_df["ticker"] == ticker]
                sl, tp = None, None
                if not row_ticker.empty:
                    sl = row_ticker.iloc[0].get("SL1_ATR_precio")
                    tp = row_ticker.iloc[0].get("TP3_3R_SL1_ATR_precio")
                    
                if sl is not None and tp is not None and not pd.isna(sl) and not pd.isna(tp):
                    quantity = order.get("quantity", 0)
                    if ticker not in open_positions_meta and quantity > 0:
                        cash_antes = portfolio.cash
                        portfolio.execute_trade(ticker, quantity, price, "BUY")
                        if portfolio.cash < cash_antes:
                            open_positions_meta[ticker] = {
                                "quantity": quantity, "entry_price": price,
                                "stop_loss": sl, "take_profit": tp, "entry_date": current_date
                            }
                            tracker.log_buy(current_date, ticker, quantity, price, sl, tp, tier, order.get("probability"))
                            
            current_equity_after = portfolio.get_portfolio_value(current_prices)
            tracker.record_daily_equity(
                date=current_date,
                cash=portfolio.cash,
                current_prices=current_prices,
                positions=portfolio.positions
            )
            
        # Fin de simulación para este threshold
        stats = tracker.summary()
        results_win_rate.append(stats.get("win_rate_pct", 0.0))
        results_drawdown.append(stats.get("max_drawdown_pct", 0.0))
        results_return.append(stats.get("total_return_pct", 0.0))
        
    # Graficar
    plt.figure(figsize=(15, 5))
    
    # 1. Win Rate
    plt.subplot(1, 3, 1)
    plt.plot(thresholds, results_win_rate, marker='o', color='green')
    plt.title('Win Rate vs Threshold (Test Set 2025)')
    plt.xlabel('Probability Threshold')
    plt.ylabel('Win Rate (%)')
    plt.grid(alpha=0.3)
    
    # 2. Max Drawdown
    plt.subplot(1, 3, 2)
    plt.plot(thresholds, results_drawdown, marker='o', color='red')
    plt.title('Max Drawdown vs Threshold (Test Set 2025)')
    plt.xlabel('Probability Threshold')
    plt.ylabel('Max Drawdown (%)')
    plt.grid(alpha=0.3)
    
    # 3. Total Return
    plt.subplot(1, 3, 3)
    plt.plot(thresholds, results_return, marker='o', color='blue')
    plt.title('Total Return vs Threshold (Test Set 2025)')
    plt.xlabel('Probability Threshold')
    plt.ylabel('Total Return (%)')
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    os.makedirs('results/figures', exist_ok=True)
    plt.savefig('results/figures/simulator_threshold_analysis.png', dpi=300)
    print("Gráfico guardado en results/figures/simulator_threshold_analysis.png")

if __name__ == "__main__":
    simulate_thresholds()
