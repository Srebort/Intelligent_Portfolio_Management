"""
Motor Central de Simulación Histórica — run_simulation.py

Implementa el Orquestador Temporal de Backtesting: el bucle cronológico
que simula el paso del tiempo en el mercado financiero señal a señal,
procesando todas las empresas del universo de inversión de forma simultánea.

Rutina diaria estricta (orden obligatorio para evitar lookahead bias):
    1. Actualizar precios de mercado del Portfolio.
    2. Gestionar VENTAS: evaluar SL, TP y Time-Stop con el ExitManager.
    3. Procesar nuevas SEÑALES: filtrar con el PortfolioAgent (IA + reglas).
    4. Ejecutar COMPRAS: abrir las posiciones aprobadas.
    5. Registrar el equity total al cierre del día (Equity Curve).

Uso:
    python -m src.environment.run_simulation

Referencias:
    López de Prado, M. (2018). Advances in Financial Machine Learning.
    John Wiley & Sons. (Capítulo 3 y 4: Triple-Barrier & Backtesting).
"""

import logging
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.environment.equity_tracker import EquityTracker
from src.environment.exit_manager import ExitManager
from src.environment.portfolio import Portfolio
from src.models.agent_logic import PortfolioAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M",
)
logger = logging.getLogger("RunSimulation")

# ---------------------------------------------------------------------------
# CONFIGURACIÓN DE LA SIMULACIÓN
DATASET_PATH    = "data/processed/MULTI_LABELED_DATASET.csv"
MODEL_PATH      = "models/best_model.pkl"
SCALER_PATH     = "models/scaler.pkl"
INITIAL_CAPITAL = 100_000.0          # Capital inicial en USD
COMMISSION_RATE = 0.001              # 0.1% por operación
SLIPPAGE_RATE   = 0.0005             # 0.05% de deslizamiento
TIME_STOP_DAYS  = 9999              # Time-Stop: Desactivado temporalmente (antes 30)

# Columnas de features EXACTAS que usó el MLPipeline en el Sprint 4
# (obtenidas ejecutando: MLPipeline().prepare_features_and_target())
FEATURES_COLS = [
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
]    # One-Hot Encoding de Tier (generado por MLPipeline con pd.get_dummies)



def load_dataset(path: str) -> pd.DataFrame:
    """
    Carga el dataset etiquetado y lo ordena cronológicamente.
    Garantiza que fecha_entrada sea datetime con timezone.
    """
    logger.info(f"Cargando dataset desde: {path}")
    df = pd.read_csv(path)
    df["fecha_entrada"] = pd.to_datetime(df["fecha_entrada"], utc=True)
    df = df.sort_values("fecha_entrada").reset_index(drop=True)
    tickers = df["Ticker"].unique().tolist()
    fechas  = df["fecha_entrada"].dt.date.nunique()
    logger.info(f"Dataset cargado: {len(df)} señales | {len(tickers)} tickers | {fechas} fechas únicas")
    return df


def load_price_panel(raw_dir: str = "data/raw") -> pd.DataFrame:
    """
    Carga todos los CSVs de precios 4H descargados de Tiingo y construye
    un panel de precios diarios: filas=fechas, columnas=tickers.

    Esto permite al simulador conocer el precio REAL de cada activo en
    cada día de mercado, no solo en los días con señales de Tier.
    Así el ExitManager puede evaluar SL/TP con precios actualizados.

    Returns:
        pd.DataFrame: Panel con índice de fechas (UTC, normalizado a día)
                      y una columna por ticker con el precio de cierre.
    """
    raw_path = Path(raw_dir)
    frames = {}

    for csv_file in sorted(raw_path.glob("*_4Hour.csv")):
        ticker = csv_file.stem.replace("_4Hour", "")
        try:
            df_raw = pd.read_csv(csv_file, index_col="datetime", parse_dates=True)
            if df_raw.index.tz is None:
                df_raw.index = df_raw.index.tz_localize("UTC")
            else:
                df_raw.index = df_raw.index.tz_convert("UTC")
            # Resamplear a diario: tomar el último cierre de cada día de mercado
            daily_close = df_raw["close"].resample("1D").last().dropna()
            frames[ticker] = daily_close
        except Exception as e:
            logger.warning(f"No se pudo cargar precio raw para {ticker}: {e}")

    if not frames:
        logger.warning("No se encontraron CSVs en data/raw/. Los precios diarios no estarán disponibles.")
        return pd.DataFrame()

    panel = pd.DataFrame(frames)
    # Forward-fill para cubrir fines de semana / festivos
    panel = panel.ffill()
    logger.info(
        f"Panel de precios cargado: {len(panel)} días × {len(panel.columns)} tickers "
        f"({panel.index[0].date()} → {panel.index[-1].date()})"
    )
    return panel


def run_simulation():
    """
    Función principal. Ejecuta la simulación histórica completa y devuelve
    el equity_curve (lista de dicts) y el trade_log (lista de dicts).
    """
    # ------------------------------------------------------------------
    # 1. INICIALIZACIÓN
    # ------------------------------------------------------------------
    df = load_dataset(DATASET_PATH)

    # Cargar panel de precios diarios reales desde los CSVs raw.
    # FIX: sin esto, los precios se quedan congelados en el valor de entrada
    # y el ExitManager nunca puede evaluar SL/TP correctamente.
    price_panel = load_price_panel("data/raw")

    portfolio = Portfolio(
        initial_capital=INITIAL_CAPITAL,
        commission_rate=COMMISSION_RATE,
        slippage_rate=SLIPPAGE_RATE,
    )

    agent = PortfolioAgent(
        model_path=MODEL_PATH,
        scaler_path=SCALER_PATH,
        prob_threshold=0.0
    )
    exit_mgr = ExitManager(
        portfolio=portfolio,
        time_stop_days=TIME_STOP_DAYS,
    )

    tracker = EquityTracker(initial_capital=INITIAL_CAPITAL)

    # open_positions_meta: metadatos de cada posición abierta
    # { ticker: {quantity, entry_price, stop_loss, take_profit, entry_date} }
    open_positions_meta: dict = {}

    # latest_known_prices: { ticker: close_price }
    # Se inicializa con datos del panel raw (precios reales) y se actualiza
    # cada día. Garantiza que nunca usemos precios congelados.
    latest_known_prices: dict = {}
    


    # ------------------------------------------------------------------
    # 2. BUCLE CRONOLÓGICO: señal por señal, agrupando por fecha
    # ------------------------------------------------------------------
    fechas_unicas_todas = sorted(df["fecha_entrada"].dt.normalize().unique())
    # Filtro Out-Of-Sample: Solo simulamos de 2025 en adelante
    split_date = pd.to_datetime("2025-01-01", utc=True)
    fechas_unicas = [f for f in fechas_unicas_todas if f >= split_date]
    total_fechas  = len(fechas_unicas)

    if not fechas_unicas:
        logger.error("No hay fechas disponibles a partir de 2025. Revisa el dataset.")
        return [], []

    logger.info(f"Iniciando simulación: {fechas_unicas[0].date()} → {fechas_unicas[-1].date()}")
    logger.info(f"Capital inicial: ${INITIAL_CAPITAL:,.0f}")
    logger.info("-" * 60)

    for i, fecha_ts in enumerate(fechas_unicas):
        current_date = fecha_ts.to_pydatetime()

        # Filtrar señales del día actual para todos los tickers
        mask    = df["fecha_entrada"].dt.normalize() == fecha_ts
        day_df  = df[mask].copy()

        # ---------------------------------------------------------------
        # ACTUALIZACIÓN DE PRECIOS: panel raw (fuente primaria)
        # ---------------------------------------------------------------
        # 1. Obtener precios reales del día desde el panel de CSVs raw.
        #    Buscamos la fecha más cercana anterior disponible (ffill ya aplicado).
        if not price_panel.empty:
            # Buscar la fila del panel más cercana a la fecha actual
            panel_dates = price_panel.index
            # Fecha de cierre de mercado: tomar la última fecha <= fecha_ts
            valid_dates = panel_dates[panel_dates <= fecha_ts]
            if len(valid_dates) > 0:
                panel_row = price_panel.loc[valid_dates[-1]]
                for ticker_col in panel_row.index:
                    price = panel_row[ticker_col]
                    if pd.notna(price) and price > 0:
                        latest_known_prices[ticker_col] = float(price)

        # 2. Sobrescribir con los precios exactos de las señales de hoy
        #    (precio de cierre de la vela de señal — máxima precisión)
        for ticker, close_price in zip(day_df["Ticker"], day_df["close"]):
            latest_known_prices[ticker] = close_price

        # Precios actuales = combinación de panel real + señales del día
        current_prices = latest_known_prices.copy()

        # ---------------------------------------------------------------
        # PASO 1: Calcular equity actual antes de cualquier operación
        # ---------------------------------------------------------------
        equity_before = portfolio.get_portfolio_value(current_prices)

        # ---------------------------------------------------------------
        # PASO 2: GESTIONAR VENTAS (SL, TP, Time-Stop)
        # ---------------------------------------------------------------
        if open_positions_meta:
            closed_today = exit_mgr.evaluate_exits(
                current_date=current_date,
                current_prices=current_prices,
                open_positions_meta=open_positions_meta,
            )

            for closed in closed_today:
                meta = open_positions_meta.get(closed["ticker"], {})
                tracker.log_sell(
                    date=current_date,
                    ticker=closed["ticker"],
                    quantity=closed["quantity"],
                    entry_price=closed["entry_price"],
                    exit_price=closed["exit_price"],
                    reason=closed["reason"],
                )
                
                # Registrar resultado en el Agente (para Cooldown y Kelly)
                res_enum = "WIN" if closed["pnl"] >= 0 else "LOSS"
                if closed["reason"] == "TIME_STOP":
                    res_enum = "TIME_STOP"
                    
                agent.register_trade_result(
                    ticker=closed["ticker"],
                    result=res_enum,
                    current_date=current_date
                )
                
                agent.trade_log.append({
                    "result": "WIN" if closed["pnl"] >= 0 else "LOSS",
                    "pnl":    closed["pnl"],
                })

            # Recalcular Kelly Fraccional cada 50 operaciones
            if len(agent.trade_log) > 0 and len(agent.trade_log) % 50 == 0:
                agent.update_kelly_factor(lookback=50)

        # ---------------------------------------------------------------
        # PASO 3: PROCESAR NUEVAS SEÑALES con el PortfolioAgent (IA)
        # ---------------------------------------------------------------
        approved_orders = []

        if not day_df.empty:
            # Renombrar Ticker → ticker para process_signals()
            day_df = day_df.rename(columns={"Ticker": "ticker"})

            # One-Hot Encoding de Tier (replicando lo que hizo MLPipeline en entrenamiento)
            if "Tier" in day_df.columns:
                tier_dummies = pd.get_dummies(day_df["Tier"], prefix="Tier")
                for col in ["Tier_A", "Tier_B", "Tier_C"]:
                    if col in tier_dummies.columns:
                        day_df[col] = tier_dummies[col].astype(int)
                    else:
                        day_df[col] = 0

            # Filtrar solo las features que el modelo conoce y que existen en el dataset
            features_available = [c for c in FEATURES_COLS if c in day_df.columns]

            # Extraer VIX del día si está disponible
            vix_value = 0.0
            if "VIXCLS" in day_df.columns:
                vix_value = day_df["VIXCLS"].iloc[0]

            current_equity = portfolio.get_portfolio_value(current_prices)

            approved_orders = agent.process_signals(
                signals_df=day_df,
                features_cols=features_available,
                current_equity=current_equity,
                current_date=current_date,
                vix_value=vix_value,
            )

        # ---------------------------------------------------------------
        # PASO 4: EJECUTAR COMPRAS aprobadas por el Agente
        # ---------------------------------------------------------------
        for order in approved_orders:
            ticker   = order["ticker"]
            quantity = order["quantity"]
            price    = order["price"]
            sl       = None
            tp       = None
            tier     = order.get("tier")

            # Asignación dinámica de SL/TP alineada con las etiquetas de entrenamiento
            row_ticker = day_df[day_df["ticker"] == ticker]
            if not row_ticker.empty:
                sl_col = "SL1_ATR_precio"
                tp_col = "TP3_3R_SL1_ATR_precio"
                
                if sl_col in row_ticker.columns:
                    sl = row_ticker[sl_col].iloc[0]
                if tp_col in row_ticker.columns:
                    tp = row_ticker[tp_col].iloc[0]

            # Fallback de seguridad
            if pd.isna(sl): 
                sl = order.get("stop_loss")

            # Verificar que no hay ya una posición abierta en ese ticker
            if ticker in open_positions_meta:
                logger.info(f"[Simulación] {ticker} ya tiene posición abierta. Señal ignorada.")
                continue

            # Ejecutar la compra en el Portfolio
            cash_antes = portfolio.cash
            portfolio.execute_trade(ticker=ticker, quantity=quantity, price=price, side="BUY")

            # Si la compra fue ejecutada (el cash bajó), registrar metadatos
            if portfolio.cash < cash_antes:
                open_positions_meta[ticker] = {
                    "quantity":    quantity,
                    "entry_price": price,
                    "stop_loss":   sl,
                    "take_profit": tp,
                    "entry_date":  current_date,
                }
                # Registrar la compra en el Trade Log
                tracker.log_buy(
                    date=current_date,
                    ticker=ticker,
                    quantity=quantity,
                    entry_price=price,
                    stop_loss=sl,
                    take_profit=tp,
                    tier=order.get("tier"),
                    probability=order.get("probability"),
                )
                sl_str = f"{sl:.2f}" if sl is not None else "N/A"
                tp_str = f"{tp:.2f}" if tp is not None else "N/A"
                logger.info(
                    f"[BUY] {ticker} | Qty={quantity} | Precio={price:.2f} | "
                    f"SL={sl_str} | TP={tp_str}"
                )

        # ---------------------------------------------------------------
        # PASO 5: REGISTRAR EQUITY AL CIERRE DEL DÍA (con desglose completo)
        # ---------------------------------------------------------------
        tracker.record_daily_equity(
            date=current_date,
            cash=portfolio.cash,
            current_prices=current_prices,
            positions=portfolio.positions,
        )
        equity_end = portfolio.get_portfolio_value(current_prices)

        # Log de progreso cada 30 días
        if (i + 1) % 30 == 0 or i == total_fechas - 1:
            equity_end = portfolio.get_portfolio_value(current_prices)
            pct_change = (equity_end / INITIAL_CAPITAL - 1) * 100
            trade_count = len(tracker.get_trade_log())
            logger.info(
                f"[{current_date.strftime('%Y-%m-%d')}] "
                f"Equity=${equity_end:,.0f} ({pct_change:+.1f}%) | "
                f"Posiciones abiertas={len(open_positions_meta)} | "
                f"Operaciones={trade_count}"
            )

    # ------------------------------------------------------------------
    # 3. RESUMEN FINAL
    # ------------------------------------------------------------------
    stats = tracker.summary()

    logger.info("=" * 60)
    logger.info("SIMULACIÓN COMPLETADA")
    logger.info(f"Capital inicial:        ${INITIAL_CAPITAL:>12,.0f}")
    logger.info(f"Rentabilidad total:     {stats.get('total_return_pct', 0):>+10.1f}%")
    logger.info(f"Max Drawdown:           {stats.get('max_drawdown_pct', 0):>+10.1f}%")
    logger.info(f"Total operaciones:      {stats.get('total_trades', 0):>12}")
    logger.info(f"Win Rate:               {stats.get('win_rate_pct', 0):>11.1f}%")
    logger.info(f"Profit Factor:          {stats.get('profit_factor', 0):>12.2f}")
    logger.info("=" * 60)

    return tracker.get_equity_curve(), tracker.get_trade_log()


if __name__ == "__main__":
    equity_df, trades_df = run_simulation()

    # Guardar resultados para el Sprint 7 (análisis y gráficas)
    import os
    os.makedirs("results/logs", exist_ok=True)
    equity_df.to_csv("results/logs/equity_curve.csv", index=False)
    trades_df.to_csv("results/logs/trade_log.csv", index=False)
    logger.info("Resultados guardados en results/logs/")
