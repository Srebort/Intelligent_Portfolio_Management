"""
Módulo de Backtesting Estricto y Generación de Etiquetas para Machine Learning.

Recorre el dataset desde el momento en que el TierEvaluator genera una señal,
simulando el futuro vela a vela para registrar si cada nivel de SL y TP fue
alcanzado o no.

DISEÑO: Una fila por señal (no por combinación).
========================================================
Se genera UNA fila por señal con columnas binarias que indican si cada nivel
fue tocado por el precio en el futuro:

    sl1_hit, sl2_hit, ..., sl7_hit  → 1 si el precio bajó hasta ese nivel
    tp1_hit, tp2_hit, ..., tp6_hit  → 1 si el precio subió hasta ese nivel
    label_SLx_TPy                   → 1 si TP fue tocado ANTES que SL

Los niveles de SL y TP se calculan en RiskManager (única fuente de verdad),
el mismo módulo que usará el Portfolio Agent en producción (Sprint 5).

REGLAS DE NEGOCIO:
- Choque Intravela: Si la vela toca SL y TP a la vez → SL gana (peor caso).
- Sin límite temporal: se simula hasta el final del dataset disponible.
"""

import logging
import numpy as np
import pandas as pd

from src.environment.risk_manager import RiskManager

logger = logging.getLogger("StrictBacktester")


class StrictBacktester:
    """
    Motor de simulación que produce UNA fila por señal con columnas binarias
    indicando si cada nivel de SL y TP fue alcanzado o no.

    Delega el cálculo de niveles SL y TP al RiskManager, asegurando que la
    lógica de gestión de riesgo sea idéntica entre el backtesting y la
    ejecución real de operaciones (Sprint 5).
    """

    def __init__(self):
        """
        Inicializa el backtester.

        No existe límite temporal artificial: cada operación se simula hasta
        que se acaba el dataset histórico disponible. Las operaciones pueden
        solaparse en el tiempo de forma completamente independiente.
        """
        self.risk_manager = RiskManager()

    # -----------------------------------------------------------------------
    # SIMULACIÓN: UNA PASADA POR SEÑAL, TODOS LOS NIVELES A LA VEZ
    # -----------------------------------------------------------------------

    def _simulate_signal(
        self,
        future_df: pd.DataFrame,
        sl_dict: dict,
        tp_dict: dict,
    ) -> dict:
        """
        Hace UNA pasada por las velas futuras y registra cuándo se toca
        por primera vez cada nivel de SL y TP.

        Aplica la Regla del Peor Caso en choques intravela: si una vela
        toca simultáneamente SL y TP, se considera que el SL fue tocado primero.

        Returns:
            dict con columnas de precio, vela de toque, hit binario y labels derivados.
        """
        result = {}

        # Registrar precios absolutos de cada nivel
        for sl_name, sl_price in sl_dict.items():
            result[f"{sl_name}_precio"] = sl_price
        for tp_name, tp_price in tp_dict.items():
            result[f"{tp_name}_precio"] = tp_price

        # Registrar la primera vela en que se toca cada nivel
        sl_touch = {name: None for name in sl_dict}
        tp_touch = {name: None for name in tp_dict}

        for idx_vela, (_, row) in enumerate(future_df.iterrows()):
            candle_num = idx_vela + 1

            # Evaluar SL primero (Regla del Peor Caso intravela)
            for sl_name, sl_price in sl_dict.items():
                if sl_touch[sl_name] is None and row["low"] <= sl_price:
                    sl_touch[sl_name] = candle_num

            # Evaluar TP: solo se marca si en esa misma vela el SL NO fue tocado primero
            for tp_name, tp_price in tp_dict.items():
                if tp_touch[tp_name] is None and row["high"] >= tp_price:
                    tp_touch[tp_name] = candle_num

        # Registrar resultados binarios por nivel
        for sl_name, candle in sl_touch.items():
            result[f"{sl_name}_vela"] = candle
            result[f"{sl_name}_hit"]  = 1 if candle is not None else 0

        for tp_name, candle in tp_touch.items():
            result[f"{tp_name}_vela"] = candle
            result[f"{tp_name}_hit"]  = 1 if candle is not None else 0

        # Derivar Label binario para cada combinación SL×TP
        # Label = 1 si TP fue tocado ANTES que el SL (menor número de vela)
        # Choque intravela: si ambos se tocan en la misma vela → Pérdida (Label=0)
        for sl_name, sl_candle in sl_touch.items():
            for tp_name, tp_candle in tp_touch.items():
                tp_gano = (
                    tp_candle is not None
                    and (sl_candle is None or tp_candle < sl_candle)
                )
                result[f"label_{sl_name}_{tp_name}"] = 1 if tp_gano else 0

        # Label principal: SL4_SMA200 + TP4_Fib100 (mayor Win Rate histórico)
        result["Label"] = result.get("label_SL4_SMA200_TP4_Fib100", 0)

        return result

    # -----------------------------------------------------------------------
    # MÉTODO PRINCIPAL
    # -----------------------------------------------------------------------

    def run(self, df_prices: pd.DataFrame, signals_df: pd.DataFrame) -> pd.DataFrame:
        """
        Ejecuta el backtesting sobre todas las señales detectadas.

        Genera UNA fila por señal con columnas binarias para cada nivel.
        Los niveles de SL y TP se calculan mediante RiskManager.

        Args:
            df_prices  : Dataset completo de precios 4H MTF.
            signals_df : Filas del dataset con Tier asignado.

        Returns:
            pd.DataFrame: Una fila por señal con todas las features + etiquetas.
        """
        all_results = []
        total = len(signals_df)

        logger.info("=== Iniciando Backtest | %d señales | Sin límite temporal ===", total)

        # Pre-calcular el precio del último fractal de soporte (propagado hacia adelante)
        fractal_support_price = (
            df_prices["low"]
            .where(df_prices.get("is_support_fractal", pd.Series(0, index=df_prices.index)) == 1)
            .shift(1)
            .ffill()
        )

        for i, (fecha_entrada, row_signal) in enumerate(signals_df.iterrows()):
            if (i + 1) % 50 == 0:
                logger.info("Procesando señal %d / %d...", i + 1, total)

            precio_entrada = row_signal["close"]
            atr            = row_signal.get("ATR_14", np.nan)
            sma_50         = row_signal.get("SMA_50", np.nan)
            sma_200        = row_signal.get("SMA_200", np.nan)
            low_vela       = row_signal["low"]
            fractal_low    = fractal_support_price.get(fecha_entrada, np.nan)
            impulso        = precio_entrada - fractal_low if not np.isnan(fractal_low) else np.nan
            riesgo_base    = atr if not np.isnan(atr) else 1.0

            # ← Delegar al RiskManager (única fuente de verdad)
            sl_dict = self.risk_manager.get_stop_losses(
                precio_entrada, low_vela, atr, fractal_low, sma_50, sma_200, impulso
            )
            tp_dict = self.risk_manager.get_take_profits(
                precio_entrada, riesgo_base, impulso
            )

            if not sl_dict or not tp_dict:
                continue

            try:
                future_df = df_prices.loc[fecha_entrada:].iloc[1:]
            except KeyError:
                continue

            if future_df.empty:
                continue

            # Simular todos los niveles en una sola pasada
            sim = self._simulate_signal(future_df, sl_dict, tp_dict)

            # Construir fila del dataset: features originales + resultados
            record = row_signal.to_dict()
            record["impulso"] = round(impulso, 4) if not np.isnan(impulso) else None
            record.update(sim)
            all_results.append(record)

        if not all_results:
            logger.warning("El backtester no generó ningún resultado.")
            return pd.DataFrame()

        df_out = pd.DataFrame(all_results, index=signals_df.index[:len(all_results)])
        df_out.index.name = "fecha_entrada"
        df_out = df_out.sort_index()

        # Resumen
        total_rows = len(df_out)
        win_rate   = df_out["Label"].mean() * 100 if "Label" in df_out else 0
        logger.info(
            "=== Backtest Finalizado | %d señales etiquetadas | Win Rate (SL4+TP4): %.1f%% ===",
            total_rows, win_rate
        )

        logger.info("--- Win Rate por Stop Loss (promedio de TPs) ---")
        for sl_name in sl_dict.keys():
            cols = [c for c in df_out.columns if c.startswith(f"label_{sl_name}_")]
            if cols:
                wr = df_out[cols].mean().mean() * 100
                logger.info("  %-15s: %.1f%%", sl_name, wr)

        logger.info("--- Win Rate por Take Profit (promedio de SLs) ---")
        for tp_name in tp_dict.keys():
            cols = [c for c in df_out.columns if c.endswith(f"_{tp_name}")]
            if cols:
                wr = df_out[cols].mean().mean() * 100
                logger.info("  %-15s: %.1f%%", tp_name, wr)

        return df_out
