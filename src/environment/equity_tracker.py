"""
Módulo de Tracking Estadístico de la Cartera — equity_tracker.py

Registra con precisión milimétrica el estado de la billetera virtual
en cada ciclo de simulación para poder generar las métricas de rendimiento
del Sprint 7 (Sharpe Ratio, Max Drawdown, Win Rate, etc.).

Estructura de datos generada:
    equity_curve: DataFrame con Fecha, Cash, Valor_Acciones, Total_Equity,
                  Posiciones_Abiertas, Drawdown.
    trade_log:    DataFrame con todas las operaciones (BUY y SELL) incluyendo
                  Ticker, Precio_Entrada, Precio_Salida, Cantidad, PnL, Motivo.
"""

import logging
from datetime import datetime

import pandas as pd

logger = logging.getLogger("EquityTracker")


class EquityTracker:
    """
    Registrador estadístico de la cartera virtual.

    Lleva dos tipos de registros:
        - equity_curve: Snapshot diario del valor total de la cartera.
        - trade_log:    Registro de cada operación abierta y cerrada.

    Al finalizar la simulación, ambos registros se pueden exportar
    a CSV para ser analizados y graficados en el Sprint 7.
    """

    def __init__(self, initial_capital: float):
        """
        Inicializa el tracker.

        Args:
            initial_capital (float): Capital inicial de la simulación (para
                                     calcular el drawdown desde el primer día).
        """
        self.initial_capital = initial_capital
        self.peak_equity     = initial_capital

        # Registros internos
        self._equity_records: list = []
        self._trade_records:  list = []

    # ------------------------------------------------------------------
    # REGISTRO DIARIO DEL EQUITY (Equity Curve)
    # ------------------------------------------------------------------

    def record_daily_equity(
        self,
        date: datetime,
        cash: float,
        current_prices: dict,
        positions: dict,
    ) -> dict:
        """
        Registra el snapshot del valor de la cartera al cierre del día.

        Desglose completo:
            - Cash disponible en cuenta.
            - Valor de las posiciones abiertas (acciones × precio actual).
            - Total Equity = Cash + Valor Acciones.
            - Drawdown respecto al máximo histórico.
            - Rentabilidad acumulada respecto al capital inicial.

        Args:
            date (datetime): Fecha actual de la simulación.
            cash (float): Saldo en efectivo del Portfolio.
            current_prices (dict): {ticker: precio_actual} para calcular el
                                   valor de mercado de las posiciones abiertas.
            positions (dict): {ticker: cantidad_acciones} del Portfolio.

        Returns:
            dict: El registro del día (también se añade internamente).
        """
        # Calcular valor de mercado de todas las posiciones abiertas
        stock_value = sum(
            qty * current_prices.get(ticker, 0.0)
            for ticker, qty in positions.items()
        )

        total_equity = cash + stock_value

        # Actualizar el pico histórico para calcular drawdown
        if total_equity > self.peak_equity:
            self.peak_equity = total_equity

        drawdown = (total_equity - self.peak_equity) / self.peak_equity if self.peak_equity > 0 else 0.0
        returns_pct = (total_equity / self.initial_capital - 1) * 100

        record = {
            "date":             date,
            "cash":             round(cash, 2),
            "stock_value":      round(stock_value, 2),
            "total_equity":     round(total_equity, 2),
            "open_positions":   len(positions),
            "drawdown_pct":     round(drawdown * 100, 4),
            "cumulative_return_pct": round(returns_pct, 4),
        }

        self._equity_records.append(record)
        return record

    # ------------------------------------------------------------------
    # REGISTRO DE OPERACIONES (Trade Log)
    # ------------------------------------------------------------------

    def log_buy(
        self,
        date: datetime,
        ticker: str,
        quantity: float,
        entry_price: float,
        stop_loss: float | None,
        take_profit: float | None,
        tier: str | None = None,
        probability: float | None = None,
    ) -> None:
        """
        Registra la apertura de una nueva posición (BUY).

        Args:
            date        : Fecha de apertura.
            ticker      : Símbolo del activo comprado.
            quantity    : Número de acciones compradas.
            entry_price : Precio de entrada efectivo.
            stop_loss   : Nivel de Stop Loss definido por el RiskManager.
            take_profit : Nivel de Take Profit objetivo.
            tier        : Tier de la señal (A*, A, B, C).
            probability : Probabilidad de la IA que aprobó la señal.
        """
        self._trade_records.append({
            "date":         date,
            "ticker":       ticker,
            "side":         "BUY",
            "quantity":     quantity,
            "entry_price":  round(entry_price, 4),
            "exit_price":   None,
            "stop_loss":    round(stop_loss, 4) if stop_loss else None,
            "take_profit":  round(take_profit, 4) if take_profit else None,
            "gross_pnl":    None,
            "net_pnl":      None,
            "result":       None,
            "close_reason": None,
            "tier":         tier,
            "ai_prob":      round(probability, 4) if probability else None,
        })

        sl_str = f"{stop_loss:.2f}" if stop_loss else "N/A"
        tp_str = f"{take_profit:.2f}" if take_profit else "N/A"
        prob_str = f"{probability:.1%}" if probability else "N/A"

        logger.info(
            f"[TradeLog BUY]  {ticker} | Qty={quantity} | "
            f"Precio={entry_price:.2f} | SL={sl_str} | TP={tp_str} | "
            f"Tier={tier} | P(TP)={prob_str}"
        )

    def log_sell(
        self,
        date: datetime,
        ticker: str,
        quantity: float,
        entry_price: float,
        exit_price: float,
        reason: str,
        commission_rate: float = 0.001,
        slippage_rate: float = 0.0005,
    ) -> None:
        """
        Registra el cierre de una posición (SELL) y calcula el P&L real.

        El P&L neto descuenta comisiones de compra + venta y slippage.

        Args:
            date            : Fecha de cierre.
            ticker          : Símbolo del activo vendido.
            quantity        : Número de acciones vendidas.
            entry_price     : Precio al que se abrió la posición.
            exit_price      : Precio de cierre (precio de mercado en el día de salida).
            reason          : Motivo de cierre ('STOP_LOSS', 'TAKE_PROFIT', 'TIME_STOP').
            commission_rate : Tasa de comisión por operación.
            slippage_rate   : Tasa de slippage aplicada.
        """
        # Precio real de ejecución (con slippage aplicado)
        real_exit   = exit_price * (1 - slippage_rate)
        real_entry  = entry_price * (1 + slippage_rate)

        gross_pnl = (exit_price - entry_price) * quantity
        comm_buy  = entry_price * quantity * commission_rate
        comm_sell = exit_price  * quantity * commission_rate
        net_pnl   = (real_exit - real_entry) * quantity - comm_buy - comm_sell

        result = "WIN" if net_pnl >= 0 else "LOSS"

        # Actualizar el registro de BUY correspondiente si existe
        for record in reversed(self._trade_records):
            if record["ticker"] == ticker and record["side"] == "BUY" and record["exit_price"] is None:
                record["exit_price"]    = round(exit_price, 4)
                record["gross_pnl"]     = round(gross_pnl, 2)
                record["net_pnl"]       = round(net_pnl, 2)
                record["result"]        = result
                record["close_reason"]  = reason
                break
        else:
            # Si no hay BUY correspondiente, crear registro SELL independiente
            self._trade_records.append({
                "date":         date,
                "ticker":       ticker,
                "side":         "SELL",
                "quantity":     quantity,
                "entry_price":  round(entry_price, 4),
                "exit_price":   round(exit_price, 4),
                "stop_loss":    None,
                "take_profit":  None,
                "gross_pnl":    round(gross_pnl, 2),
                "net_pnl":      round(net_pnl, 2),
                "result":       result,
                "close_reason": reason,
                "tier":         None,
                "ai_prob":      None,
            })

        logger.info(
            f"[TradeLog SELL] {ticker} | {reason} | "
            f"P&L bruto={gross_pnl:+.2f}$ | P&L neto={net_pnl:+.2f}$ | {result}"
        )

    # ------------------------------------------------------------------
    # EXPORTACIÓN A CSV
    # ------------------------------------------------------------------

    def get_equity_curve(self) -> pd.DataFrame:
        """Devuelve el equity curve como DataFrame."""
        return pd.DataFrame(self._equity_records)

    def get_trade_log(self) -> pd.DataFrame:
        """Devuelve el trade log completo (BUY + SELL) como DataFrame."""
        return pd.DataFrame(self._trade_records)

    def save_results(self, output_dir: str = "data/results") -> None:
        """
        Guarda ambos registros en CSV listos para el Sprint 7.

        Args:
            output_dir (str): Directorio de salida. Se crea si no existe.
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        equity_path = f"{output_dir}/equity_curve.csv"
        trades_path = f"{output_dir}/trade_log.csv"

        self.get_equity_curve().to_csv(equity_path, index=False)
        self.get_trade_log().to_csv(trades_path, index=False)

        logger.info(f"Equity Curve guardada en: {equity_path}")
        logger.info(f"Trade Log guardado en:    {trades_path}")

    # ------------------------------------------------------------------
    # RESUMEN ESTADÍSTICO
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """
        Calcula y devuelve las métricas de rendimiento principales.

        Returns:
            dict: {
                total_return_pct, max_drawdown_pct,
                total_trades, win_rate_pct,
                avg_win, avg_loss, profit_factor
            }
        """
        equity_df = self.get_equity_curve()
        trades_df = self.get_trade_log()

        if equity_df.empty:
            return {}

        # Operaciones cerradas: aquellas con net_pnl calculado (tienen exit_price)
        if trades_df.empty or "net_pnl" not in trades_df.columns:
            closed = pd.DataFrame()
        total_return = equity_df["cumulative_return_pct"].iloc[-1] if not equity_df.empty else 0.0
        max_dd       = equity_df["drawdown_pct"].min() if not equity_df.empty else 0.0

        if self._trade_records:
            closed = pd.DataFrame(self._trade_records)
            closed = closed[closed["net_pnl"].notna()]
            total_trades = len(closed)
            wins   = closed[closed["net_pnl"] >= 0]
            losses = closed[closed["net_pnl"] < 0]
            
            win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0.0
            avg_win  = wins["net_pnl"].mean() if not wins.empty else 0.0
            avg_loss = losses["net_pnl"].mean() if not losses.empty else 0.0
            
            gross_profit = wins["net_pnl"].sum() if not wins.empty else 0.0
            gross_loss   = abs(losses["net_pnl"].sum()) if not losses.empty else 0.0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else (99.9 if gross_profit > 0 else 0.0)
        else:
            total_trades = 0
            win_rate = 0.0
            avg_win = 0.0
            avg_loss = 0.0
            profit_factor = 0.0

        stats = {
            "total_return_pct": round(total_return, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "total_trades":     total_trades,
            "win_rate_pct":     round(win_rate, 1),
            "avg_win_usd":      round(avg_win, 2),
            "avg_loss_usd":     round(avg_loss, 2),
            "profit_factor":    round(profit_factor, 2),
        }

        logger.info("--- RESUMEN ESTADÍSTICO ---")
        for k, v in stats.items():
            logger.info(f"  {k:25s}: {v}")

        return stats
