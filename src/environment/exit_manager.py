"""
Módulo de Gestión de Salidas del Mercado (Exit Manager).

Evalúa diariamente las posiciones abiertas de la cartera para determinar
si alguna debe cerrarse por haber tocado su Stop Loss, Take Profit o
por activación del Time-Stop (triple barrera — López de Prado, 2018).

Filosofía del módulo:
    La gestión de ENTRADAS la hace el PortfolioAgent + XGBoost.
    La gestión de SALIDAS es puramente matemática y temporal:

        - Stop Loss (SL): Si el precio baja hasta el nivel de invalidación,
          la tesis de la operación es errónea → vender para limitar pérdidas.
        - Take Profit (TP): Si el precio sube al objetivo → realizar ganancias.
        - Time-Stop: Si la posición lleva demasiado tiempo abierta sin
          moverse, el capital está "atrapado" → vender para liberar liquidez.

Referencias:
    López de Prado, M. (2018). Advances in Financial Machine Learning.
    John Wiley & Sons. (Capítulo 3: The Triple-Barrier Method).
"""

import logging
from datetime import datetime, timedelta

import pandas as pd

from src.environment.portfolio import Portfolio

logger = logging.getLogger("ExitManager")


class ExitManager:
    """
    Gestor de salidas que evalúa las posiciones abiertas cada día y
    decide si deben cerrarse por SL, TP o Time-Stop.

    Para funcionar, cada posición registrada en el Portfolio debe llevar
    los siguientes campos adicionales al número de acciones:
        - 'entry_price'  (float): Precio al que se compró.
        - 'stop_loss'    (float): Nivel de precio de Stop Loss.
        - 'take_profit'  (float): Nivel de precio de Take Profit.
        - 'entry_date'   (datetime): Fecha en la que se abrió la posición.
    """

    def __init__(
        self,
        portfolio: Portfolio,
        time_stop_days: int = 30,
    ):
        """
        Inicializa el Exit Manager.

        Args:
            portfolio (Portfolio): Instancia de la billetera virtual. Las
                                   ventas se ejecutarán directamente sobre ella.
            time_stop_days (int): Días máximos que una posición puede estar
                                  abierta sin llegar a TP o SL (por defecto 30).
        """
        self.portfolio = portfolio
        self.time_stop_days = time_stop_days

    def evaluate_exits(
        self,
        current_date: datetime,
        current_prices: dict,
        open_positions_meta: dict,
    ) -> list:
        """
        Método principal. Evalúa TODAS las posiciones abiertas y cierra
        las que hayan tocado SL, TP o Time-Stop.

        El orden de prioridad es: SL > TP > Time-Stop.
        (Si en la misma vela se tocan SL y TP, gana el SL — peor caso).

        Args:
            current_date (datetime): Fecha actual de la simulación.
            current_prices (dict): {ticker: precio_actual_del_dia}.
            open_positions_meta (dict): Metadatos de cada posición abierta.
                Estructura esperada:
                {
                    'AAPL': {
                        'quantity':    100,
                        'entry_price': 150.0,
                        'stop_loss':   145.0,
                        'take_profit': 165.0,
                        'entry_date':  datetime(2024, 1, 10),
                    },
                    ...
                }

        Returns:
            list: Lista de diccionarios con las salidas ejecutadas.
                  Cada diccionario contiene: ticker, reason, price, quantity.
        """
        closed_positions = []

        # Iteramos sobre una copia de las claves para poder modificar el dict
        for ticker in list(open_positions_meta.keys()):
            meta = open_positions_meta[ticker]
            current_price = current_prices.get(ticker)

            # Si no hay precio de mercado para este ticker, lo saltamos
            if current_price is None:
                logger.warning(
                    f"[ExitManager] Sin precio de mercado para {ticker}. "
                    "Posición no evaluada."
                )
                continue

            quantity    = meta.get("quantity", 0)
            stop_loss   = meta.get("stop_loss")
            take_profit = meta.get("take_profit")
            entry_date  = meta.get("entry_date")
            entry_price = meta.get("entry_price", 0.0)

            if quantity <= 0:
                continue

            close_reason = None

            # ------------------------------------------------------------------
            # LÓGICA BREAK EVEN (Alineada con backtester.py)
            # Si el precio alcanza +1R de beneficio, el SL se mueve a Entry.
            # ------------------------------------------------------------------
            be_activated = meta.get("be_activated", False)
            if not be_activated and stop_loss is not None:
                riesgo = entry_price - stop_loss
                if riesgo > 0 and current_price >= (entry_price + (riesgo * 2)):
                    meta["be_activated"] = True
                    be_activated = True
                    logger.info(f"[ExitManager] Break Even activado para {ticker} (+2R)")

            eff_stop_loss = entry_price if be_activated else stop_loss

            # ------------------------------------------------------------------
            # BARRERA 1: Stop Loss / Break Even
            # Si el precio actual toca o baja del nivel de SL, cerramos.
            # ------------------------------------------------------------------
            if eff_stop_loss is not None and current_price <= eff_stop_loss:
                close_reason = "BREAK_EVEN" if be_activated else "STOP_LOSS"

            # ------------------------------------------------------------------
            # BARRERA 2: Take Profit
            # Si el precio actual toca o supera el nivel de TP, cerramos.
            # (Solo si no se activó ya el SL — SL tiene prioridad)
            # ------------------------------------------------------------------
            elif take_profit is not None and current_price >= take_profit:
                close_reason = "TAKE_PROFIT"

            # ------------------------------------------------------------------
            # BARRERA 3: Time-Stop (triple barrera — López de Prado, 2018)
            # Si la posición lleva más de N días sin llegar a TP ni SL, cerramos.
            # ------------------------------------------------------------------
            elif entry_date is not None:
                days_held = (current_date - entry_date).days
                if days_held >= self.time_stop_days:
                    close_reason = "TIME_STOP"

            # Si se activó alguna barrera, ejecutar la venta
            if close_reason:
                self._execute_exit(
                    ticker=ticker,
                    quantity=quantity,
                    current_price=current_price,
                    reason=close_reason,
                    entry_price=entry_price,
                )
                closed_positions.append({
                    "ticker":        ticker,
                    "reason":        close_reason,
                    "exit_price":    current_price,
                    "entry_price":   entry_price,
                    "quantity":      quantity,
                    "pnl":           (current_price - entry_price) * quantity,
                    "date":          current_date,
                })

                # Eliminar los metadatos de la posición cerrada
                del open_positions_meta[ticker]

        return closed_positions

    def _execute_exit(
        self,
        ticker: str,
        quantity: float,
        current_price: float,
        reason: str,
        entry_price: float,
    ) -> None:
        """
        Ejecuta la orden de venta en el Portfolio y registra el resultado.

        Args:
            ticker       (str):   Ticker del activo a vender.
            quantity     (float): Número de acciones a vender.
            current_price(float): Precio actual de mercado (precio de ejecución).
            reason       (str):   Motivo de la salida ('STOP_LOSS', 'TAKE_PROFIT', 'TIME_STOP').
            entry_price  (float): Precio de entrada (para calcular el P&L del log).
        """
        pnl = (current_price - entry_price) * quantity
        result_str = "WIN" if pnl >= 0 else "LOSS"

        logger.info(
            f"[ExitManager] {reason} | {ticker} | "
            f"Qty={quantity} | Precio={current_price:.2f} | "
            f"P&L={pnl:+.2f}$ ({result_str})"
        )

        # Delegar la ejecución real al Portfolio (aplica slippage + comisiones)
        self.portfolio.execute_trade(
            ticker=ticker,
            quantity=quantity,
            price=current_price,
            side="SELL",
        )
