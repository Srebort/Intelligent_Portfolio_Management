"""
Módulo del Agente Autónomo de Gestión de Cartera.

Implementa el PortfolioAgent, el cerebro orquestador que integra:
  - Filtrado de señales mediante Machine Learning (Sprint 4).
  - Gestión matemática del riesgo mediante el RiskManager (Sprint 5).
  - Reglas defensivas de rebalanceo dinámico:
      · Filtro VIX (Pánico Macro)             — Ang & Bekaert (2002)
      · Kill-Switch Global (Max Drawdown)     — Black & Jones (1987)
      · Time-Stop (Triple Barrera)            — López de Prado (2018)
      · Kelly Fraccional                      — MacLean, Thorp & Ziemba (2011)
      · Límite de Correlación (Markowitz)     — Markowitz (1952)
      · Filtro de Amplitud de Mercado (Breadth)— Faber (2007)
"""

import logging
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd

from src.environment.risk_manager import RiskManager

logger = logging.getLogger("PortfolioAgent")
logging.basicConfig(level=logging.INFO, format="%(message)s")


class PortfolioAgent:
    """
    Agente autónomo que decide qué operaciones abrir, cuándo cerrarlas y
    cuándo suspender toda actividad según el régimen del mercado.

    Parámetros de rebalanceo configurables:
        prob_threshold      : Probabilidad mínima de la IA para aprobar una señal (0.75).
        vix_panic_threshold : Nivel de VIX a partir del cual se bloquean nuevas compras (30).
        max_drawdown_limit  : Caída máxima del equity desde máximos antes del Kill-Switch (-0.10).
        kill_switch_days    : Días de enfriamiento tras activar el Kill-Switch (30 días = 1 mes).
        time_stop_days      : Días máximos que una posición puede estar abierta sin resultado (30).
        time_stop_min_return: Retorno mínimo aceptable antes de forzar el cierre por Time-Stop (-0.03).
        max_correlation     : Correlación de Pearson máxima permitida entre activos simultáneos (0.8).
        min_breadth_pct     : % mínimo de activos sobre su SMA200 para permitir nuevas compras (0.40).
    """

    def __init__(
        self,
        model_path: str = "models/best_model.pkl",
        scaler_path: str = "models/scaler.pkl",
        prob_threshold: float = 0.75,
        vix_panic_threshold: float = 30.0,
        max_drawdown_limit: float = -0.10,
        kill_switch_days: int = 30,
        time_stop_days: int = 30,
        time_stop_min_return: float = -0.03,
        max_correlation: float = 0.8,
        min_breadth_pct: float = 0.40,
    ):
        """
        Inicializa el Agente de Cartera cargando los modelos de IA y configurando
        los parámetros de todas las reglas de rebalanceo dinámico.
        """
        self.prob_threshold = prob_threshold
        self.vix_panic_threshold = vix_panic_threshold
        self.max_drawdown_limit = max_drawdown_limit
        self.kill_switch_days = kill_switch_days
        self.time_stop_days = time_stop_days
        self.time_stop_min_return = time_stop_min_return
        self.max_correlation = max_correlation
        self.min_breadth_pct = min_breadth_pct

        # Estado interno del Kill-Switch
        self.peak_equity: float = 0.0
        self.trading_halted: bool = False
        self.halt_until: datetime | None = None

        # Historial de operaciones cerradas (para Kelly Fraccional)
        self.trade_log: list = []
        self.kelly_factor: float = 1.0  # Multiplicador sobre el riesgo base (1.0 = 100%)

        # Historial de precios por ticker para cálculo de correlaciones
        self.price_history: dict = {}

        self.risk_manager = RiskManager()

        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            logger.info(" Modelo y scaler cargados correctamente en el Agente.")
        except Exception as e:
            logger.error(f" Error cargando los modelos en {model_path}: {e}")
            self.model = None
            self.scaler = None

    # -----------------------------------------------------------------------
    # REGLA 1: FILTRO VIX (Pánico Macroeconómico)
    # Ref: Ang, A. & Bekaert, G. (2002). International Asset Allocation with
    #      Regime Shifts. The Review of Financial Studies, 15(4), 1137-1187.
    # -----------------------------------------------------------------------
    def _is_vix_panic(self, vix_value: float) -> bool:
        """
        Detecta si el mercado está en un régimen de pánico macro.

        Returns:
            bool: True si VIX > umbral de pánico (bloquear compras).
        """
        if vix_value > self.vix_panic_threshold:
            logger.warning(
                f"  [VIX] Pánico macro detectado: VIX={vix_value:.1f} > {self.vix_panic_threshold}. "
                "Bloqueando nuevas compras."
            )
            return True
        return False

    # -----------------------------------------------------------------------
    # REGLA 2: KILL-SWITCH GLOBAL (Max Drawdown — Portfolio Insurance)
    # Ref: Black, F. & Jones, R. (1987). Simplifying Portfolio Insurance.
    #      The Journal of Portfolio Management, 14(1), 48-51.
    # -----------------------------------------------------------------------
    def check_kill_switch(self, current_equity: float, current_date: datetime) -> bool:
        """
        Comprueba si el drawdown acumulado ha superado el límite máximo permitido.
        Si se activa, suspende toda nueva operativa durante 'kill_switch_days' días (1 mes).

        Args:
            current_equity (float): Equity total de la cartera en la fecha actual.
            current_date (datetime): Fecha actual de la simulación.

        Returns:
            bool: True si el trading está suspendido.
        """
        # Desactivar el halt si ya pasó el periodo de enfriamiento
        if self.trading_halted and self.halt_until and current_date >= self.halt_until:
            self.trading_halted = False
            self.halt_until = None
            self.peak_equity = current_equity  # Resetear el pico tras el enfriamiento
            logger.info(" [Kill-Switch] Período de enfriamiento terminado. Operativa reanudada.")

        if self.trading_halted:
            logger.warning(
                f" [Kill-Switch] Operativa suspendida hasta {self.halt_until.strftime('%Y-%m-%d')}."
            )
            return True

        # Actualizar el pico histórico de equity
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

        # Calcular drawdown actual
        if self.peak_equity > 0:
            drawdown = (current_equity - self.peak_equity) / self.peak_equity
            if drawdown <= self.max_drawdown_limit:
                self.trading_halted = True
                self.halt_until = current_date + timedelta(days=self.kill_switch_days)
                logger.critical(
                    f" [Kill-Switch] ACTIVADO. Drawdown={drawdown*100:.1f}% "
                    f"< {self.max_drawdown_limit*100:.0f}%. "
                    f"Operativa suspendida 1 mes hasta {self.halt_until.strftime('%Y-%m-%d')}."
                )
                return True

        return False

    # -----------------------------------------------------------------------
    # REGLA 3: TIME-STOP (Triple Barrera)
    # Ref: López de Prado, M. (2018). Advances in Financial Machine Learning.
    #      John Wiley & Sons. Capítulo 3: The Triple-Barrier Method.
    # -----------------------------------------------------------------------
    def get_time_stop_closes(
        self, open_positions: dict, current_prices: dict, current_date: datetime
    ) -> list:
        """
        Evalúa las posiciones abiertas y genera órdenes de venta para las que
        hayan superado el Time-Stop (1 mes de inactividad O retorno mínimo negativo).

        Criterios de cierre anticipado:
            1. La posición lleva más de 'time_stop_days' días abierta.
            2. O bien, el retorno actual es inferior a 'time_stop_min_return' (-3%)
               sin haber tocado el Stop Loss definido por el RiskManager.

        Args:
            open_positions (dict): {ticker: {'quantity': N, 'entry_price': P, 'entry_date': D}}.
            current_prices (dict): {ticker: precio_actual}.
            current_date (datetime): Fecha actual de la simulación.

        Returns:
            list: Lista de tickers a cerrar por Time-Stop.
        """
        tickers_to_close = []

        for ticker, position in open_positions.items():
            entry_date = position.get("entry_date")
            entry_price = position.get("entry_price", 0.0)
            current_price = current_prices.get(ticker, 0.0)

            if entry_date is None or current_price == 0.0:
                continue

            days_held = (current_date - entry_date).days
            current_return = (current_price - entry_price) / entry_price if entry_price > 0 else 0.0

            # Criterio 1: Demasiado tiempo abierta (1 mes)
            if days_held >= self.time_stop_days:
                logger.info(
                    f"⏱️  [Time-Stop] {ticker}: {days_held} días abierta. "
                    f"Retorno actual: {current_return*100:.1f}%. Cerrando por inactividad."
                )
                tickers_to_close.append(ticker)

            # Criterio 2: Retorno mínimo negativo sin haber tocado el SL
            elif current_return <= self.time_stop_min_return:
                logger.info(
                    f" [Time-Stop] {ticker}: Retorno {current_return*100:.1f}% "
                    f"< {self.time_stop_min_return*100:.0f}% mínimo. Cerrando posición deteriorada."
                )
                tickers_to_close.append(ticker)

        return tickers_to_close

    # -----------------------------------------------------------------------
    # REGLA 4: KELLY FRACCIONAL
    # Ref: MacLean, Thorp & Ziemba (2011). The Kelly Capital Growth Investment
    #      Criterion: Theory and Practice. World Scientific.
    # -----------------------------------------------------------------------
    def update_kelly_factor(self, lookback: int = 50) -> None:
        """
        Recalcula el factor multiplicador de riesgo (Half-Kelly) basándose en
        las últimas 'lookback' operaciones del historial.

        Si el Win Rate reciente cae, el factor baja y el agente arriesga menos.
        Si el modelo está en racha positiva, el factor sube hacia 1.0.

        Args:
            lookback (int): Número de operaciones recientes a evaluar (por defecto 50).
        """
        if len(self.trade_log) < lookback:
            return  # No hay suficiente historial para recalcular

        recent_trades = self.trade_log[-lookback:]
        wins = [t for t in recent_trades if t.get("result") == "WIN"]
        losses = [t for t in recent_trades if t.get("result") == "LOSS"]

        if not wins or not losses:
            return

        win_rate = len(wins) / len(recent_trades)
        avg_win = np.mean([t["pnl"] for t in wins])
        avg_loss = abs(np.mean([t["pnl"] for t in losses]))

        if avg_loss == 0:
            return

        payoff_ratio = avg_win / avg_loss

        # Fórmula de Kelly: f* = W - (1-W)/R
        kelly_pure = win_rate - (1 - win_rate) / payoff_ratio

        # Half-Kelly para mayor conservadurismo
        self.kelly_factor = max(0.25, min(1.0, kelly_pure * 0.5))

        logger.info(
            f" [Kelly] WinRate={win_rate*100:.1f}% | Payoff={payoff_ratio:.2f} | "
            f"Kelly puro={kelly_pure*100:.1f}% | Half-Kelly={self.kelly_factor*100:.1f}%"
        )

        # Aplicar el factor al capital del RiskManager
        self.risk_manager.capital = self.risk_manager.capital * self.kelly_factor

    # -----------------------------------------------------------------------
    # REGLA 5: LÍMITE DE CORRELACIÓN (Teoría Moderna de Carteras)
    # Ref: Markowitz, H. (1952). Portfolio Selection. The Journal of Finance,
    #      7(1), 77-91.
    # -----------------------------------------------------------------------
    def _is_too_correlated(self, new_ticker: str, approved_tickers: list) -> bool:
        """
        Comprueba si el nuevo activo está demasiado correlacionado con los
        activos ya aprobados para comprar en este mismo día.

        Args:
            new_ticker (str): Ticker de la nueva señal a evaluar.
            approved_tickers (list): Lista de tickers ya aprobados este día.

        Returns:
            bool: True si la correlación supera el umbral (rechazar la señal).
        """
        if not approved_tickers or new_ticker not in self.price_history:
            return False

        new_prices = pd.Series(self.price_history.get(new_ticker, []))
        if len(new_prices) < 20:
            return False

        for existing_ticker in approved_tickers:
            existing_prices = pd.Series(self.price_history.get(existing_ticker, []))
            if len(existing_prices) < 20:
                continue

            min_len = min(len(new_prices), len(existing_prices))
            correlation = new_prices.iloc[-min_len:].corr(existing_prices.iloc[-min_len:])

            if not np.isnan(correlation) and abs(correlation) >= self.max_correlation:
                logger.info(
                    f"🔗 [Correlación] {new_ticker} vs {existing_ticker}: "
                    f"correlación={correlation:.2f} >= {self.max_correlation}. Señal descartada."
                )
                return True

        return False

    # -----------------------------------------------------------------------
    # REGLA 6: FILTRO DE AMPLITUD DE MERCADO (Market Breadth)
    # Ref: Faber, M. T. (2007). A Quantitative Approach to Tactical Asset
    #      Allocation. The Journal of Wealth Management, 10(4), 12-28.
    # -----------------------------------------------------------------------
    def _is_market_breadth_ok(self, universe_df: pd.DataFrame) -> bool:
        """
        Verifica que al menos el 'min_breadth_pct' del universo de acciones
        esté cotizando por encima de su SMA de 200 periodos.

        Args:
            universe_df (pd.DataFrame): DataFrame con columnas 'close' y 'SMA_200'
                                        para todos los activos del universo en la fecha actual.

        Returns:
            bool: True si el mercado es lo suficientemente sano para comprar.
        """
        if universe_df is None or universe_df.empty:
            return True  # Si no hay datos macro, no bloqueamos por defecto

        if "close" not in universe_df.columns or "SMA_200" not in universe_df.columns:
            return True

        above_sma200 = (universe_df["close"] > universe_df["SMA_200"]).sum()
        breadth_pct = above_sma200 / len(universe_df)

        if breadth_pct < self.min_breadth_pct:
            logger.warning(
                f" [Breadth] Solo el {breadth_pct*100:.1f}% del universo está sobre su SMA200 "
                f"(mínimo: {self.min_breadth_pct*100:.0f}%). Bloqueando nuevas compras."
            )
            return False

        return True

    # -----------------------------------------------------------------------
    # MÉTODO LEGADO (compatibilidad)
    # -----------------------------------------------------------------------
    def calculate_weights(self, predictions: dict, current_portfolio: dict, risk_metrics: dict) -> dict:
        """
        Decides capital allocation for each asset based on model predictions and risk metrics.
        (Mantenido por compatibilidad histórica)
        """
        pass

    # -----------------------------------------------------------------------
    # MÉTODO PRINCIPAL
    # -----------------------------------------------------------------------
    def process_signals(
        self,
        signals_df: pd.DataFrame,
        features_cols: list,
        current_equity: float,
        current_date: datetime | None = None,
        vix_value: float = 0.0,
        universe_df: pd.DataFrame | None = None,
    ) -> list:
        """
        Procesa las señales técnicas del día aplicando todas las reglas de
        rebalanceo dinámico antes de aprobar cualquier orden de compra.

        Flujo de decisión:
            1. Kill-Switch Global: ¿Estamos en período de enfriamiento?
            2. Filtro VIX: ¿El mercado está en pánico?
            3. Filtro Breadth: ¿El mercado es estructuralmente sano?
            4. Por cada señal:
               a. Predicción IA: ¿Probabilidad >= umbral?
               b. Filtro Correlación: ¿No está demasiado correlacionada con las aprobadas?
               c. RiskManager: ¿Cuántas acciones comprar?
            5. Ranking final por probabilidad (mayor primero).

        Args:
            signals_df     : DataFrame con las señales técnicas del día.
            features_cols  : Columnas que el modelo necesita para predecir.
            current_equity : Equity total actual de la cartera.
            current_date   : Fecha actual de la simulación.
            vix_value      : Nivel del índice VIX del día.
            universe_df    : DataFrame con close y SMA_200 de todos los activos del universo.

        Returns:
            list: Órdenes de compra aprobadas, ordenadas de mayor a menor probabilidad.
        """
        if current_date is None:
            current_date = datetime.today()

        # --- GUARDIA 1: Kill-Switch Global ---
        if self.check_kill_switch(current_equity, current_date):
            return []

        # --- GUARDIA 2: Filtro VIX ---
        if vix_value > 0 and self._is_vix_panic(vix_value):
            return []

        # --- GUARDIA 3: Filtro de Amplitud de Mercado ---
        if not self._is_market_breadth_ok(universe_df):
            return []

        if signals_df.empty or self.model is None or self.scaler is None:
            return []

        # Actualizamos el capital del RiskManager al equity real actual
        self.risk_manager.capital = current_equity
        approved_orders = []
        approved_tickers = []

        for index, row in signals_df.iterrows():
            # Extraer y escalar features para el modelo
            try:
                features = row[features_cols].to_frame().T
                features_scaled = self.scaler.transform(features)
                prob = self.model.predict_proba(features_scaled)[0][1]
            except Exception as e:
                logger.warning(f"  Error al predecir señal: {e}")
                continue

            ticker = row.get("ticker", f"Signal_{index}")

            # --- FILTRO IA: Probabilidad mínima ---
            if prob < self.prob_threshold:
                logger.info(
                    f" [IA] Señal RECHAZADA para {ticker}. "
                    f"Prob={prob*100:.1f}% < {self.prob_threshold*100:.0f}%"
                )
                continue

            # --- FILTRO CORRELACIÓN (Markowitz) ---
            if self._is_too_correlated(ticker, approved_tickers):
                continue

            logger.info(f" [IA] Señal APROBADA para {ticker} con prob={prob*100:.1f}%")

            # --- RISK MANAGER: Calcular tamaño de posición ---
            trade_params = self.risk_manager.calculate_trade_parameters(row)

            if trade_params and trade_params.get("num_acciones", 0) > 0:
                order = {
                    "ticker": ticker,
                    "action": "BUY",
                    "quantity": trade_params["num_acciones"],
                    "price": trade_params["precio_entrada"],
                    "stop_loss": trade_params["stop_loss"],
                    "entry_date": current_date,   # Para el Time-Stop
                    "probability": prob,
                    "tier": trade_params["tier"],
                }
                approved_orders.append(order)
                approved_tickers.append(ticker)
            else:
                logger.info(
                    f"  [RiskManager] {ticker} descartada: SL incorrecto o capital insuficiente."
                )

        # Ranking final: mayor probabilidad primero
        approved_orders.sort(key=lambda x: x["probability"], reverse=True)

        return approved_orders
