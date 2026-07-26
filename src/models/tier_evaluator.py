"""
Módulo para la evaluación algorítmica de oportunidades de inversión.

Traduce las características técnicas y de Price Action precalculadas
(por src/features/technical.py y src/features/patterns.py) en una
clasificación jerárquica de calidad de entrada: los Tiers de probabilidad.

FILTRO BASE MULTI-TIMEFRAME (obligatorio para todos los Tiers)
===============================================================
Antes de evaluar ningún patrón de Price Action, el sistema comprueba
que la tendencia sea alcista en las tres temporalidades jerárquicas:

    1W : slope_SMA_200_1W > 0    (tendencia macro positiva)
    1D : close_1D > SMA_200_1D   (precio sobre la media de largo plazo diaria)
    4H : close    > SMA_200      (precio sobre la media operativa)

Si alguna de estas tres condiciones falla, la señal NO se genera,
independientemente de lo válido que sea el patrón en 4H.

Tier A (Probabilidad Alta):
    - Retroceso ordenado a favor de la tendencia.
    - El precio visita y rechaza el nivel Fibonacci 61.8%.
    - Cierre próximo a la SMA 200.
    - Incluye señales con divergencia RSI (antes Tier A*).

Tier B (Probabilidad Media):
    - Doble Suelo / Re-testeo de una zona de valor.
    - El precio respeta el último fractal de soporte algorítmico.
    - Se confirma el rechazo mediante un Wick Reclaim alcista.

Tier C (Probabilidad Baja / Confirmación Tardía):
    - Breakout direccional: el precio cierra por encima del último
      fractal de resistencia confirmado.
"""

import logging
import pandas as pd
import numpy as np

logger = logging.getLogger("TierEvaluator")


class TierEvaluator:
    """
    Evalúa y clasifica cada vela del DataFrame en un Tier de probabilidad
    utilizando operaciones vectorizadas (sin bucles for) para escalar a
    millones de filas sin degradar el rendimiento.

    La asignación se realiza de mayor a menor prioridad: si una vela cumple
    La asignación se realiza de mayor a menor prioridad:
    el sistema tiene 3 Tiers: A (alta), B (media) y C (baja).
    """

    def __init__(
        self,
        fib_tolerance: float = 3.0,
        max_sma_dist: float = 5.0,
        slope_1w_min: float = 0.0,
    ):
        """
        Inicializa el evaluador con los umbrales de las condiciones de entrada.

        Args:
            fib_tolerance (float): Distancia porcentual máxima al nivel Fib 61.8%
                                   para considerar que el precio lo está "rechazando".
                                   Por defecto 3.0%.
            max_sma_dist  (float): Distancia porcentual máxima a la SMA 200 en 4H
                                   para que la entrada no esté sobreextendida.
                                   Por defecto 5.0%.
            slope_1w_min  (float): Pendiente mínima de la SMA 200 semanal (%).
                                   Valores > 0 exigen tendencia macro positiva.
                                   Por defecto 0.0 (plana o subiendo).
        """
        self.fib_tolerance = fib_tolerance
        self.max_sma_dist  = max_sma_dist
        self.slope_1w_min  = slope_1w_min

    def evaluate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Asigna el Tier correspondiente a cada fila del DataFrame.

        Lee las columnas de features técnicas y de Price Action ya calculadas
        y devuelve el DataFrame original con una nueva columna 'Tier' que
        puede contener los valores 'A', 'B', 'C' o None (sin señal).

        Args:
            df (pd.DataFrame): DataFrame con todas las features de los módulos
                               technical.py y patterns.py ya calculadas.

        Returns:
            pd.DataFrame: DataFrame con la columna 'Tier' añadida.
        """
        df_out = df.copy()

        # -----------------------------------------------------------------
        # FILTRO BASE MULTI-TIMEFRAME
        # La estrategia exige confirmación en las tres temporalidades:
        #   1W: slope_SMA_200_1W > slope_1w_min  (tendencia macro positiva)
        #   1D: close_1D > SMA_200_1D            (tendencia de medio plazo)
        #   4H: close    > SMA_200               (tendencia operativa)
        #
        # Si las columnas de 1D/1W no existen aún (el merger MTF no se ha
        # ejecutado), el sistema opera en modo degradado con solo la condición
        # de 4H y emite un aviso en el log para recordar que el filtro MTF
        # completo está pendiente de implementar.
        # -----------------------------------------------------------------

        # Condición 4H (siempre presente)
        cond_4h = df_out["close"] > df_out.get(
            "SMA_200",
            pd.Series([float("inf")] * len(df_out), index=df_out.index)
        ).fillna(float("inf"))

        # Condición 1D: el precio diario debe estar sobre su SMA 200
        if "close_1D" in df_out.columns and "SMA_200_1D" in df_out.columns:
            cond_1d = df_out["close_1D"] > df_out["SMA_200_1D"].fillna(float("inf"))
        else:
            logger.warning(
                "[MTF] Columnas 'close_1D' / 'SMA_200_1D' no encontradas. "
                "Ejecuta el merger multi-timeframe para activar el filtro 1D. "
                "Operando en modo degradado (solo 4H)."
            )
            cond_1d = pd.Series([True] * len(df_out), index=df_out.index)

        # Condición 1W: la pendiente de la SMA 200 semanal debe ser positiva
        if "slope_SMA_200_1W" in df_out.columns:
            cond_1w = df_out["slope_SMA_200_1W"].fillna(-999) > self.slope_1w_min
        else:
            logger.warning(
                "[MTF] Columna 'slope_SMA_200_1W' no encontrada. "
                "Ejecuta el merger multi-timeframe para activar el filtro 1W. "
                "Operando en modo degradado (solo 4H)."
            )
            cond_1w = pd.Series([True] * len(df_out), index=df_out.index)

        # Filtro base final: las tres condiciones deben cumplirse simultáneamente
        filtro_base = cond_4h & cond_1d & cond_1w

        logger.info(
            "Filtro base MTF | 4H OK=%d | 1D OK=%d | 1W OK=%d | TOTAL OK=%d / %d velas",
            cond_4h.sum(), cond_1d.sum(), cond_1w.sum(),
            filtro_base.sum(), len(df_out),
        )

        # -----------------------------------------------------------------
        # PRE-CÁLCULOS VECTORIZADOS
        # -----------------------------------------------------------------

        # Cercanía al nivel de Fibonacci 61.8% de retroceso
        dist_fib_618 = df_out.get(
            "dist_fib_retr_618",
            pd.Series([100.0] * len(df_out), index=df_out.index)
        )
        near_fib_618 = dist_fib_618.abs() < self.fib_tolerance

        # Cercanía a la SMA 200 (filtro de sobreextensión)
        dist_sma = df_out.get(
            "dist_SMA_200",
            pd.Series([100.0] * len(df_out), index=df_out.index)
        )
        near_sma = dist_sma.abs() < self.max_sma_dist

        # Divergencia alcista del RSI confirmada
        bullish_div = df_out.get("is_bullish_divergence", pd.Series([0]*len(df_out), index=df_out.index)) == 1

        # Wick Reclaim: rechazo agresivo del precio (mecha inferior > 60% del rango)
        wick_reclaim = df_out.get("is_bullish_wick_reclaim", pd.Series([0]*len(df_out), index=df_out.index)) == 1

        # Fractal de soporte (suelo algorítmico confirmado)
        is_support = df_out.get("is_support_fractal", pd.Series([0]*len(df_out), index=df_out.index)) == 1

        # Breakout (Tier C): el cierre supera el último fractal de resistencia
        if "is_resistance_fractal" in df_out.columns and "high" in df_out.columns:
            # Nivel del último máximo fractal conocido (propagado con ffill)
            last_resistance = df_out["high"].where(df_out["is_resistance_fractal"] == 1).ffill()
            
            # Breakout confirmado: cierre actual > resistencia y cierre anterior <= resistencia
            cross_up = (df_out["close"] > last_resistance) & (df_out["close"].shift(1) <= last_resistance.shift(1))
            
            # Filtro de Vela Escapada en Ruptura: No perseguir el precio si cerró muy por encima de la resistencia
            # Máxima distancia permitida: 1.5% del valor de la resistencia
            not_escaped_breakout = ((df_out["close"] - last_resistance) / last_resistance) <= 0.015
            
            # Filtro de Resistencia Madura (Edad del Fractal):
            # Eliminamos fractales de 1-2 velas (ruido puro). Exigimos mínimo 5 velas
            # de antigüedad para asegurar que el nivel fue "testado" antes de romperse.
            # (La criba macro de 30 velas en patterns.py ya garantiza la importancia
            # estructural, este filtro añade el requisito mínimo de tiempo probado.)
            bars = pd.Series(np.arange(len(df_out)), index=df_out.index)
            last_res_bar = bars.where(df_out["is_resistance_fractal"] == 1).ffill()
            fractal_age = bars - last_res_bar
            mature_resistance = fractal_age >= 5

            # -----------------------------------------------------------------
            # FILTRO "RESISTENCIA CONSUMIDA" (Anti-reentrada en la misma zona)
            # -----------------------------------------------------------------
            raw_breakout = cross_up & not_escaped_breakout & mature_resistance
            
            # Construimos la serie final filtrando re-entradas en el mismo nivel
            is_breakout_list = [False] * len(df_out)
            consumed_resistance = None  # Precio de la resistencia ya "consumida"

            for i, (idx, val) in enumerate(raw_breakout.items()):
                if val:  # Hay un breakout potencial
                    res_level = last_resistance.iloc[i]
                    if consumed_resistance is None or abs(res_level - consumed_resistance) / consumed_resistance > 0.01:
                        # Nuevo nivel de resistencia (>1% alejado del consumido) → señal válida
                        is_breakout_list[i] = True
                        consumed_resistance = res_level
                    # Si es el mismo nivel consumido → ignorar (re-entrada bloqueada)
                elif consumed_resistance is not None:
                    # Si el precio cae por debajo de la resistencia consumida, la "liberamos"
                    close_i = df_out["close"].iloc[i]
                    if close_i < consumed_resistance * 0.99:  # 1% por debajo → liberada
                        consumed_resistance = None

            is_breakout = pd.Series(is_breakout_list, index=df_out.index)
        else:
            is_breakout = pd.Series([False] * len(df_out), index=df_out.index)

        # -----------------------------------------------------------------
        # MEJORAS DE ENTRADA (Filtros de Excepciones SMA)
        # -----------------------------------------------------------------
        
        # Filtro de Vela Escapada (Runaway Candle)
        # Aseguramos que no entramos si el precio ya se ha escapado muy lejos del soporte de la SMA.
        # Máxima distancia permitida: 1.5 * ATR (si cierra más lejos, se descarta el Tier).
        atr = df_out.get("ATR_14", pd.Series([10.0] * len(df_out), index=df_out.index))
        sma_200 = df_out.get("SMA_200", pd.Series([0.0] * len(df_out), index=df_out.index))
        not_escaped = (df_out["close"] - sma_200) <= (1.5 * atr)

        # Filtro Pullback Válido (De arriba hacia abajo)
        # Contamos cuántas de las últimas 20 velas cerraron por debajo de la SMA.
        # Si son <= 3, significa que el precio venía sólidamente navegando por encima
        # y este toque es un retroceso válido, no un cruce de tendencia bajista a alcista.
        is_below_sma = df_out["close"] < sma_200
        candles_below = is_below_sma.rolling(window=20, min_periods=1).sum()
        valid_pullback = candles_below <= 6

        # -----------------------------------------------------------------
        # LÓGICA DE TIERS (Asignación por prioridad descendente)
        # Tier A* eliminado: sus señales se absorben en Tier A
        # Tier A: Alta probabilidad — retroceso a la SMA 200 (Fibonacci 61.8% desactivado)
        # cond_A = cond_4h & near_fib_618 & near_sma & valid_pullback & not_escaped
        cond_A_raw = cond_4h & near_sma & not_escaped # valid_pullback desactivado
        
        # Cooldown de 12 velas (aprox 2 días en 4H) para evitar entrar repetidas veces en el mismo punto
        cooldown_period = 12
        cond_A_shifted = cond_A_raw.shift(1, fill_value=False)
        recent_signals = cond_A_shifted.rolling(window=cooldown_period, min_periods=1).max()
        cond_A = cond_A_raw & (recent_signals == 0)

        # Tier B: Media probabilidad — doble suelo en zona de soporte estructural
        cond_B = cond_4h & is_support

        # Tier C: Breakout puro sobre resistencia madura
        # Sin filtros de MTF diario/semanal para capturar rupturas tempranas
        cond_C = cond_4h & is_breakout

        # Asignar Tier
        df_out["Tier"] = None
        df_out.loc[cond_C, "Tier"] = "C"
        df_out.loc[cond_B, "Tier"] = "B"
        df_out.loc[cond_A, "Tier"] = "A"

        logger.info(
            "Tiers evaluados | A=%d | B=%d | C=%d | Sin señal=%d",
            cond_A.sum(), cond_B.sum(), cond_C.sum(),
            df_out["Tier"].isna().sum(),
        )
        return df_out
