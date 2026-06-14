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

Tier A* (Probabilidad Extrema):
    - Falsa ruptura de soportes + barrido de liquidez (Wick Reclaim).
    - Rechazo del nivel de retroceso de Fibonacci 61.8%.
    - El precio recupera y cierra cerca de la SMA 200.
    - Exige divergencia alcista confirmada en el RSI.

Tier A (Probabilidad Alta):
    - Retroceso ordenado a favor de la tendencia.
    - El precio visita y rechaza el nivel Fibonacci 61.8%.
    - Cierre próximo a la SMA 200, sin exigir wick agresivo ni divergencia.

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
    el criterio de Tier A*, se le asigna A* aunque también cumpla el de A.
    """

    def __init__(
        self,
        fib_tolerance: float = 1.5,
        max_sma_dist: float = 5.0,
        slope_1w_min: float = 0.0,
    ):
        """
        Inicializa el evaluador con los umbrales de las condiciones de entrada.

        Args:
            fib_tolerance (float): Distancia porcentual máxima al nivel Fib 61.8%
                                   para considerar que el precio lo está "rechazando".
                                   Por defecto 1.5%.
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
        puede contener los valores 'A*', 'A', 'B', 'C' o None (sin señal).

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
            is_breakout = (
                (df_out["close"] > last_resistance) &
                (df_out["close"].shift(1) <= last_resistance.shift(1))
            )
        else:
            is_breakout = pd.Series([False] * len(df_out), index=df_out.index)

        # -----------------------------------------------------------------
        # LÓGICA DE TIERS (Asignación por prioridad descendente)
        # -----------------------------------------------------------------

        # Tier A*: Mayor exigencia — requiere TODOS los filtros simultáneos
        cond_A_star = filtro_base & wick_reclaim & near_fib_618 & bullish_div & near_sma

        # Tier A: Alta probabilidad — retroceso profundo sin exigir divergencia
        cond_A = filtro_base & near_fib_618 & near_sma & ~cond_A_star

        # Tier B: Media probabilidad — doble suelo con rechazo en soporte
        cond_B = filtro_base & is_support & wick_reclaim & near_sma & ~cond_A_star & ~cond_A

        # Tier C: Baja probabilidad — confirmación tardía vía breakout
        cond_C = filtro_base & is_breakout & ~cond_A_star & ~cond_A & ~cond_B

        # Asignar Tier (de menor a mayor prioridad para que los mejores sobreescriban)
        df_out["Tier"] = None
        df_out.loc[cond_C,      "Tier"] = "C"
        df_out.loc[cond_B,      "Tier"] = "B"
        df_out.loc[cond_A,      "Tier"] = "A"
        df_out.loc[cond_A_star, "Tier"] = "A*"

        logger.info(
            "Tiers evaluados | A*=%d | A=%d | B=%d | C=%d | Sin señal=%d",
            cond_A_star.sum(), cond_A.sum(), cond_B.sum(), cond_C.sum(),
            df_out["Tier"].isna().sum(),
        )
        return df_out
