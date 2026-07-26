"""
Módulo para la detección de patrones de Price Action (Acción del Precio).

Este módulo traduce conceptos visuales de trading en características
(features) matemáticas estandarizadas para el modelo de Machine Learning.
Las operaciones están fuertemente vectorizadas utilizando pandas y numpy
para garantizar un rendimiento óptimo sobre millones de velas.

GARANTÍA ANTI-LOOKAHEAD BIAS:
    Al igual que en technical.py, ninguna de estas funciones accede a datos
    del futuro. Los patrones que requieren validación de velas posteriores
    (como los fractales) emiten su señal en la vela de confirmación,
    no retrospectivamente en la vela central.

Patrones implementados:
    - calculate_bill_williams_fractals() → Suelos y techos locales.
    - calculate_candlestick_patterns()   → Martillos y estrellas fugaces.
    - calculate_wick_reclaims()          → Rechazos bruscos del precio (mechas largas).
    - calculate_fibonacci_levels()       → Retrocesos y extensiones (38.2%, 61.8%).
    - add_price_action_features()        → Pipeline completo.
"""

import logging
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Configuración del sistema de logs
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("PriceActionFeatures")


# ===========================================================================
# 1. FRACTALES DE BILL WILLIAMS
# ===========================================================================

def calculate_bill_williams_fractals(
    df: pd.DataFrame,
    period: int = 2,
    high_col: str = "high",
    low_col: str = "low",
) -> pd.DataFrame:
    """
    Detecta Fractales de Bill Williams.

    Un fractal alcista (Up Fractal / Resistencia) se forma cuando una vela
    tiene un máximo superior a las N velas anteriores y N velas posteriores.
    Un fractal bajista (Down Fractal / Soporte) se forma cuando una vela
    tiene un mínimo inferior a las N velas anteriores y N velas posteriores.

    ANTI-LOOKAHEAD BIAS:
        Para un period=2, la vela actual t se usa para confirmar el fractal
        que ocurrió en t-2. La señal booleana (1 o 0) se marca en t, que es
        el momento exacto en el que el mercado (y el modelo) confirma el patrón.

    Filtro Macro (Criba de Fractales):
        Para evitar que fractales menores (ruido de 4H) sobreescriban los
        fractales estructurales mayores (de 1D o 1W), se aplica un filtro
        'macro_period'. Un fractal solo se valida si es el máximo/mínimo
        absoluto de las últimas N velas. Esto consolida los niveles.

    Args:
        df       (pd.DataFrame): DataFrame con precios.
        period   (int): Distancia a cada lado del centro. Por defecto 2.
        macro_period (int): Ventana de consolidación de fractales. Defecto 30 (~1 semana en 4H).
        high_col (str): Nombre de la columna de máximos.
        low_col  (str): Nombre de la columna de mínimos.

    Returns:
        pd.DataFrame: DataFrame con las columnas 'is_resistance_fractal' y
                      'is_support_fractal' (1 = True, 0 = False).
    """
    if high_col not in df.columns or low_col not in df.columns:
        logger.error("Columnas %s o %s no encontradas.", high_col, low_col)
        return pd.DataFrame(index=df.index)

    resultado = pd.DataFrame(index=df.index)

    # Evaluar resistencia (Up Fractal)
    # El centro está en t - period (shift(period))
    cond_resistencia = True
    for i in range(1, period * 2 + 1):
        if i == period:
            continue
        # El máximo del centro debe ser estrictamente mayor que el máximo de sus vecinos
        cond_resistencia &= (df[high_col].shift(period) > df[high_col].shift(i))
    # La vela actual (shift(0)) debe ser también menor al centro
    cond_resistencia &= (df[high_col].shift(period) > df[high_col])

    # Evaluar soporte (Down Fractal)
    cond_soporte = True
    for i in range(1, period * 2 + 1):
        if i == period:
            continue
        # El mínimo del centro debe ser estrictamente menor que el mínimo de sus vecinos
        cond_soporte &= (df[low_col].shift(period) < df[low_col].shift(i))
    cond_soporte &= (df[low_col].shift(period) < df[low_col])

    # Criba Macro: solo guarda el fractal si es el max/min absoluto de las últimas 30 velas
    macro_period = 30  # ~1 semana en 4H
    rolling_max = df[high_col].shift(period).rolling(window=macro_period, min_periods=1).max()
    rolling_min = df[low_col].shift(period).rolling(window=macro_period, min_periods=1).min()
    cond_resistencia &= (df[high_col].shift(period) >= rolling_max)
    cond_soporte     &= (df[low_col].shift(period)  <= rolling_min)

    # Rellenar los valores raw (1 o 0) usando np.where (vectorizado)

    resultado["is_resistance_fractal"] = np.where(cond_resistencia, 1, 0)
    resultado["is_support_fractal"]    = np.where(cond_soporte, 1, 0)

    # -----------------------------------------------------------------
    # AGRUPACIÓN POR ZONA DE PRECIO (Zone Clustering)
    # -----------------------------------------------------------------
    # Tras la criba temporal, agrupamos fractales que estén muy cerca en
    # precio (dentro del ±1% del primer fractal "ancla" de cada zona).
    # Solo el primer fractal de cada zona sobrevive; los demás se eliminan.
    # Esto evita que el sistema compre la misma resistencia 3 veces seguidas.
    # -----------------------------------------------------------------
    zone_tolerance = 0.01  # 1% — configurable

    def _cluster_fractals(series: pd.Series, prices: pd.Series) -> pd.Series:
        """
        Elimina fractales que caen dentro del ±zone_tolerance% del primer
        fractal "ancla" de su zona. Los recorre cronológicamente.
        """
        clustered = series.copy()
        anchor_price = None

        for idx in series.index:
            if series[idx] == 1:
                price = prices[idx]
                if anchor_price is None:
                    # Primer fractal: es el ancla de la nueva zona
                    anchor_price = price
                else:
                    dist = abs(price - anchor_price) / anchor_price
                    if dist <= zone_tolerance:
                        # Dentro de la zona del ancla → eliminar este fractal
                        clustered[idx] = 0
                    else:
                        # Nuevo nivel de precio → es el ancla de una nueva zona
                        anchor_price = price
        return clustered

    # Extraer precios en los índices de las velas centrales de los fractales
    # (el centro del fractal está en t - period, pero lo mapeamos al índice actual)
    highs = df[high_col].shift(period)
    lows  = df[low_col].shift(period)

    resultado["is_resistance_fractal"] = _cluster_fractals(
        resultado["is_resistance_fractal"], highs
    )
    resultado["is_support_fractal"] = _cluster_fractals(
        resultado["is_support_fractal"], lows
    )

    logger.debug(
        "Fractales estructurales calculados. Soportes: %d, Resistencias: %d",
        resultado["is_support_fractal"].sum(),
        resultado["is_resistance_fractal"].sum()
    )
    return resultado



# ===========================================================================
# 2. PATRONES DE VELAS JAPONESAS
# ===========================================================================

def calculate_candlestick_patterns(
    df: pd.DataFrame,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
) -> pd.DataFrame:
    """
    Identifica patrones clásicos de velas utilizando operaciones vectorizadas.

    Patrones implementados:
    - Hammer (Martillo): Vela alcista o bajista con cuerpo pequeño en la
      parte superior, y una sombra/mecha inferior al menos 2x el tamaño del cuerpo.
      Indica fuerte rechazo de precios bajos.
    - Inverted Hammer (Martillo invertido / Shooting Star): Cuerpo pequeño en la
      parte inferior, con sombra superior al menos 2x el cuerpo.

    Args:
        df        (pd.DataFrame): DataFrame OHLC.
        open_col, high_col, low_col, close_col (str): Nombres de columnas.

    Returns:
        pd.DataFrame: Columnas 'is_hammer' y 'is_inverted_hammer' (1 o 0).
    """
    rango = df[high_col] - df[low_col]
    cuerpo = (df[close_col] - df[open_col]).abs()
    
    mecha_superior = df[high_col] - df[[open_col, close_col]].max(axis=1)
    mecha_inferior = df[[open_col, close_col]].min(axis=1) - df[low_col]

    # Prevenir divisiones por cero sumando un epsilon minúsculo
    epsilon = 1e-8
    
    # Lógica Martillo (Hammer)
    # 1. La mecha inferior debe ser >= 2 * cuerpo
    # 2. La mecha superior debe ser muy pequeña (<= 10% del rango total)
    # 3. El cuerpo debe ser mayor que 0 (no un doji perfecto, aunque a veces se acepta)
    cond_hammer = (mecha_inferior >= 2 * cuerpo) & \
                  (mecha_superior <= 0.1 * rango) & \
                  (cuerpo > epsilon)

    # Lógica Martillo Invertido (Inverted Hammer / Shooting Star)
    # 1. La mecha superior debe ser >= 2 * cuerpo
    # 2. La mecha inferior debe ser muy pequeña (<= 10% del rango total)
    cond_inverted = (mecha_superior >= 2 * cuerpo) & \
                    (mecha_inferior <= 0.1 * rango) & \
                    (cuerpo > epsilon)

    resultado = pd.DataFrame(index=df.index)
    resultado["is_hammer"]          = np.where(cond_hammer, 1, 0)
    resultado["is_inverted_hammer"] = np.where(cond_inverted, 1, 0)

    logger.debug(
        "Patrones calculados. Hammers: %d, Inverted: %d",
        resultado["is_hammer"].sum(),
        resultado["is_inverted_hammer"].sum()
    )
    return resultado


# ===========================================================================
# 3. WICK RECLAIMS (MECHAS DE RECHAZO)
# ===========================================================================

def calculate_wick_reclaims(
    df: pd.DataFrame,
    threshold: float = 0.60,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
) -> pd.DataFrame:
    """
    Detecta velas de Wick Reclaim (Rechazo agresivo).

    Un wick reclaim indica un barrido de liquidez. Ocurre cuando el mercado
    empuja el precio hacia una dirección pero es inmediatamente absorbido y
    revertido dentro de la misma vela, dejando una mecha enorme.

    Lógica:
    - Bullish Wick Reclaim: La mecha inferior representa más del `threshold`%
      del rango total de la vela.
    - Bearish Wick Reclaim: La mecha superior representa más del `threshold`%
      del rango total de la vela.

    Args:
        df        (pd.DataFrame): DataFrame OHLC.
        threshold (float): Porcentaje mínimo del rango que debe ser mecha (ej 0.60).
        open_col, high_col, low_col, close_col (str): Nombres de columnas.

    Returns:
        pd.DataFrame: Columnas 'is_bullish_wick_reclaim', 'is_bearish_wick_reclaim'.
    """
    rango = df[high_col] - df[low_col]
    
    # Reemplazar 0 con un epsilon minúsculo para evitar ZeroDivisionError en dojis
    rango_seguro = rango.replace(0, 1e-8)
    
    mecha_superior = df[high_col] - df[[open_col, close_col]].max(axis=1)
    mecha_inferior = df[[open_col, close_col]].min(axis=1) - df[low_col]

    ratio_inferior = mecha_inferior / rango_seguro
    ratio_superior = mecha_superior / rango_seguro

    cond_bullish = ratio_inferior >= threshold
    cond_bearish = ratio_superior >= threshold

    resultado = pd.DataFrame(index=df.index)
    resultado["is_bullish_wick_reclaim"] = np.where(cond_bullish, 1, 0)
    resultado["is_bearish_wick_reclaim"] = np.where(cond_bearish, 1, 0)

    logger.debug(
        "Wick Reclaims calculados con threshold %.2f", threshold
    )
    return resultado


# ===========================================================================
# 4. FIBONACCI RETRACEMENTS & EXTENSIONS (ROLUING WINDOW)
# ===========================================================================

def calculate_fibonacci_levels(
    df: pd.DataFrame,
    window: int = 50,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
) -> pd.DataFrame:
    """
    Calcula niveles dinámicos de retroceso y extensión de Fibonacci.

    Para evitar el lookahead bias y mantener la automatización, los niveles
    se calculan sobre una ventana temporal rodante (rolling window).
    Se obtiene el Máximo (H) y el Mínimo (L) de los últimos `window` períodos.

    Niveles de retroceso internos (entre 0 y 100%):
        - Fib 38.2% = Máximo - 0.382 * Rango (o Mínimo + 0.618 * Rango)
        - Fib 61.8% = Máximo - 0.618 * Rango

    Niveles de extensión (objetivos de salida, > 100% o < 0%):
        - Ext Up 38.2% = Máximo + 0.382 * Rango
        - Ext Up 61.8% = Máximo + 0.618 * Rango
        - Ext Down 38.2% = Mínimo - 0.382 * Rango
        - Ext Down 61.8% = Mínimo - 0.618 * Rango

    Nota ML: Al modelo no le sirven los precios absolutos de estos niveles.
    Se calculan las distancias relativas porcentuales del cierre actual a los niveles.

    Args:
        df        (pd.DataFrame): DataFrame OHLC.
        window    (int): Período de lookback para buscar High y Low (ej. 50).
        high_col, low_col, close_col (str): Columnas.

    Returns:
        pd.DataFrame: Columnas con las distancias porcentuales a cada nivel Fib.
                      Valores cercanos a 0 significan que el precio está en el nivel.
    """
    resultado = pd.DataFrame(index=df.index)

    # Identificar extremos en la ventana móvil (sin lookahead bias)
    rolling_max = df[high_col].rolling(window=window, min_periods=window).max()
    rolling_min = df[low_col].rolling(window=window, min_periods=window).min()
    rango = rolling_max - rolling_min
    
    # Precios absolutos de los niveles
    fib_retr_382 = rolling_max - 0.382 * rango
    fib_retr_618 = rolling_max - 0.618 * rango
    
    fib_ext_up_382 = rolling_max + 0.382 * rango
    fib_ext_up_618 = rolling_max + 0.618 * rango
    
    fib_ext_dn_382 = rolling_min - 0.382 * rango
    fib_ext_dn_618 = rolling_min - 0.618 * rango

    # Para el ML, la característica útil es la distancia % entre el precio y el nivel
    # Distancia = (Precio_Actual - Nivel_Fib) / Precio_Actual * 100
    cierre_seguro = df[close_col].replace(0, pd.NA)

    resultado["dist_fib_retr_382"] = ((df[close_col] - fib_retr_382) / cierre_seguro) * 100
    resultado["dist_fib_retr_618"] = ((df[close_col] - fib_retr_618) / cierre_seguro) * 100
    
    resultado["dist_fib_ext_up_382"] = ((fib_ext_up_382 - df[close_col]) / cierre_seguro) * 100
    resultado["dist_fib_ext_up_618"] = ((fib_ext_up_618 - df[close_col]) / cierre_seguro) * 100
    
    resultado["dist_fib_ext_dn_382"] = ((df[close_col] - fib_ext_dn_382) / cierre_seguro) * 100
    resultado["dist_fib_ext_dn_618"] = ((df[close_col] - fib_ext_dn_618) / cierre_seguro) * 100

    logger.debug("Niveles de Fibonacci calculados en ventana de %d periodos.", window)
    return resultado


# ===========================================================================
# PIPELINE COMPLETO: PRICE ACTION
# ===========================================================================

def add_price_action_features(
    df: pd.DataFrame,
    fractal_period: int = 2,
    wick_threshold: float = 0.60,
    fib_window: int = 50,
) -> pd.DataFrame:
    """
    Añade todas las características de Price Action al DataFrame.

    Ejecuta todas las funciones vectorizadas y une los resultados al
    DataFrame original, proporcionando las características listas para el ML.

    Args:
        df             (pd.DataFrame): DataFrame OHLC original.
        fractal_period (int): Distancia lateral para fractales (def 2).
        wick_threshold (float): Porcentaje para wick reclaims (def 0.60).
        fib_window     (int): Ventana lookback para Fibonacci (def 50).

    Returns:
        pd.DataFrame: DataFrame con las nuevas columnas categóricas/numéricas.
    """
    df_out = df.copy()

    logger.info("Calculando características de Price Action...")

    df_fractals = calculate_bill_williams_fractals(df_out, period=fractal_period)
    df_candles  = calculate_candlestick_patterns(df_out)
    df_wicks    = calculate_wick_reclaims(df_out, threshold=wick_threshold)
    df_fib      = calculate_fibonacci_levels(df_out, window=fib_window)

    df_out = pd.concat([df_out, df_fractals, df_candles, df_wicks, df_fib], axis=1)

    logger.info(
        "Price Action completado. Nuevas columnas: %d",
        len(df_fractals.columns) + len(df_candles.columns) + 
        len(df_wicks.columns) + len(df_fib.columns)
    )

    return df_out


# ---------------------------------------------------------------------------
# BLOQUE DE PRUEBA RÁPIDA (smoke-test)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from pathlib import Path

    print("\n=== SMOKE-TEST: Price Action sobre SPY_4Hour ===\n")

    ruta = Path("data/raw/SPY_4Hour.csv")
    if not ruta.exists():
        print("ERROR: No se encuentra SPY_4Hour.csv. Ejecuta tiingo_loader.py.")
        exit(1)

    df_raw = pd.read_csv(ruta, index_col="datetime", parse_dates=True)
    df_features = add_price_action_features(df_raw)

    columnas_bool = [
        "is_resistance_fractal", "is_support_fractal", 
        "is_hammer", "is_inverted_hammer",
        "is_bullish_wick_reclaim", "is_bearish_wick_reclaim"
    ]
    
    print("--- Resumen de Patrones Encontrados ---")
    for col in columnas_bool:
        total = df_features[col].sum()
        print(f"{col:<25} : {total} ocurrencias")

    print("\n--- Primeras 5 filas con Fibonaccis calculados ---")
    columnas_fib = [c for c in df_features.columns if "dist_fib" in c]
    # Mostrar a partir de la fila 50 (cuando la ventana rolling empieza a tener datos)
    print(df_features[["close"] + columnas_fib].iloc[50:55].round(2).to_string())
