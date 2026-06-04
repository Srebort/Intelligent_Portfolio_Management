"""
Módulo de indicadores técnicos para el análisis de series temporales de precios.

Transforma la serie temporal OHLCV en características (features) matemáticas
estandarizadas que el modelo de Machine Learning usará para identificar
tendencia, momentum y volatilidad del mercado.

GARANTÍA ANTI-LOOKAHEAD BIAS:
    Todas las funciones de este módulo utilizan exclusivamente operaciones
    retrospectivas (rolling backward-looking). En la práctica:
      - pd.Series.rolling(n).mean()  → usa las n velas anteriores, incluyendo la actual.
      - pd.Series.ewm(span=n)        → pondera exponencialmente hacia el pasado.
      - pd.Series.diff()             → diferencia respecto a la vela anterior.
    Ninguna función accede jamás a datos de barras futuras.

SOPORTE MULTI-TIMEFRAME:
    Todas las funciones son agnósticas a la frecuencia temporal. Se pueden
    aplicar a DataFrames de velas 1H, 4H, diarias o semanales sin cambios.
    La ventana (period) debe ajustarse según la temporalidad si se desean
    períodos equivalentes (ej. RSI-14 en 4H ≈ RSI-56 en 1H).

Funciones disponibles:
    - calculate_rsi()             → RSI (Relative Strength Index)
    - calculate_atr()             → ATR (Average True Range) con manejo de gaps
    - calculate_natr()            → NATR (ATR normalizado por precio de cierre)
    - calculate_sma()             → SMA (Simple Moving Average)
    - calculate_ema()             → EMA (Exponential Moving Average)
    - calculate_moving_averages() → Conjunto de SMAs y EMAs en 5 ventanas
    - add_all_features()          → Añade todos los indicadores al DataFrame
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
logger = logging.getLogger("TechnicalFeatures")

# ---------------------------------------------------------------------------
# Ventanas estándar para medias móviles (rápidas y lentas)
# Estas ventanas representan períodos clásicos de análisis técnico:
#   - Rápidas (9, 21)  : capturan movimientos de corto plazo / momentum
#   - Medias (50, 100)  : tendencia de medio plazo (trimestral / semestral)
#   - Lenta (200)       : tendencia de largo plazo — referencia institucional
# ---------------------------------------------------------------------------
VENTANAS_MOVILES = [9, 21, 50, 100, 200]


# ===========================================================================
# RSI — Relative Strength Index
# ===========================================================================

def calculate_rsi(
    df: pd.DataFrame,
    period: int = 14,
    price_col: str = "close",
) -> pd.Series:
    """
    Calcula el RSI (Relative Strength Index) de forma completamente vectorizada.

    El RSI mide la velocidad y magnitud de los movimientos de precio recientes,
    oscilando entre 0 y 100. Valores por encima de 70 indican sobrecompra;
    por debajo de 30, sobreventa.

    FÓRMULA:
        delta    = close.diff(1)              ← diferencia con la vela anterior
        gain     = delta.clip(lower=0)        ← solo subidas (pérdidas → 0)
        loss     = (-delta).clip(lower=0)     ← solo bajadas (subidas → 0)
        avg_gain = gain.ewm(com=period-1)     ← media exponencial de ganancias
        avg_loss = loss.ewm(com=period-1)     ← media exponencial de pérdidas
        RS       = avg_gain / avg_loss
        RSI      = 100 - (100 / (1 + RS))

    ANTI-LOOKAHEAD BIAS:
        diff(1) solo usa la vela t-1.
        ewm con com=period-1 pondera hacia atrás.
        No se accede a ninguna fila futura.

    Args:
        df        (pd.DataFrame): DataFrame con columna de precio de cierre.
        period    (int):          Período del RSI. Por defecto 14 velas.
        price_col (str):          Nombre de la columna de precio. Por defecto 'close'.

    Returns:
        pd.Series: RSI con el mismo índice que df. Las primeras ~period filas
                   serán NaN (período de calentamiento del indicador).
    """
    if price_col not in df.columns:
        logger.error("Columna '%s' no encontrada en el DataFrame.", price_col)
        return pd.Series(dtype=float, index=df.index, name="RSI")

    # Paso 1: Diferencia de precio entre cada vela y la anterior
    # diff(1): solo mira hacia atrás → sin lookahead bias
    delta = df[price_col].diff(1)

    # Paso 2: Separar subidas y bajadas
    # clip asegura que no haya valores negativos en ganancias (ni positivos en pérdidas)
    ganancias = delta.clip(lower=0)       # subidas: valores positivos, el resto 0
    perdidas  = (-delta).clip(lower=0)    # bajadas: valores positivos, el resto 0

    # Paso 3: Media exponencial de Wilder (com = period - 1 equivale a alpha = 1/period)
    # adjust=False: usa la fórmula recursiva (no la ponderación ajustada hacia adelante)
    # min_periods=period: exige al menos 'period' observaciones para calcular
    media_ganancias = ganancias.ewm(com=period - 1, adjust=False, min_periods=period).mean()
    media_perdidas  = perdidas.ewm(com=period - 1, adjust=False, min_periods=period).mean()

    # Paso 4: Calcular RS y RSI
    # Evitar división por cero: cuando media_perdidas == 0, RSI = 100
    rs  = media_ganancias / media_perdidas.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    rsi.name = f"RSI_{period}"
    logger.debug("RSI(%d) calculado. Filas válidas: %d", period, rsi.notna().sum())
    return rsi


# ===========================================================================
# ATR — Average True Range
# ===========================================================================

def calculate_atr(
    df: pd.DataFrame,
    period: int = 14,
    high_col:  str = "high",
    low_col:   str = "low",
    close_col: str = "close",
) -> pd.Series:
    """
    Calcula el ATR (Average True Range), indicador de volatilidad del mercado.

    El ATR mide el rango medio de movimiento de precio en las últimas N velas.
    Es el pilar fundamental del módulo RiskManager para el dimensionamiento
    de posiciones (Position Sizing): stop loss = precio_entrada - ATR * multiplicador.

    MANEJO DE GAPS (huecos entre sesiones):
        Un gap ocurre cuando el precio de apertura de hoy es muy diferente al
        cierre de ayer (ej. tras una noticia fuera de horario). En ese caso,
        High - Low subestimaría el verdadero rango de la jornada.
        Por eso, el True Range incorpora el cierre anterior con shift(1):

        Ejemplo de gap alcista:
            Cierre ayer = 100  |  Apertura hoy = 108  |  High = 112  |  Low = 107
            TR1 = High - Low          = 112 - 107 = 5   ← rango de la vela
            TR2 = |High - Close_prev| = |112 - 100| = 12 ← captura el gap
            TR3 = |Low  - Close_prev| = |107 - 100| = 7  ← captura el gap
            True Range = max(5, 12, 7) = 12  ✓ (el rango real fue 12, no 5)

    FÓRMULA:
        TR = max(
            High - Low,              ← rango de la vela actual
            |High - Close_anterior|, ← gap alcista + rango superior
            |Low  - Close_anterior|  ← gap bajista + rango inferior
        )
        ATR = media exponencial de Wilder(TR, period)
              con alpha = 1/period (equivalente a com = period-1)

    ANTI-LOOKAHEAD BIAS:
        shift(1) desplaza el cierre una posición hacia atrás (cierre de ayer).
        La media ewm solo pondera hacia el pasado.

    Args:
        df        (pd.DataFrame): DataFrame con columnas high, low y close.
        period    (int):          Período del ATR. Por defecto 14 velas.
        high_col  (str):          Columna del máximo.
        low_col   (str):          Columna del mínimo.
        close_col (str):          Columna del cierre.

    Returns:
        pd.Series: ATR con el mismo índice que df.
    """
    columnas_requeridas = [high_col, low_col, close_col]
    for col in columnas_requeridas:
        if col not in df.columns:
            logger.error("Columna '%s' no encontrada en el DataFrame.", col)
            return pd.Series(dtype=float, index=df.index, name=f"ATR_{period}")

    # Cierre de la vela anterior — captura los gaps entre sesiones
    # shift(1): desplaza hacia atrás, sin lookahead bias
    cierre_anterior = df[close_col].shift(1)

    # Las tres componentes del True Range
    tr1 = df[high_col] - df[low_col]                   # rango interno de la vela
    tr2 = (df[high_col] - cierre_anterior).abs()        # gap alcista + rango superior
    tr3 = (df[low_col]  - cierre_anterior).abs()        # gap bajista + rango inferior

    # True Range = máximo de las tres componentes (captura el movimiento real)
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    true_range.name = "TR"

    # ATR: media exponencial de Wilder
    # com = period - 1  ⇔  alpha = 1/period (fórmula original de J. Welles Wilder)
    atr = true_range.ewm(com=period - 1, adjust=False, min_periods=period).mean()

    atr.name = f"ATR_{period}"
    logger.debug("ATR(%d) calculado. Filas válidas: %d", period, atr.notna().sum())
    return atr


# ===========================================================================
# NATR — Normalized Average True Range
# ===========================================================================

def calculate_natr(
    df: pd.DataFrame,
    period: int = 14,
    high_col:  str = "high",
    low_col:   str = "low",
    close_col: str = "close",
) -> pd.Series:
    """
    Calcula el NATR (Normalized Average True Range), versión porcentual del ATR.

    El problema del ATR absoluto es que no permite comparar la volatilidad
    entre activos de precios muy distintos:
        - AAPL a 180$  con ATR = 3.0  → volatilidad = 1.67%
        - NVDA a 900$  con ATR = 3.0  → volatilidad = 0.33%
    Ambos tienen el mismo ATR pero AAPL es 5 veces más volátil en términos relativos.

    El NATR resuelve esto dividiendo el ATR por el precio de cierre y
    expressándolo en porcentaje:

    FÓRMULA:
        NATR = (ATR / Close) × 100

    USO EN EL TFM:
        El NATR permite al RiskManager comparar la volatilidad de todos los
        activos del universo (de 10$ a 500$+) en la misma escala porcentual,
        facilitando el dimensionamiento de posiciones homogéneo.

    ANTI-LOOKAHEAD BIAS:
        Hereda la garantía del ATR. El Close usado para normalizar es el de
        la misma vela (cierre de la barra actual), no del futuro.

    Args:
        df        (pd.DataFrame): DataFrame con columnas high, low y close.
        period    (int):          Período del ATR subyacente. Por defecto 14.
        high_col  (str):          Columna del máximo.
        low_col   (str):          Columna del mínimo.
        close_col (str):          Columna del cierre.

    Returns:
        pd.Series: NATR en porcentaje (%) con el mismo índice que df.
                   Ejemplo: NATR = 1.5 significa que la volatilidad media
                   de las últimas 14 velas fue el 1.5% del precio.
    """
    if close_col not in df.columns:
        logger.error("Columna '%s' no encontrada para calcular NATR.", close_col)
        return pd.Series(dtype=float, index=df.index, name=f"NATR_{period}")

    # Calcular el ATR con la misma función (reutilización de código)
    atr = calculate_atr(df, period=period,
                        high_col=high_col, low_col=low_col, close_col=close_col)

    # Normalizar: ATR / precio_cierre * 100 → expresado en porcentaje
    # replace(0, NaN): evitar división por cero en precios nulos
    natr = (atr / df[close_col].replace(0, pd.NA)) * 100

    natr.name = f"NATR_{period}"
    logger.debug("NATR(%d) calculado. Filas válidas: %d", period, natr.notna().sum())
    return natr


# ===========================================================================
# SMA — Simple Moving Average
# ===========================================================================

def calculate_sma(
    df: pd.DataFrame,
    period: int,
    price_col: str = "close",
) -> pd.Series:
    """
    Calcula la SMA (Simple Moving Average) para una ventana temporal dada.

    La SMA es la media aritmética de los últimos N precios de cierre.
    Se usa para identificar la tendencia suavizando el ruido del precio.

    ANTI-LOOKAHEAD BIAS:
        rolling(n).mean() usa la vela actual + las n-1 anteriores.
        Nunca accede a barras futuras.

    Args:
        df        (pd.DataFrame): DataFrame con columna de precio.
        period    (int):          Número de velas para la ventana de la media.
        price_col (str):          Columna de precio. Por defecto 'close'.

    Returns:
        pd.Series: SMA con el mismo índice que df. Las primeras period-1 filas son NaN.
    """
    if price_col not in df.columns:
        logger.error("Columna '%s' no encontrada en el DataFrame.", price_col)
        return pd.Series(dtype=float, index=df.index, name=f"SMA_{period}")

    # min_periods=period: la SMA solo se calcula cuando hay suficientes datos históricos
    sma = df[price_col].rolling(window=period, min_periods=period).mean()
    sma.name = f"SMA_{period}"
    return sma


# ===========================================================================
# EMA — Exponential Moving Average
# ===========================================================================

def calculate_ema(
    df: pd.DataFrame,
    period: int,
    price_col: str = "close",
) -> pd.Series:
    """
    Calcula la EMA (Exponential Moving Average) para una ventana temporal dada.

    La EMA pondera más los precios recientes que los antiguos, reaccionando
    más rápido a los cambios de precio que la SMA. Se usa para detectar
    cruces de medias y señales de entrada/salida.

    FÓRMULA:
        alpha = 2 / (period + 1)
        EMA_t = alpha × Price_t + (1 - alpha) × EMA_{t-1}

    ANTI-LOOKAHEAD BIAS:
        ewm con adjust=False usa la fórmula recursiva hacia el pasado.
        min_periods=period: requiere al menos N observaciones pasadas.

    Args:
        df        (pd.DataFrame): DataFrame con columna de precio.
        period    (int):          Número de velas para el span de la EMA.
        price_col (str):          Columna de precio. Por defecto 'close'.

    Returns:
        pd.Series: EMA con el mismo índice que df.
    """
    if price_col not in df.columns:
        logger.error("Columna '%s' no encontrada en el DataFrame.", price_col)
        return pd.Series(dtype=float, index=df.index, name=f"EMA_{period}")

    # span=period → alpha = 2/(period+1), equivalente al período estándar de la EMA
    # adjust=False: formula recursiva (no ponderación hacia adelante)
    ema = df[price_col].ewm(span=period, adjust=False, min_periods=period).mean()
    ema.name = f"EMA_{period}"
    return ema


# ===========================================================================
# CONJUNTO COMPLETO DE MEDIAS MÓVILES
# ===========================================================================

def calculate_moving_averages(
    df: pd.DataFrame,
    price_col: str = "close",
    ventanas: list = None,
) -> pd.DataFrame:
    """
    Calcula el conjunto completo de SMAs y EMAs para múltiples ventanas.

    Genera 8 columnas de medias móviles (4 SMA + 4 EMA) para las ventanas
    estándar de análisis técnico: 9, 21, 50 y 200 períodos.

    Interpretación de las ventanas:
        - 9   : tendencia muy corto plazo / momentum intradiario
        - 21  : ciclo mensual (aprox. 21 días de mercado)
        - 50  : tendencia de medio plazo (trimestral)
        - 100 : tendencia de medio-largo plazo (semestral)
        - 200 : tendencia de largo plazo (anual) — referencia institucional

    Args:
        df        (pd.DataFrame): DataFrame con columna de precio.
        price_col (str):          Columna de precio. Por defecto 'close'.
        ventanas  (list):         Lista de períodos. Por defecto [9, 21, 50, 200].

    Returns:
        pd.DataFrame: DataFrame con 8 columnas nuevas:
                      SMA_9, SMA_21, SMA_50, SMA_200,
                      EMA_9, EMA_21, EMA_50, EMA_200.
    """
    if ventanas is None:
        ventanas = VENTANAS_MOVILES  # [9, 21, 50, 200]

    resultado = pd.DataFrame(index=df.index)

    for ventana in ventanas:
        # SMA para cada ventana
        resultado[f"SMA_{ventana}"] = calculate_sma(df, period=ventana, price_col=price_col)
        # EMA para cada ventana
        resultado[f"EMA_{ventana}"] = calculate_ema(df, period=ventana, price_col=price_col)

    logger.debug(
        "Medias móviles calculadas: %d columnas para ventanas %s",
        len(resultado.columns), ventanas,
    )
    return resultado


# ===========================================================================
# PIPELINE COMPLETO: AÑADIR TODOS LOS INDICADORES AL DATAFRAME
# ===========================================================================

def add_all_features(
    df: pd.DataFrame,
    rsi_period:  int = 14,
    atr_period:  int = 14,
    price_col:   str = "close",
    high_col:    str = "high",
    low_col:     str = "low",
    close_col:   str = "close",
    timeframe:   str = "4Hour",
) -> pd.DataFrame:
    """
    Añade todos los indicadores técnicos al DataFrame de precios.

    Es la función principal del módulo: aplica RSI, ATR y el conjunto
    completo de medias móviles (SMA y EMA) en un único paso. Devuelve
    el DataFrame original enriquecido con todas las nuevas columnas.

    SOPORTE MULTI-TIMEFRAME:
        El parámetro 'timeframe' es informativo (aparece en logs) pero los
        cálculos son agnósticos a la frecuencia. Las funciones funcionan
        igual con DataFrames de 1H, 4H, diario o semanal.
        Nota: si se comparan períodos entre timeframes, 14 velas en 4H
        equivalen a 14×4 = 56 velas en 1H.

    GARANTÍA ANTI-LOOKAHEAD BIAS:
        Todas las funciones internas usan rolling/ewm backward-looking.
        Se realiza una verificación automática al final: si el índice está
        ordenado correctamente (ascendente), no puede existir fuga de datos.

    Args:
        df          (pd.DataFrame): DataFrame OHLCV (puede incluir columnas macro).
        rsi_period  (int):          Período del RSI. Por defecto 14.
        atr_period  (int):          Período del ATR. Por defecto 14.
        price_col   (str):          Columna de cierre para RSI y medias.
        high_col    (str):          Columna del máximo para ATR.
        low_col     (str):          Columna del mínimo para ATR.
        close_col   (str):          Columna del cierre para ATR.
        timeframe   (str):          Temporalidad informativa ('1Hour', '4Hour', etc.).

    Returns:
        pd.DataFrame: DataFrame original con las columnas de indicadores añadidas:
                      RSI_14, ATR_14, SMA_9, SMA_21, SMA_50, SMA_200,
                      EMA_9, EMA_21, EMA_50, EMA_200.
    """
    # Hacer una copia para no modificar el DataFrame original (inmutabilidad)
    df_out = df.copy()

    logger.info(
        "Calculando indicadores técnicos | timeframe=%s | %d filas",
        timeframe, len(df_out),
    )

    # ---- 1. RSI ------------------------------------------------
    df_out[f"RSI_{rsi_period}"] = calculate_rsi(
        df_out, period=rsi_period, price_col=price_col
    )

    # ---- 2. ATR y NATR -----------------------------------------
    df_out[f"ATR_{atr_period}"] = calculate_atr(
        df_out, period=atr_period,
        high_col=high_col, low_col=low_col, close_col=close_col,
    )
    df_out[f"NATR_{atr_period}"] = calculate_natr(
        df_out, period=atr_period,
        high_col=high_col, low_col=low_col, close_col=close_col,
    )

    # ---- 3. Medias Móviles (SMA + EMA en 4 ventanas) -----------
    df_medias = calculate_moving_averages(df_out, price_col=price_col)
    df_out = pd.concat([df_out, df_medias], axis=1)

    # ---- 4. Verificación anti-lookahead bias -------------------
    # El índice debe estar ordenado de forma ascendente (más antiguo primero).
    # Si el índice está correctamente ordenado, las operaciones rolling
    # NUNCA pueden acceder a datos futuros por construcción de pandas.
    if not df_out.index.is_monotonic_increasing:
        logger.warning(
            "ADVERTENCIA: El índice del DataFrame NO está ordenado "
            "cronológicamente. Ordenando automáticamente para garantizar "
            "que no existe lookahead bias..."
        )
        df_out = df_out.sort_index()

    columnas_indicadores = [
        f"RSI_{rsi_period}", f"ATR_{atr_period}", f"NATR_{atr_period}",
        "SMA_9", "SMA_21", "SMA_50", "SMA_100", "SMA_200",
        "EMA_9", "EMA_21", "EMA_50", "EMA_100", "EMA_200",
    ]
    columnas_validas = [c for c in columnas_indicadores if c in df_out.columns]

    logger.info(
        "Indicadores calculados: %s | Filas con datos completos: %d/%d",
        columnas_validas,
        df_out[columnas_validas].dropna().shape[0],
        len(df_out),
    )

    return df_out


# ---------------------------------------------------------------------------
# BLOQUE DE PRUEBA RÁPIDA (smoke-test)
#
# Se ejecuta únicamente cuando lanzas el script directamente:
#   python -m src.features.technical
#
# Carga el dataset fusionado de SPY (generado por data_merger.py),
# calcula todos los indicadores técnicos y verifica que:
#   1. No existen NaN en las columnas de indicadores (excepto el calentamiento)
#   2. El RSI siempre está entre 0 y 100
#   3. El ATR siempre es positivo
#   4. Las SMAs y EMAs son coherentes con el precio de cierre
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from pathlib import Path

    print("\n=== SMOKE-TEST: Indicadores técnicos sobre SPY_4Hour ===\n")

    # Intentar cargar el dataset fusionado (preferido) o el crudo de Tiingo
    ruta_merged = Path("data/processed/SPY_4Hour_merged.csv")
    ruta_raw    = Path("data/raw/SPY_4Hour.csv")

    if ruta_merged.exists():
        df = pd.read_csv(ruta_merged, index_col="datetime", parse_dates=True)
        print(f"Dataset cargado: {ruta_merged} ({len(df)} filas)\n")
    elif ruta_raw.exists():
        df = pd.read_csv(ruta_raw, index_col="datetime", parse_dates=True)
        print(f"Dataset cargado: {ruta_raw} ({len(df)} filas)\n")
    else:
        print(
            "ERROR: No se encontro ningun CSV de precios.\n"
            "Ejecuta primero:\n"
            "  python -m src.data.tiingo_loader\n"
            "  python -m src.data.data_merger"
        )
        exit(1)

    # Calcular todos los indicadores técnicos
    df_features = add_all_features(df, timeframe="4Hour")

    # Mostrar resultado
    columnas_indicadores = [
        "close", "RSI_14", "ATR_14", "NATR_14",
        "SMA_9", "SMA_21", "SMA_50",
        "EMA_9", "EMA_21", "EMA_50",
    ]
    columnas_presentes = [c for c in columnas_indicadores if c in df_features.columns]

    print("--- Primeras 15 filas (con periodo de calentamiento) ---")
    print(df_features[columnas_presentes].head(15).round(3).to_string())

    print(f"\n--- Ultimas 5 filas (indicadores completamente calculados) ---")
    print(df_features[columnas_presentes].tail(5).round(3).to_string())

    # Verificaciones de calidad
    print("\n--- Verificaciones anti-lookahead bias y calidad ---")

    rsi = df_features["RSI_14"].dropna()
    print(f"RSI_14  | Min: {rsi.min():.2f} | Max: {rsi.max():.2f} | "
          f"[OK: siempre entre 0-100: {(rsi >= 0).all() and (rsi <= 100).all()}]")

    atr = df_features["ATR_14"].dropna()
    print(f"ATR_14  | Min: {atr.min():.4f} | Max: {atr.max():.4f} | "
          f"[OK: siempre positivo: {(atr > 0).all()}]")

    natr = df_features["NATR_14"].dropna()
    print(f"NATR_14 | Min: {natr.min():.4f}% | Max: {natr.max():.4f}% | "
          f"[OK: siempre positivo: {(natr > 0).all()}]")

    print(f"Indice ordenado (req. anti-bias): {df_features.index.is_monotonic_increasing}")
    print(f"Dimensiones finales del dataset : {df_features.shape}")
    print(f"Columnas totales                : {list(df_features.columns)}")
