"""
Módulo para la fusión de datos de mercado con datos macroeconómicos.

Este módulo genera el dataset definitivo que consumirá el modelo de Machine
Learning del TFM. Combina las velas intradiarias de Tiingo (OHLCV) con los
indicadores macroeconómicos de FRED (inflación, tipos de interés, VIX…)
en un único DataFrame alineado temporalmente.

El principal reto es que ambas fuentes tienen frecuencias distintas:
    - Datos de mercado (Tiingo) : velas de 1H, 4H... (intradiario)
    - Datos macro (FRED)        : diarios o mensuales

La solución es un LEFT JOIN por fecha + forward-fill, que propaga el
último valor conocido de cada indicador macro a todas las velas de mercado
del mismo período (ej. el CPI de enero se propaga a todas las velas de enero).

Flujo del pipeline:
    data/raw/{ticker}_{freq}.csv   ──┐
                                     ├──► DataMerger ──► data/processed/{ticker}_{freq}_merged.csv
    data/raw/macro_data.csv        ──┘
"""

import logging
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuración del sistema de logs
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("DataMerger")

# ---------------------------------------------------------------------------
# Rutas de los ficheros de entrada y salida (relativas a la raíz del proyecto)
# ---------------------------------------------------------------------------
DIR_RAW       = Path("data/raw")        # Directorio con los CSVs originales
DIR_PROCESSED = Path("data/processed")  # Directorio con el dataset final


class DataMerger:
    """
    Clase encargada de fusionar las velas de mercado con los datos macro.

    Flujo de uso típico:
        1. Instanciar la clase (sin parámetros obligatorios).
        2. Llamar a merge_ticker() para fusionar un único activo con el macro.
        3. Llamar a merge_all_tickers() para procesar todos los CSVs de mercado
           disponibles en data/raw/ de una vez.

    Ejemplo rápido:
        merger = DataMerger()
        df = merger.merge_ticker("SPY", "4Hour")

    Resultado: data/processed/SPY_4Hour_merged.csv
    """

    def __init__(self):
        """
        Inicializa el DataMerger.

        Crea el directorio data/processed/ si no existe y carga el fichero
        de datos macroeconómicos (macro_data.csv) en memoria, ya que es
        compartido por todos los activos que se vayan a procesar.

        Raises:
            FileNotFoundError: Si macro_data.csv no existe en data/raw/.
                               Ejecuta primero src/data/fred_loader.py.
        """
        # Crear el directorio de salida si aún no existe
        DIR_PROCESSED.mkdir(parents=True, exist_ok=True)

        # Cargar el DataFrame macroeconómico compartido
        self.df_macro = self._load_macro()
        logger.info("DataMerger inicializado correctamente.")

    # ------------------------------------------------------------------
    # MÉTODOS PÚBLICOS
    # ------------------------------------------------------------------

    def merge_ticker(self, ticker: str, resample_freq: str = "4Hour") -> pd.DataFrame:
        """
        Fusiona las velas de UN único activo con los indicadores macroeconómicos.

        Pasos que realiza:
            1. Carga el CSV de precios del activo desde data/raw/.
            2. Alinea ambos DataFrames a zona horaria UTC.
            3. Normaliza el índice del macro a UTC para el join.
            4. Ejecuta un LEFT JOIN por índice de fecha.
            5. Aplica forward-fill para propagar los valores macro.
            6. Elimina filas residuales con NaN en columnas OHLCV.
            7. Guarda el resultado en data/processed/.

        Args:
            ticker        (str): Símbolo del activo, ej. 'AAPL' o 'SPY'.
            resample_freq (str): Frecuencia de las velas, ej. '4Hour' o '1Hour'.

        Returns:
            pd.DataFrame: Dataset fusionado con columnas OHLCV + indicadores macro.
                          Devuelve DataFrame vacío si el CSV del ticker no existe.
        """
        # Paso 1: Cargar el CSV de precios del activo
        df_precios = self._load_prices(ticker, resample_freq)
        if df_precios is None or df_precios.empty:
            return pd.DataFrame()

        logger.info(
            "Fusionando '%s' (%s) | %d velas + %d filas macro",
            ticker, resample_freq, len(df_precios), len(self.df_macro),
        )

        # Paso 2 y 3: Alinear zonas horarias de ambos DataFrames a UTC
        df_precios = self._normalizar_timezone(df_precios, nombre="precios")
        df_macro   = self._normalizar_timezone(self.df_macro.copy(), nombre="macro")

        # Paso 4: LEFT JOIN por índice de fecha
        # Usamos merge_asof para emparejar cada vela con el dato macro anterior
        # más cercano (asume que df_precios y df_macro están ordenados)
        df_precios = df_precios.sort_index()
        df_macro   = df_macro.sort_index()

        # pd.merge_asof: para cada fila del DataFrame izquierdo (precios),
        # busca la fila del DataFrame derecho (macro) con el índice más cercano
        # anterior o igual. Equivale a un LEFT JOIN + forward-fill automático.
        df_fusionado = pd.merge_asof(
            left=df_precios.reset_index(),
            right=df_macro.reset_index(),
            left_on="datetime",   # columna de fecha en el DataFrame de precios
            right_on="fecha",     # columna de fecha en el DataFrame macro
            direction="backward", # propaga el dato macro más reciente hacia adelante
        )

        # Restaurar el índice temporal y limpiar columna auxiliar de fecha macro
        df_fusionado.set_index("datetime", inplace=True)
        if "fecha" in df_fusionado.columns:
            df_fusionado.drop(columns=["fecha"], inplace=True)

        # Paso 5: Aplicar forward-fill adicional para cubrir posibles huecos
        # residuales (ej. festivos donde no hay dato macro posterior)
        df_fusionado = df_fusionado.ffill()

        # Paso 6: Eliminar filas con NaN en columnas OHLCV críticas
        # (no eliminamos filas donde solo falten datos macro, ya que el ffill
        # debería haberlos cubierto; solo eliminamos si faltan precios reales)
        columnas_ohlcv = [c for c in ["open", "high", "low", "close", "volume"]
                          if c in df_fusionado.columns]
        df_fusionado.dropna(subset=columnas_ohlcv, inplace=True)

        logger.info(
            "Fusión completada: %d filas × %d columnas | %s -> %s",
            df_fusionado.shape[0], df_fusionado.shape[1],
            df_fusionado.index.min().date(), df_fusionado.index.max().date(),
        )

        # Paso 7: Guardar el dataset fusionado en data/processed/
        self._save_csv(df_fusionado, ticker, resample_freq)

        return df_fusionado

    def merge_all_tickers(self) -> dict:
        """
        Fusiona TODOS los CSVs de velas disponibles en data/raw/ con el macro.

        Detecta automáticamente todos los ficheros con el patrón
        {ticker}_{freq}.csv en data/raw/ (excluyendo macro_data.csv) y
        llama a merge_ticker() para cada uno.

        Returns:
            dict[str, pd.DataFrame]: Diccionario {ticker_freq: DataFrame}
                                     con todos los datasets fusionados.
        """
        # Buscar todos los CSVs de precios (excluir macro_data.csv)
        archivos_precios = [
            f for f in DIR_RAW.glob("*.csv")
            if f.name != "macro_data.csv"
        ]

        if not archivos_precios:
            logger.warning(
                "No se encontraron ficheros de precios en '%s'. "
                "Ejecuta primero src/data/tiingo_loader.py.", DIR_RAW,
            )
            return {}

        logger.info(
            "Procesando %d ficheros de precios encontrados en '%s'.",
            len(archivos_precios), DIR_RAW,
        )

        resultados = {}

        for archivo in archivos_precios:
            # Extraer ticker y frecuencia del nombre del fichero
            # Formato esperado: AAPL_4Hour.csv → ticker='AAPL', freq='4Hour'
            partes = archivo.stem.split("_")  # archivo.stem = 'AAPL_4Hour'
            if len(partes) < 2:
                logger.warning("Nombre de fichero inesperado: '%s'. Se omite.", archivo.name)
                continue

            ticker = partes[0]
            freq   = "_".join(partes[1:])  # por si el freq tiene guiones bajos
            clave  = f"{ticker}_{freq}"

            df = self.merge_ticker(ticker, freq)
            if not df.empty:
                resultados[clave] = df

        logger.info(
            "Fusión masiva completada: %d/%d activos procesados.",
            len(resultados), len(archivos_precios),
        )
        return resultados

    # ------------------------------------------------------------------
    # MÉTODOS PRIVADOS (helpers internos)
    # ------------------------------------------------------------------

    def _load_macro(self) -> pd.DataFrame:
        """
        Carga el fichero data/raw/macro_data.csv en un DataFrame.

        El fichero lo genera FredLoader (src/data/fred_loader.py).
        El índice es la columna 'fecha', parseada como DatetimeIndex.

        Returns:
            pd.DataFrame: DataFrame con los indicadores macroeconómicos.

        Raises:
            FileNotFoundError: Si macro_data.csv no existe.
        """
        ruta_macro = DIR_RAW / "macro_data.csv"
        if not ruta_macro.exists():
            raise FileNotFoundError(
                f"No se encontró '{ruta_macro}'. "
                "Ejecuta primero: python -m src.data.fred_loader"
            )

        # Leer CSV con la columna 'fecha' como índice de tipo datetime
        df = pd.read_csv(ruta_macro, index_col="fecha", parse_dates=True)

        logger.info(
            "Datos macro cargados: %d filas × %d columnas desde '%s'.",
            df.shape[0], df.shape[1], ruta_macro,
        )
        return df

    def _load_prices(self, ticker: str, resample_freq: str) -> pd.DataFrame | None:
        """
        Carga el CSV de velas de un activo desde data/raw/.

        El fichero lo genera TiingoLoader (src/data/tiingo_loader.py).
        El índice es la columna 'datetime', parseada como DatetimeIndex UTC.

        Args:
            ticker        (str): Símbolo del activo, ej. 'SPY'.
            resample_freq (str): Frecuencia de las velas, ej. '4Hour'.

        Returns:
            pd.DataFrame | None: DataFrame de precios, o None si no existe el CSV.
        """
        nombre_archivo = f"{ticker}_{resample_freq}.csv"
        ruta_precios   = DIR_RAW / nombre_archivo

        if not ruta_precios.exists():
            logger.error(
                "No se encontró '%s'. "
                "Ejecuta primero: python -m src.data.tiingo_loader",
                ruta_precios,
            )
            return None

        # Leer CSV con la columna 'datetime' como índice
        df = pd.read_csv(ruta_precios, index_col="datetime", parse_dates=True)

        logger.info(
            "Precios cargados: %d filas para '%s' (%s).",
            len(df), ticker, resample_freq,
        )
        return df

    def _normalizar_timezone(self, df: pd.DataFrame, nombre: str = "") -> pd.DataFrame:
        """
        Asegura que el índice del DataFrame tenga zona horaria UTC.

        Maneja tres casos posibles:
            1. Índice ya tiene timezone UTC       → no hace nada.
            2. Índice tiene otra timezone          → convierte a UTC.
            3. Índice es naive (sin timezone)      → asigna UTC directamente.

        Estandarizar la timezone a UTC es imprescindible antes del JOIN
        para evitar errores de comparación entre índices tz-aware y tz-naive.

        Args:
            df     (pd.DataFrame): DataFrame cuyo índice se va a normalizar.
            nombre (str):          Nombre descriptivo para los logs.

        Returns:
            pd.DataFrame: El mismo DataFrame con índice en UTC.
        """
        tz_actual = df.index.tz

        if tz_actual is None:
            # Caso 3: índice sin timezone → asignar UTC (localize)
            df.index = df.index.tz_localize("UTC")
            logger.debug("Índice '%s' localizado a UTC (era naive).", nombre)

        elif str(tz_actual) != "UTC":
            # Caso 2: otra timezone → convertir a UTC
            df.index = df.index.tz_convert("UTC")
            logger.debug("Índice '%s' convertido de %s a UTC.", nombre, tz_actual)

        # Caso 1: ya es UTC → no hay nada que hacer

        return df

    def _save_csv(
        self, df: pd.DataFrame, ticker: str, resample_freq: str
    ) -> Path:
        """
        Guarda el DataFrame fusionado en data/processed/{ticker}_{freq}_merged.csv.

        El sufijo '_merged' distingue los ficheros procesados de los crudos.

        Args:
            df            (pd.DataFrame): Dataset fusionado final.
            ticker        (str):          Símbolo del activo, ej. 'SPY'.
            resample_freq (str):          Frecuencia, ej. '4Hour'.

        Returns:
            Path: Ruta completa al fichero CSV guardado.
        """
        ruta_csv = DIR_PROCESSED / f"{ticker}_{resample_freq}_merged.csv"
        df.to_csv(ruta_csv)
        logger.info(
            "Dataset fusionado guardado -> %s  (%d filas x %d columnas)",
            ruta_csv, df.shape[0], df.shape[1],
        )
        return ruta_csv


# ---------------------------------------------------------------------------
# BLOQUE DE PRUEBA RÁPIDA (smoke-test)
#
# Se ejecuta únicamente cuando lanzas el script directamente:
#   python -m src.data.data_merger
#
# Requiere que existan previamente en data/raw/:
#   - SPY_4Hour.csv      (generado por tiingo_loader.py)
#   - macro_data.csv     (generado por fred_loader.py)
#
# Fusiona ambos datasets, muestra las primeras filas del resultado y
# verifica que el índice es UTC y que no hay NaN en columnas OHLCV.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n=== SMOKE-TEST: Fusión de datos de mercado y macroeconómicos ===\n")

    # Instanciar el merger (carga macro_data.csv automáticamente)
    merger = DataMerger()

    # Fusionar SPY con velas de 4 horas (fichero de prueba generado por tiingo_loader)
    print("--- Test: Fusionando SPY_4Hour.csv + macro_data.csv ---\n")
    df_resultado = merger.merge_ticker("SPY", "4Hour")

    if not df_resultado.empty:
        print("--- Primeras 10 filas del dataset fusionado ---")
        print(df_resultado.head(10).to_string())

        print(f"\nDimensiones del dataset   : {df_resultado.shape}")
        print(f"Columnas                  : {list(df_resultado.columns)}")
        print(f"Tipo del indice           : {df_resultado.index.dtype}")
        print(f"Zona horaria del indice   : {df_resultado.index.tz}")

        # Comprobar valores nulos en las columnas OHLCV
        nulos_ohlcv = df_resultado[["open", "high", "low", "close", "volume"]].isnull().sum()
        print(f"\nNulos en columnas OHLCV   :")
        print(nulos_ohlcv)

        # Comprobar valores nulos en las columnas macro
        columnas_macro = [c for c in df_resultado.columns
                          if c not in ["open", "high", "low", "close", "volume"]]
        if columnas_macro:
            nulos_macro = df_resultado[columnas_macro].isnull().sum()
            print(f"\nNulos en columnas macro   :")
            print(nulos_macro)

        print(f"\nFichero guardado en       : data/processed/SPY_4Hour_merged.csv")
    else:
        print(
            "ERROR: No se generó el dataset fusionado.\n"
            "Asegurate de ejecutar primero:\n"
            "  python -m src.data.tiingo_loader\n"
            "  python -m src.data.fred_loader"
        )
