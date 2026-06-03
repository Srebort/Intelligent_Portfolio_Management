"""
Módulo para la descarga de datos intradía desde la API de Tiingo.

Este módulo proporciona la clase TiingoLoader, que se conecta al endpoint
IEX de Tiingo para obtener el histórico de precios OHLCV (apertura, máximo,
mínimo, cierre y volumen) de los activos definidos en config/settings.yaml.

La clave API se carga de forma segura desde el fichero .env local mediante
python-dotenv, garantizando que nunca quede expuesta en el repositorio.

Endpoint utilizado:
    GET https://api.tiingo.com/iex/{ticker}/prices
    Documentación oficial: https://www.tiingo.com/documentation/iex
"""

import os
import time
import logging
import yaml
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuración del sistema de logs
# Muestra: fecha, nivel (INFO/WARNING/ERROR), nombre del logger y mensaje.
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("TiingoLoader")

# ---------------------------------------------------------------------------
# Tabla de conversión de frecuencias.
# En settings.yaml usamos '1Hour', '4Hour', '1Day', '1Week' para mayor
# legibilidad, pero la API de Tiingo IEX espera minúsculas: '1hour', etc.
# ---------------------------------------------------------------------------
FREQ_MAP = {
    "1Hour": "1hour",   # Velas de 1 hora
    "4Hour": "4hour",   # Velas de 4 horas (temporalidad principal del TFM)
    "1Day":  "1day",    # Velas diarias
    "1Week": "1week",   # Velas semanales
    "30Min": "30min",   # Velas de 30 minutos
    "15Min": "15min",   # Velas de 15 minutos
}


def load_config(config_path: str = "config/settings.yaml") -> dict:
    """
    Carga la configuración del proyecto desde el fichero YAML indicado.

    Lee el archivo settings.yaml que contiene los parámetros globales
    del proyecto: lista de tickers, fechas de inicio/fin, temporalidades,
    URLs de las APIs, parámetros del modelo, etc.

    Args:
        config_path (str): Ruta al fichero YAML. Por defecto 'config/settings.yaml'.

    Returns:
        dict: Diccionario con toda la configuración parseada.

    Raises:
        FileNotFoundError: Si el fichero de configuración no existe en la ruta indicada.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Fichero de configuración no encontrado: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class TiingoLoader:
    """
    Clase encargada de descargar el histórico de precios OHLCV intradía
    desde el endpoint IEX de la API de Tiingo.

    Flujo de uso típico:
        1. Instanciar la clase → carga .env y settings.yaml automáticamente.
        2. Llamar a download_all_tickers() para descargar todos los activos
           configurados de una vez, o a download_ticker() para uno concreto.
        3. Los datos se guardan en data/raw/{ticker}_{frecuencia}.csv.

    Ejemplo rápido:
        loader = TiingoLoader()
        df = loader.download_ticker("AAPL", "2018-01-01", "2026-05-01", "4Hour")

    La clave API se lee de la variable de entorno TIINGO_API_KEY,
    que debe estar definida en el fichero .env local (nunca en el repositorio).
    """

    # URL base del endpoint IEX de Tiingo
    BASE_URL = "https://api.tiingo.com/iex"

    # Columnas OHLCV que pedimos a la API (se pasan como query param)
    COLUMNS = "open,high,low,close,volume"

    def __init__(self, config_path: str = "config/settings.yaml"):
        """
        Inicializa el TiingoLoader.

        Pasos que realiza:
            1. Carga las variables del fichero .env en el entorno del sistema.
            2. Lee TIINGO_API_KEY y valida que esté definida.
            3. Carga la configuración del proyecto desde settings.yaml.
            4. Construye las cabeceras HTTP reutilizables para todas las llamadas.

        Args:
            config_path (str): Ruta al fichero de configuración YAML.

        Raises:
            ValueError: Si TIINGO_API_KEY no está definida en el entorno.
        """
        # Paso 1: Cargar variables de entorno desde el .env local
        load_dotenv()

        # Paso 2: Leer y validar la clave API de Tiingo
        self.api_key = os.getenv("TIINGO_API_KEY")
        if not self.api_key:
            raise ValueError(
                "TIINGO_API_KEY no está definida. Añádela a tu fichero .env."
            )

        # Paso 3: Leer la configuración del proyecto (tickers, fechas, timeframes…)
        self.config = load_config(config_path)

        # Paso 4: Construir las cabeceras HTTP con el token de autenticación
        # Estas cabeceras se reutilizan en cada petición a la API
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Token {self.api_key}",
        }

        logger.info(
            "TiingoLoader inicializado correctamente. Configuración cargada desde '%s'.",
            config_path,
        )

    # ------------------------------------------------------------------
    # MÉTODOS PÚBLICOS
    # ------------------------------------------------------------------

    def download_ticker(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        resample_freq: str = "4Hour",
    ) -> pd.DataFrame:
        """
        Descarga el histórico de precios OHLCV para UN único activo.

        Realiza una petición GET al endpoint de Tiingo IEX con los parámetros
        indicados, valida la respuesta y devuelve un DataFrame limpio y listo
        para usar en el pipeline de análisis.

        Args:
            ticker        (str): Símbolo del activo, por ejemplo 'AAPL' o 'SPY'.
            start_date    (str): Fecha de inicio en formato YYYY-MM-DD.
            end_date      (str): Fecha de fin   en formato YYYY-MM-DD.
            resample_freq (str): Clave de temporalidad definida en settings.yaml.
                                 Valores válidos: '1Hour', '4Hour', '1Day', '1Week'.

        Returns:
            pd.DataFrame: DataFrame con índice DatetimeIndex (UTC) y las columnas
                          [open, high, low, close, volume].
                          Devuelve un DataFrame vacío si ocurre algún error.
        """
        # Traducir la frecuencia del formato del settings al formato de Tiingo
        tiingo_freq = FREQ_MAP.get(resample_freq)
        if tiingo_freq is None:
            logger.error(
                "Frecuencia desconocida '%s'. Valores válidos: %s",
                resample_freq, list(FREQ_MAP.keys()),
            )
            return pd.DataFrame()

        # Construir la URL específica del ticker (Tiingo espera minúsculas)
        url = f"{self.BASE_URL}/{ticker.lower()}/prices"

        # Parámetros de la query: rango de fechas, frecuencia y columnas deseadas
        params = {
            "startDate":    start_date,
            "endDate":      end_date,
            "resampleFreq": tiingo_freq,
            "columns":      self.COLUMNS,
        }

        logger.info(
            "Petición a Tiingo IEX | ticker=%s | frecuencia=%s | %s → %s",
            ticker, tiingo_freq, start_date, end_date,
        )

        # Realizar la petición HTTP con manejo de errores
        try:
            response = requests.get(
                url, headers=self.headers, params=params, timeout=30
            )
            # Lanza excepción si el código de respuesta es 4xx o 5xx
            response.raise_for_status()

        except requests.exceptions.HTTPError as e:
            # Error de la API (clave inválida, ticker no encontrado, límite superado…)
            logger.error("Error HTTP para '%s': %s", ticker, e)
            return pd.DataFrame()

        except requests.exceptions.ConnectionError:
            # Sin conexión a internet o dominio inaccesible
            logger.error("Error de conexión — comprueba tu acceso a internet.")
            return pd.DataFrame()

        except requests.exceptions.Timeout:
            # La API no respondió en 30 segundos
            logger.error("Tiempo de espera agotado para el ticker '%s'.", ticker)
            return pd.DataFrame()

        # Decodificar el cuerpo de la respuesta como JSON
        raw_data = response.json()

        # Comprobar que la API devolvió datos (podría estar vacío en festivos, etc.)
        if not raw_data:
            logger.warning(
                "Tiingo devolvió datos vacíos para el ticker '%s'. "
                "Puede ser un festivo o un rango de fechas incorrecto.",
                ticker,
            )
            return pd.DataFrame()

        # Parsear el JSON y devolver el DataFrame limpio
        return self._parse_response(raw_data, ticker)

    def download_all_tickers(self, delay_seconds: float = 0.5) -> dict:
        """
        Descarga el histórico de TODOS los activos definidos en settings.yaml.

        Lee la lista completa de tickers, las fechas y la temporalidad principal
        del fichero de configuración y llama a download_ticker() para cada uno.
        Al finalizar cada descarga correcta, guarda automáticamente el CSV.

        Incluye una pausa entre peticiones (delay_seconds) para respetar el
        límite de frecuencia de la cuenta gratuita de Tiingo.

        Args:
            delay_seconds (float): Segundos de espera entre peticiones.
                                   Por defecto 0.5s (recomendado para cuenta free).

        Returns:
            dict[str, pd.DataFrame]: Diccionario {ticker: DataFrame} con los
                                     activos descargados correctamente.
        """
        # Leer parámetros globales de la configuración
        tickers      = self.config.get("tickers", [])
        start_date   = self.config.get("start_date", "2018-01-01")
        end_date     = self.config.get("end_date",   "2026-05-01")
        primary_freq = self.config.get("timeframe_primary", "4Hour")

        if not tickers:
            logger.warning(
                "No se encontraron tickers en settings.yaml. "
                "Revisa la clave 'tickers' del fichero de configuración."
            )
            return {}

        logger.info(
            "Iniciando descarga masiva: %d activos | frecuencia=%s | %s → %s",
            len(tickers), primary_freq, start_date, end_date,
        )

        resultados: dict = {}

        for ticker in tickers:
            # Descargar datos para el ticker actual
            df = self.download_ticker(ticker, start_date, end_date, primary_freq)

            if not df.empty:
                # Guardar el CSV y registrar el resultado
                self._save_csv(df, ticker, primary_freq)
                resultados[ticker] = df
            else:
                logger.warning("No se guardaron datos para '%s'.", ticker)

            # Pausa para no superar el límite de peticiones por segundo de Tiingo
            time.sleep(delay_seconds)

        logger.info(
            "Descarga masiva completada. %d/%d activos descargados correctamente.",
            len(resultados), len(tickers),
        )
        return resultados

    # ------------------------------------------------------------------
    # MÉTODOS PRIVADOS (helpers internos, no se llaman desde fuera)
    # ------------------------------------------------------------------

    def _parse_response(self, raw_data: list, ticker: str) -> pd.DataFrame:
        """
        Convierte la respuesta JSON de Tiingo en un DataFrame OHLCV limpio.

        Pasos que realiza:
            1. Convierte la lista de diccionarios JSON en un DataFrame de pandas.
            2. Parsea la columna 'date' como datetime con zona horaria UTC
               y la establece como índice del DataFrame.
            3. Conserva únicamente las columnas OHLCV requeridas.
            4. Elimina filas donde todos los valores son NaN.
            5. Ordena el DataFrame cronológicamente (de más antiguo a más reciente).

        Args:
            raw_data (list): Lista de diccionarios devuelta por la API de Tiingo.
            ticker   (str):  Símbolo del activo (solo se usa para los logs).

        Returns:
            pd.DataFrame: DataFrame limpio con DatetimeIndex UTC y columnas OHLCV.
                          Devuelve DataFrame vacío si falta la columna 'date'.
        """
        # Paso 1: Convertir la lista JSON en DataFrame
        df = pd.DataFrame(raw_data)

        # Paso 2: Parsear y configurar el índice temporal
        if "date" not in df.columns:
            logger.error(
                "La respuesta de Tiingo para '%s' no contiene la columna 'date'.", ticker
            )
            return pd.DataFrame()

        # Convertir a datetime con zona horaria UTC y establecer como índice
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df.set_index("date", inplace=True)
        df.index.name = "datetime"  # Nombre descriptivo para el índice

        # Paso 3: Filtrar solo las columnas OHLCV necesarias
        columnas_ohlcv = ["open", "high", "low", "close", "volume"]
        disponibles    = [c for c in columnas_ohlcv if c in df.columns]
        faltantes      = set(columnas_ohlcv) - set(disponibles)

        if faltantes:
            logger.warning(
                "Ticker '%s': columnas ausentes en la respuesta de Tiingo: %s",
                ticker, faltantes,
            )

        df = df[disponibles]

        # Paso 4: Eliminar filas donde todos los valores OHLCV sean NaN
        df.dropna(how="all", inplace=True)

        # Paso 5: Ordenar cronológicamente (de más antiguo a más reciente)
        df.sort_index(inplace=True)

        logger.info(
            "Parseadas %d filas para '%s' (%s → %s).",
            len(df), ticker, df.index.min(), df.index.max(),
        )
        return df

    def _save_csv(self, df: pd.DataFrame, ticker: str, resample_freq: str) -> Path:
        """
        Guarda un DataFrame como fichero CSV en la carpeta data/raw/.

        El nombre del fichero sigue el patrón: {ticker}_{frecuencia}.csv
        Ejemplo: data/raw/AAPL_4Hour.csv

        Si la carpeta data/raw/ no existe, la crea automáticamente.

        Args:
            df            (pd.DataFrame): DataFrame a guardar.
            ticker        (str):          Símbolo del activo (ej. 'AAPL').
            resample_freq (str):          Clave de frecuencia (ej. '4Hour').

        Returns:
            Path: Ruta completa al fichero CSV guardado.
        """
        # Crear la carpeta de destino si no existe
        carpeta_destino = Path("data/raw")
        carpeta_destino.mkdir(parents=True, exist_ok=True)

        # Construir el nombre del fichero con el patrón definido
        ruta_csv = carpeta_destino / f"{ticker}_{resample_freq}.csv"

        # Guardar el DataFrame incluyendo el índice (datetime)
        df.to_csv(ruta_csv)

        logger.info("CSV guardado → %s  (%d filas)", ruta_csv, len(df))
        return ruta_csv


# ---------------------------------------------------------------------------
# BLOQUE DE PRUEBA RÁPIDA (smoke-test)
#
# Se ejecuta únicamente cuando lanzas el script directamente:
#   python -m src.data.tiingo_loader
#
# Descarga datos de SPY para el primer trimestre de 2024 con velas de 4 horas,
# guarda el CSV en data/raw/SPY_4Hour.csv y muestra las 10 primeras filas.
# Sirve para verificar que la API Key, la conexión y el parseo funcionan bien.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Instanciar el loader (carga .env y settings.yaml automáticamente)
    loader = TiingoLoader()

    # Ticker de prueba: SPY (ETF del S&P 500, muy líquido y siempre disponible)
    ticker_prueba = "SPY"

    # Descargar velas de 4 horas para el Q1 2024
    df_test = loader.download_ticker(
        ticker=ticker_prueba,
        start_date="2024-01-01",
        end_date="2024-03-31",
        resample_freq="4Hour",
    )

    if not df_test.empty:
        # Guardar el CSV y mostrar un resumen por pantalla
        loader._save_csv(df_test, ticker_prueba, "4Hour")
        print("\n--- Primeras 10 filas del DataFrame ---")
        print(df_test.head(10))
        print(f"\nDimensiones del DataFrame : {df_test.shape}")
        print(f"Tipo del índice (datetime): {df_test.index.dtype}")
        print(f"Zona horaria del índice   : {df_test.index.tz}")
    else:
        print(
            "No se obtuvieron datos. "
            "Comprueba tu API Key y la conexión a internet."
        )
