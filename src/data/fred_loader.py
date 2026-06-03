"""
Módulo para la descarga de datos macroeconómicos desde la API de FRED.

FRED (Federal Reserve Economic Data) es la base de datos del Banco de la
Reserva Federal de St. Louis. Contiene más de 800.000 series temporales
económicas y financieras de libre acceso.

Este módulo descarga los indicadores macroeconómicos y de sentimiento de
mercado definidos en config/settings.yaml, los unifica en un único DataFrame
y los exporta a data/raw/macro_data.csv para su uso en el pipeline del TFM.

Series descargadas por defecto:
    - CPIAUCSL : Índice de Precios al Consumidor (Inflación mensual)
    - FEDFUNDS : Tipo de interés objetivo de la Reserva Federal
    - VIXCLS   : Índice VIX (Volatilidad implícita / Sentimiento de mercado)
    - UNRATE   : Tasa de desempleo en EE.UU.
    - T10Y2Y   : Diferencial del bono 10Y-2Y (indicador de recesión)

Librería utilizada: fredapi (wrapper oficial para Python)
    pip install fredapi
Documentación oficial: https://fred.stlouisfed.org/docs/api/fred/
"""

import os
import logging
import yaml
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from fredapi import Fred

# ---------------------------------------------------------------------------
# Configuración del sistema de logs
# Muestra: fecha, nivel (INFO/WARNING/ERROR), nombre del logger y mensaje.
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("FredLoader")


def load_config(config_path: str = "config/settings.yaml") -> dict:
    """
    Carga la configuración del proyecto desde el fichero YAML indicado.

    Lee settings.yaml para obtener las fechas de inicio/fin del análisis
    y la lista de series de FRED que se deben descargar.

    Args:
        config_path (str): Ruta al fichero YAML. Por defecto 'config/settings.yaml'.

    Returns:
        dict: Diccionario con toda la configuración parseada.

    Raises:
        FileNotFoundError: Si el fichero de configuración no existe.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Fichero de configuración no encontrado: {config_path}"
        )
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class FredLoader:
    """
    Clase encargada de descargar series temporales macroeconómicas desde FRED.

    Flujo de uso típico:
        1. Instanciar la clase → carga .env y settings.yaml automáticamente.
        2. Llamar a download_all_series() para descargar y unificar todos
           los indicadores definidos en la configuración de una vez.
        3. El resultado se guarda en data/raw/macro_data.csv.

    Ejemplo rápido:
        loader = FredLoader()
        df_macro = loader.download_all_series()

    La clave API se lee de la variable de entorno FRED_API_KEY,
    que debe estar definida en el fichero .env local (nunca en el repositorio).

    Nota sobre la frecuencia de los datos:
        FRED mezcla series con frecuencias distintas (mensual, diaria…).
        Tras la descarga, se aplica forward-fill para alinear todas las
        series a frecuencia diaria, propagando el último valor conocido.
    """

    # Nombre del fichero CSV de salida con todos los indicadores unificados
    OUTPUT_FILENAME = "macro_data.csv"

    def __init__(self, config_path: str = "config/settings.yaml"):
        """
        Inicializa el FredLoader.

        Pasos que realiza:
            1. Carga las variables del fichero .env en el entorno del sistema.
            2. Lee FRED_API_KEY y valida que esté definida.
            3. Crea la instancia del cliente oficial de fredapi con la clave.
            4. Carga la configuración del proyecto desde settings.yaml.

        Args:
            config_path (str): Ruta al fichero de configuración YAML.

        Raises:
            ValueError: Si FRED_API_KEY no está definida en el entorno.
        """
        # Paso 1: Cargar variables de entorno desde el .env local
        load_dotenv()

        # Paso 2: Leer y validar la clave API de FRED
        self.api_key = os.getenv("FRED_API_KEY")
        if not self.api_key:
            raise ValueError(
                "FRED_API_KEY no está definida. Añádela a tu fichero .env."
            )

        # Paso 3: Crear el cliente de fredapi con la clave de autenticación
        # A partir de aquí, self.fred es el objeto con el que consultamos FRED
        self.fred = Fred(api_key=self.api_key)

        # Paso 4: Leer la configuración del proyecto (fechas, series de FRED…)
        self.config = load_config(config_path)

        logger.info(
            "FredLoader inicializado correctamente. "
            "Configuración cargada desde '%s'.",
            config_path,
        )

    # ------------------------------------------------------------------
    # MÉTODOS PÚBLICOS
    # ------------------------------------------------------------------

    def download_series(self, series_id: str, start_date: str, end_date: str) -> pd.Series:
        """
        Descarga UNA única serie temporal de FRED.

        Utiliza el método get_series() de fredapi, que devuelve directamente
        un objeto pandas.Series con el índice de tipo DatetimeIndex.

        Args:
            series_id  (str): Identificador de la serie en FRED, por ejemplo
                              'CPIAUCSL' (Inflación) o 'FEDFUNDS' (tipos Fed).
            start_date (str): Fecha de inicio en formato YYYY-MM-DD.
            end_date   (str): Fecha de fin   en formato YYYY-MM-DD.

        Returns:
            pd.Series: Serie temporal con DatetimeIndex. El nombre de la
                       Serie es el propio series_id (ej. 'CPIAUCSL').
                       Devuelve una Serie vacía si ocurre algún error.
        """
        logger.info(
            "Descargando serie FRED: %s | %s → %s",
            series_id, start_date, end_date,
        )

        try:
            # fredapi devuelve un pd.Series con DatetimeIndex automáticamente
            serie = self.fred.get_series(
                series_id,
                observation_start=start_date,
                observation_end=end_date,
            )
            # Asignar el ID de la serie como nombre para identificarla en el DataFrame
            serie.name = series_id

            logger.info(
                "Serie '%s' descargada: %d observaciones (%s → %s).",
                series_id, len(serie), serie.index.min().date(), serie.index.max().date(),
            )
            return serie

        except Exception as e:
            # Captura cualquier error de FRED: serie inexistente, sin conexión, etc.
            logger.error(
                "Error al descargar la serie '%s' de FRED: %s", series_id, e
            )
            return pd.Series(name=series_id, dtype=float)

    def download_all_series(self) -> pd.DataFrame:
        """
        Descarga TODAS las series macroeconómicas definidas en settings.yaml
        y las unifica en un único DataFrame indexado por fecha.

        Pasos que realiza:
            1. Lee la lista de series y el rango de fechas de settings.yaml.
            2. Descarga cada serie de forma individual con download_series().
            3. Concatena todas las series en un único DataFrame (columna = serie).
            4. Reindexa a frecuencia diaria y aplica forward-fill para rellenar
               los valores faltantes entre observaciones (ej. datos mensuales
               se propagan día a día hasta la siguiente observación).
            5. Elimina filas iniciales donde todas las series sean NaN.
            6. Guarda el resultado en data/raw/macro_data.csv.

        Returns:
            pd.DataFrame: DataFrame con DatetimeIndex diario y una columna
                          por cada serie descargada (ej. CPIAUCSL, FEDFUNDS…).
                          Devuelve DataFrame vacío si no hay series configuradas.
        """
        # Leer parámetros de la configuración
        series_ids = self.config.get("data_sources", {}).get("fred", {}).get("series", [])
        start_date = self.config.get("start_date", "2018-01-01")
        end_date   = self.config.get("end_date",   "2026-05-01")

        if not series_ids:
            logger.warning(
                "No se encontraron series FRED en settings.yaml. "
                "Revisa la clave 'data_sources.fred.series'."
            )
            return pd.DataFrame()

        logger.info(
            "Iniciando descarga de %d series FRED | %s → %s",
            len(series_ids), start_date, end_date,
        )

        # Paso 2: Descargar cada serie individualmente
        series_descargadas = []
        for series_id in series_ids:
            serie = self.download_series(series_id, start_date, end_date)
            if not serie.empty:
                series_descargadas.append(serie)
            else:
                logger.warning("La serie '%s' se omitirá por estar vacía.", series_id)

        if not series_descargadas:
            logger.error("No se pudo descargar ninguna serie de FRED.")
            return pd.DataFrame()

        # Paso 3: Concatenar todas las series en un único DataFrame
        # Cada serie se convierte en una columna; el índice es la unión de fechas
        df_macro = pd.concat(series_descargadas, axis=1)
        df_macro.index.name = "fecha"

        # Paso 4: Reindexar a frecuencia diaria y aplicar forward-fill
        # Esto es necesario porque las series tienen frecuencias distintas:
        #   - FEDFUNDS / CPIAUCSL → mensuales (solo ~12 obs/año)
        #   - VIXCLS              → diaria
        # Con resample + ffill, las series mensuales se propagan a diario
        indice_diario = pd.date_range(
            start=df_macro.index.min(),
            end=df_macro.index.max(),
            freq="D",
        )
        df_macro = df_macro.reindex(indice_diario)
        df_macro = df_macro.ffill()   # Forward-fill: propaga el último valor conocido
        df_macro.index.name = "fecha"

        # Paso 5: Eliminar filas iniciales donde TODAS las columnas sean NaN
        # (ocurre cuando las series no arrancan en la misma fecha)
        df_macro.dropna(how="all", inplace=True)

        logger.info(
            "DataFrame macro construido: %d filas × %d columnas | %s → %s",
            df_macro.shape[0], df_macro.shape[1],
            df_macro.index.min().date(), df_macro.index.max().date(),
        )

        # Paso 6: Guardar el resultado como CSV
        self._save_csv(df_macro)

        return df_macro

    # ------------------------------------------------------------------
    # MÉTODOS PRIVADOS (helpers internos, no se llaman desde fuera)
    # ------------------------------------------------------------------

    def _save_csv(self, df: pd.DataFrame) -> Path:
        """
        Guarda el DataFrame macroeconómico unificado en data/raw/macro_data.csv.

        Si la carpeta data/raw/ no existe, la crea automáticamente.
        El fichero siempre se llama 'macro_data.csv' (definido en OUTPUT_FILENAME).

        Args:
            df (pd.DataFrame): DataFrame con todos los indicadores FRED unificados.

        Returns:
            Path: Ruta completa al fichero CSV guardado.
        """
        # Crear la carpeta de destino si no existe
        carpeta_destino = Path("data/raw")
        carpeta_destino.mkdir(parents=True, exist_ok=True)

        # Construir la ruta completa del fichero de salida
        ruta_csv = carpeta_destino / self.OUTPUT_FILENAME

        # Guardar el DataFrame incluyendo el índice de fechas
        df.to_csv(ruta_csv)

        logger.info(
            "CSV macroeconómico guardado → %s  (%d filas × %d columnas)",
            ruta_csv, df.shape[0], df.shape[1],
        )
        return ruta_csv


# ---------------------------------------------------------------------------
# BLOQUE DE PRUEBA RÁPIDA (smoke-test)
#
# Se ejecuta únicamente cuando lanzas el script directamente:
#   python -m src.data.fred_loader
#
# Descarga las 5 series macroeconómicas definidas en settings.yaml para el
# año 2023, las unifica en un DataFrame diario con forward-fill y muestra
# un resumen por pantalla. Sirve para verificar que la API Key de FRED,
# la conexión y el procesado de datos funcionan correctamente.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Instanciar el loader (carga .env y settings.yaml automáticamente)
    loader = FredLoader()

    # Descargar SOLO el año 2023 para el test (descarga rápida)
    print("\n=== SMOKE-TEST: Descargando series FRED para 2023 ===\n")

    # 1. Test de una serie individual: VIX (disponibilidad diaria, ideal para test)
    print("--- Test 1: Descarga individual de VIXCLS (VIX) ---")
    serie_vix = loader.download_series(
        series_id="VIXCLS",
        start_date="2023-01-01",
        end_date="2023-03-31",
    )
    if not serie_vix.empty:
        print(serie_vix.head(5))
        print(f"Observaciones: {len(serie_vix)} | Frecuencia: diaria\n")
    else:
        print("ERROR: No se obtuvieron datos para VIXCLS.\n")

    # 2. Test completo: todas las series unificadas desde settings.yaml
    print("--- Test 2: Descarga y unificación de todas las series ---")
    df_macro = loader.download_all_series()

    if not df_macro.empty:
        print("\n--- Primeras 10 filas del DataFrame macroeconómico ---")
        print(df_macro.head(10))
        print(f"\nDimensiones               : {df_macro.shape}")
        print(f"Columnas (series)         : {list(df_macro.columns)}")
        print(f"Tipo del índice           : {df_macro.index.dtype}")
        print(f"Rango temporal            : {df_macro.index.min().date()} -> {df_macro.index.max().date()}")
        print(f"\nValores nulos por columna :")
        print(df_macro.isnull().sum())
    else:
        print(
            "No se obtuvieron datos. "
            "Comprueba tu API Key de FRED y la conexión a internet."
        )
