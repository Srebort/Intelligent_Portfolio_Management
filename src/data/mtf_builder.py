"""
Módulo de construcción del dataset Multi-Timeframe (MTF).

Descarga o carga los datos de precio en las tres temporalidades de la
estrategia (4H operativa, 1D tendencia media, 1W tendencia macro), calcula
los indicadores técnicos en cada una y los propaga a la temporalidad
operativa de 4H mediante un merge asof + ffill.

El resultado es un único DataFrame de velas de 4H que incluye columnas
sufijadas por temporalidad:
    - Columnas nativas 4H   : close, SMA_200, RSI_14, dist_SMA_200, ...
    - Columnas de 1D (sufijo _1D) : close_1D, SMA_200_1D, dist_SMA_200_1D, ...
    - Columnas de 1W (sufijo _1W) : close_1W, SMA_200_1W, slope_SMA_200_1W, ...

Con estas columnas el TierEvaluator puede aplicar el filtro base MTF real:
    close_4H > SMA_200    AND    close_1D > SMA_200_1D    AND    slope_SMA_200_1W > 0

ANTI-LOOKAHEAD BIAS:
    El merge usa pd.merge_asof con direction='backward': solo propaga el
    último valor conocido hasta el instante de la vela de 4H, nunca valores
    del futuro. Equivale a lo que un trader vería en tiempo real.
"""

import logging
import pandas as pd
from pathlib import Path

logger = logging.getLogger("MTFBuilder")


class MTFBuilder:
    """
    Construye el dataset Multi-Timeframe unificado para un único ticker,
    propagando indicadores de temporalidades superiores (1D, 1W) a las
    velas operativas de 4H.
    """

    def __init__(self, ticker: str, data_dir: str = "data/raw"):
        """
        Inicializa el builder con el ticker y la ruta a los CSVs.

        Args:
            ticker   (str): Símbolo del activo (ej. 'SPY').
            data_dir (str): Directorio donde se encuentran los CSVs de Tiingo.
        """
        self.ticker   = ticker
        self.data_dir = Path(data_dir)

    # ------------------------------------------------------------------
    # CARGA DE DATOS
    # ------------------------------------------------------------------

    def _cargar_csv(self, timeframe: str) -> pd.DataFrame:
        """
        Carga el CSV de precios OHLCV para el ticker y la temporalidad dada.
        Espera que el fichero haya sido generado previamente por tiingo_loader.py.

        Args:
            timeframe (str): Temporalidad del fichero (ej. '4Hour', '1Day', '1Week').

        Returns:
            pd.DataFrame: DataFrame con índice DatetimeTZAware (UTC) o vacío si no existe.
        """
        ruta = self.data_dir / f"{self.ticker}_{timeframe}.csv"
        if not ruta.exists():
            logger.warning("No se encuentra el fichero %s. Ejecuta tiingo_loader.py.", ruta)
            return pd.DataFrame()

        df = pd.read_csv(ruta, index_col="datetime", parse_dates=True)

        # Asegurar que el índice tiene zona horaria UTC
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

        df.sort_index(inplace=True)
        logger.info("Cargado %s | %d filas | %s -> %s", ruta.name, len(df),
                    df.index[0].date(), df.index[-1].date())
        return df

    # ------------------------------------------------------------------
    # CÁLCULO DE INDICADORES POR TEMPORALIDAD
    # ------------------------------------------------------------------

    def _calcular_indicadores(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Aplica el pipeline de features técnicas al DataFrame de una temporalidad.
        Reutiliza add_all_features() de technical.py.

        Args:
            df        (pd.DataFrame): DataFrame OHLCV de la temporalidad.
            timeframe (str):          Nombre descriptivo para los logs.

        Returns:
            pd.DataFrame: DataFrame enriquecido con indicadores técnicos.
        """
        if df.empty:
            return df

        from src.features.technical import add_all_features
        df_feat = add_all_features(df, timeframe=timeframe)
        return df_feat

    # ------------------------------------------------------------------
    # PROPAGACIÓN MTF: merge_asof + ffill
    # ------------------------------------------------------------------

    def _propagar_timeframe(
        self,
        df_base: pd.DataFrame,
        df_superior: pd.DataFrame,
        sufijo: str,
        columnas: list,
    ) -> pd.DataFrame:
        """
        Propaga columnas de una temporalidad superior al DataFrame base (4H)
        usando merge_asof con direction='backward' para garantizar que en
        cada vela de 4H solo se usa información ya disponible (anti-lookahead).

        Args:
            df_base     (pd.DataFrame): DataFrame de 4H (temporalidad operativa).
            df_superior (pd.DataFrame): DataFrame de 1D o 1W con indicadores.
            sufijo      (str):          Sufijo para renombrar columnas (ej. '_1D').
            columnas    (list):         Lista de columnas a propagar desde df_superior.

        Returns:
            pd.DataFrame: df_base con las columnas del timeframe superior añadidas.
        """
        if df_superior.empty:
            logger.warning("DataFrame de temporalidad superior vacío. No se propagan columnas %s.", sufijo)
            return df_base

        # Seleccionar solo las columnas necesarias y renombrarlas con sufijo
        columnas_existentes = [c for c in columnas if c in df_superior.columns]
        if not columnas_existentes:
            logger.warning("Ninguna de las columnas %s está disponible en el DataFrame superior.", columnas)
            return df_base

        df_sel = df_superior[columnas_existentes].copy()
        df_sel.columns = [f"{col}{sufijo}" for col in df_sel.columns]

        # Resetear índice para merge_asof (necesita columna, no índice)
        df_base_reset = df_base.reset_index()
        df_sel_reset  = df_sel.reset_index()

        # Asegurar nombres de columna del índice uniformes
        df_base_reset.rename(columns={"datetime": "datetime"}, inplace=True)
        df_sel_reset.rename(columns={"datetime": "datetime"}, inplace=True)

        # merge_asof: para cada vela 4H, toma el último valor conocido de 1D/1W
        # direction='backward' = solo usa datos del pasado (sin lookahead bias)
        merged = pd.merge_asof(
            df_base_reset.sort_values("datetime"),
            df_sel_reset.sort_values("datetime"),
            on="datetime",
            direction="backward",
            tolerance=pd.Timedelta("8 days"),  # máximo gap aceptable
        )

        # Restaurar el índice datetime
        merged.set_index("datetime", inplace=True)
        merged.sort_index(inplace=True)

        logger.info(
            "Propagadas %d columnas con sufijo '%s' al DataFrame 4H.",
            len(columnas_existentes), sufijo,
        )
        return merged

    # ------------------------------------------------------------------
    # MÉTODO PRINCIPAL
    # ------------------------------------------------------------------

    def build(self, guardar_csv: bool = True) -> pd.DataFrame:
        """
        Construye el DataFrame Multi-Timeframe completo para el ticker.

        Pasos:
            1. Carga datos OHLCV en 4H, 1D y 1W desde los CSVs.
            2. Calcula indicadores técnicos en cada temporalidad.
            3. Propaga columnas de 1D y 1W al DataFrame de 4H.
            4. Guarda el resultado en data/processed/<TICKER>_4H_MTF.csv.

        Args:
            guardar_csv (bool): Si True, exporta el resultado a data/processed/.

        Returns:
            pd.DataFrame: Dataset MTF completo indexado por datetime UTC.
        """
        logger.info("=== Construyendo dataset MTF para '%s' ===", self.ticker)

        # 1. Cargar datos en las tres temporalidades
        df_4h = self._cargar_csv("4Hour")
        df_1d = self._cargar_csv("1Day")
        df_1w = self._cargar_csv("1Week")

        if df_4h.empty:
            logger.error("Sin datos de 4H para '%s'. Abortando build.", self.ticker)
            return pd.DataFrame()

        # 2. Calcular indicadores en cada temporalidad
        df_4h_feat = self._calcular_indicadores(df_4h, timeframe="4Hour")
        df_1d_feat = self._calcular_indicadores(df_1d, timeframe="1Day")
        df_1w_feat = self._calcular_indicadores(df_1w, timeframe="1Week")

        # 3. Definir qué columnas propagar de cada temporalidad superior
        # Columnas 1D: filtro de tendencia diario + contexto RSI
        columnas_1d = [
            "close", "SMA_200", "dist_SMA_200", "slope_SMA_200",
            "RSI_14", "is_bullish_divergence", "SMA_50",
        ]
        # Columnas 1W: filtro de tendencia macro (slope es el más importante)
        columnas_1w = [
            "close", "SMA_200", "slope_SMA_200", "dist_SMA_200",
        ]

        # 4. Propagar al DataFrame de 4H con sufijos _1D y _1W
        df_mtf = df_4h_feat.copy()

        if not df_1d_feat.empty:
            df_mtf = self._propagar_timeframe(df_mtf, df_1d_feat, sufijo="_1D", columnas=columnas_1d)

        if not df_1w_feat.empty:
            df_mtf = self._propagar_timeframe(df_mtf, df_1w_feat, sufijo="_1W", columnas=columnas_1w)

        # 5. Guardar resultado
        if guardar_csv:
            ruta_out = Path("data/processed") / f"{self.ticker}_4H_MTF.csv"
            ruta_out.parent.mkdir(parents=True, exist_ok=True)
            df_mtf.to_csv(ruta_out)
            logger.info("Dataset MTF guardado -> %s (%d filas, %d columnas)",
                        ruta_out, len(df_mtf), len(df_mtf.columns))

        return df_mtf


# ---------------------------------------------------------------------------
# SMOKE-TEST
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")

    print("\n=== SMOKE-TEST: MTFBuilder para SPY ===\n")

    builder = MTFBuilder(ticker="SPY")
    df = builder.build(guardar_csv=True)

    if df.empty:
        print("ERROR: Dataset MTF vacio. Descarga primero los CSVs de 1D y 1W con tiingo_loader.py")
        exit(1)

    print(f"Dimensiones del dataset MTF : {df.shape}")

    # Columnas MTF clave para el TierEvaluator
    columnas_clave = [
        "close",
        "SMA_200",    "dist_SMA_200",    "slope_SMA_200",
        "close_1D",   "SMA_200_1D",      "slope_SMA_200_1D",
        "close_1W",   "SMA_200_1W",      "slope_SMA_200_1W",
    ]
    disponibles = [c for c in columnas_clave if c in df.columns]
    faltantes   = [c for c in columnas_clave if c not in df.columns]

    print(f"\nColumnas clave presentes : {disponibles}")
    if faltantes:
        print(f"Columnas faltantes       : {faltantes}")
        print("  -> Descarga los CSVs de las temporalidades que faltan.")

    # Verificar que el filtro MTF del TierEvaluator puede activarse
    if "close_1D" in df.columns and "SMA_200_1D" in df.columns and "slope_SMA_200_1W" in df.columns:
        from src.models.tier_evaluator import TierEvaluator
        from src.features.patterns import add_price_action_features

        df_pa  = add_price_action_features(df)
        ev     = TierEvaluator(fib_tolerance=2.0, max_sma_dist=10.0)
        df_ev  = ev.evaluate_dataframe(df_pa)

        signals = df_ev[df_ev["Tier"].notna()]
        print(f"\nSeñales con filtro MTF completo: {len(signals)}")
        print(df_ev[["close", "SMA_200", "close_1D", "SMA_200_1D",
                      "slope_SMA_200_1W", "Tier"]].tail(10).round(3).to_string())
    else:
        print("\nFiltro MTF parcial activo (modo degradado). Descarga 1D/1W para activarlo.")
