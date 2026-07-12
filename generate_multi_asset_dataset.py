"""
Script Orquestador Multi-Activo: Generación del Dataset Etiquetado para ML.

Reutiliza la infraestructura existente del proyecto (TiingoLoader, MTFBuilder)
para generar un dataset etiquetado grande y robusto con 25 activos del S&P 500.

Flujo por cada ticker:
    1. TiingoLoader  → descarga OHLCV en 4H, 1D, 1W desde la API de Tiingo.
    2. MTFBuilder    → construye el dataset Multi-Timeframe con indicadores técnicos.
    3. PriceAction   → añade fractales, wick reclaims, niveles de Fibonacci.
    4. TierEvaluator → clasifica señales en Tier A, B o C.
    5. StrictBacktester → etiqueta cada señal (1=TP alcanzado, 0=SL alcanzado).
    6. Concatena todo en MULTI_LABELED_DATASET.csv.

Uso:
    python generate_multi_asset_dataset.py
    python generate_multi_asset_dataset.py --tickers AAPL MSFT GOOGL
"""

import argparse
import logging
import warnings
import pandas as pd
from pathlib import Path
from datetime import datetime

from src.data.tiingo_loader import TiingoLoader
from src.data.mtf_builder import MTFBuilder
from src.features.patterns import add_price_action_features
from src.models.tier_evaluator import TierEvaluator
from src.environment.backtester import StrictBacktester

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("MultiAssetGenerator")

# ---------------------------------------------------------------------------
# ACTIVOS A EXCLUIR del dataset de ML:
# - Índices ETF (SPY, QQQ...) → no son acciones individuales
# - Bonos y Commodities (TLT, GLD...) → distinto comportamiento técnico
# - Acciones europeas/asiáticas (ASML, TSM, NVO, RIO, TM) → Tiingo IEX no las cubre bien
# - Cripto-proxies de alta especulación (MSTR, COIN) → distorsionan patrones técnicos
# ---------------------------------------------------------------------------
EXCLUDE = {
    # ETFs e Índices
    "SPY", "QQQ", "DIA", "IWM",
    # Bonos y Renta Fija
    "TLT", "IEF",
    # Commodities / Metales
    "GLD", "SLV",
    # Acciones fuera del mercado US (IEX de Tiingo no las cubre)
    "TSM", "ASML", "NVO", "RIO", "TM",
    # Cripto-proxies especulativos
    "MSTR", "COIN",
    # Problemas de ticker con guiones
    "BRK-B",
}


def load_tickers_from_settings(config_path: str = "config/settings.yaml") -> list:
    """
    Lee la lista de tickers del fichero settings.yaml y filtra los benchmarks
    y activos no aptos para el análisis técnico de acciones individuales.

    Args:
        config_path (str): Ruta al fichero de configuración YAML.

    Returns:
        list: Lista de tickers válidos para el dataset de ML.
    """
    import yaml
    path = Path(config_path)
    if not path.exists():
        logger.error("No se encuentra settings.yaml en: %s", path)
        return []

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    all_tickers = config.get("tickers", [])
    filtered    = [t for t in all_tickers if t not in EXCLUDE]

    logger.info(
        "Tickers cargados de settings.yaml: %d total → %d válidos para ML (excluidos: %d)",
        len(all_tickers), len(filtered), len(all_tickers) - len(filtered)
    )
    return filtered

# Directorios de datos
DIR_RAW       = Path("data/raw")
DIR_PROCESSED = Path("data/processed")


def process_ticker(
    ticker: str,
    start_date: str,
    end_date: str,
    loader: TiingoLoader,
    force_download: bool = False,
    delay: float = 2.0,
) -> pd.DataFrame:
    """
    Ejecuta el pipeline completo de descarga y etiquetado para un único activo.

    Reutiliza los módulos existentes del proyecto:
        1. TiingoLoader  → descarga datos OHLCV (4H, 1D, 1W) si no están en disco.
        2. MTFBuilder    → construye el dataset Multi-Timeframe con indicadores.
        3. PriceAction   → añade patrones (fractales, wick reclaims, Fibonacci).
        4. TierEvaluator → clasifica señales en Tiers A, B, C.
        5. StrictBacktester → etiqueta las señales (ganadora/perdedora).

    Args:
        ticker         (str):          Símbolo del activo (ej. 'AAPL').
        start_date     (str):          Fecha de inicio 'YYYY-MM-DD'.
        end_date       (str):          Fecha de fin 'YYYY-MM-DD'.
        loader         (TiingoLoader): Instancia del loader con la API key ya cargada.
        force_download (bool):         Si True, descarga aunque el CSV ya exista en disco.

    Returns:
        pd.DataFrame: Dataset etiquetado con la columna 'Ticker' añadida.
                      Vacío si falla algún paso.
    """
    logger.info("▶ Procesando: %s", ticker)

    try:
        # ------------------------------------------------------------------
        # 1. Descargar OHLCV 4H con TiingoLoader (el endpoint IEX solo soporta intradiario)
        #    Para 1Day y 1Week hacemos resample desde el 4H para evitar errores 400.
        # ------------------------------------------------------------------
        raw_4h_path = DIR_RAW / f"{ticker}_4Hour.csv"
        if raw_4h_path.exists() and not force_download:
            logger.info("  [%s] CSV 4Hour ya existe — usando caché.", ticker)
            df_4h = pd.read_csv(raw_4h_path, index_col="datetime", parse_dates=True)
            if df_4h.index.tz is None:
                df_4h.index = df_4h.index.tz_localize("UTC")
            else:
                df_4h.index = df_4h.index.tz_convert("UTC")
        else:
            logger.info("  [%s] Descargando 4Hour desde Tiingo...", ticker)
            df_4h = loader.download_ticker(ticker, start_date, end_date, "4Hour")
            if df_4h.empty:
                logger.warning("  [%s] Descarga 4H vacía — saltando ticker.", ticker)
                return pd.DataFrame()
            raw_4h_path.parent.mkdir(parents=True, exist_ok=True)
            df_4h.to_csv(raw_4h_path)
            logger.info("  [%s] 4Hour guardado (%d filas).", ticker, len(df_4h))
            # Pausa para respetar el rate limit de Tiingo (máx ~1 req/seg en plan free)
            import time
            time.sleep(delay)

        # Generar 1Day y 1Week mediante resample desde el 4H
        # (evita el error 400 del endpoint IEX para frecuencias diarias/semanales)
        for tf, rule in [("1Day", "1D"), ("1Week", "1W")]:
            raw_path = DIR_RAW / f"{ticker}_{tf}.csv"
            if raw_path.exists() and not force_download:
                logger.info("  [%s] CSV %s ya existe — usando caché.", ticker, tf)
            else:
                logger.info("  [%s] Generando %s por resample desde 4H...", ticker, tf)
                df_resampled = df_4h.resample(rule).agg({
                    "open": "first", "high": "max",
                    "low": "min", "close": "last", "volume": "sum"
                }).dropna()
                df_resampled.to_csv(raw_path)
                logger.info("  [%s] %s guardado (%d filas).", ticker, tf, len(df_resampled))

        # ------------------------------------------------------------------
        # 2. Construir dataset MTF con MTFBuilder (reutiliza la lógica existente)
        # ------------------------------------------------------------------
        mtf_path = DIR_PROCESSED / f"{ticker}_4H_MTF.csv"
        if mtf_path.exists() and not force_download:
            logger.info("  [%s] MTF ya existe — cargando desde disco.", ticker)
            df = pd.read_csv(mtf_path, index_col="datetime", parse_dates=True)
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            else:
                df.index = df.index.tz_convert("UTC")
        else:
            logger.info("  [%s] Construyendo dataset MTF...", ticker)
            mtf_builder = MTFBuilder(ticker=ticker, data_dir=str(DIR_RAW))
            df = mtf_builder.build(guardar_csv=True)

        if df is None or df.empty:
            logger.warning("  [%s] MTF vacío — saltando.", ticker)
            return pd.DataFrame()

        logger.info("  [%s] MTF cargado: %d filas, %d columnas.", ticker, len(df), len(df.columns))

        # ------------------------------------------------------------------
        # 3. Añadir patrones de Price Action
        # ------------------------------------------------------------------
        df = add_price_action_features(df)

        # ------------------------------------------------------------------
        # 4. Evaluar Tiers
        # ------------------------------------------------------------------
        evaluator = TierEvaluator()
        df = evaluator.evaluate_dataframe(df)

        signals_df = df[df["Tier"].notna()].copy()
        logger.info("  [%s] Señales detectadas: %d", ticker, len(signals_df))

        if signals_df.empty:
            logger.warning("  [%s] Sin señales — saltando.", ticker)
            return pd.DataFrame()

        # ------------------------------------------------------------------
        # 5. Ejecutar el Backtester estricto
        # ------------------------------------------------------------------
        backtester = StrictBacktester()
        df_labeled = backtester.run(df_prices=df, signals_df=signals_df)

        if df_labeled.empty:
            logger.warning("  [%s] Backtester sin resultados — saltando.", ticker)
            return pd.DataFrame()

        # ------------------------------------------------------------------
        # 6. Añadir columna 'Ticker' para identificar el activo
        # ------------------------------------------------------------------
        df_labeled.insert(0, "Ticker", ticker)

        wins  = int(df_labeled["Label"].sum())
        total = len(df_labeled)
        wr    = wins / total * 100 if total > 0 else 0
        logger.info("  [%s] ✅ %d señales | Win Rate: %.1f%%", ticker, total, wr)

        return df_labeled

    except Exception as e:
        logger.error("  [%s] ❌ Error inesperado: %s", ticker, e, exc_info=True)
        return pd.DataFrame()


def run(
    tickers: list,
    start_date: str,
    end_date: str,
    output_dir: str = "data/processed",
    force_download: bool = False,
):
    """
    Orquesta el procesamiento de todos los activos y genera el dataset consolidado.

    Args:
        tickers        (list): Lista de símbolos a procesar.
        start_date     (str):  Fecha de inicio 'YYYY-MM-DD'.
        end_date       (str):  Fecha de fin 'YYYY-MM-DD'.
        output_dir     (str):  Directorio de salida para el dataset final.
        force_download (bool): Si True, re-descarga aunque los CSVs ya existan.
    """
    logger.info("=" * 65)
    logger.info("INICIO — Multi-Asset Dataset Generator")
    logger.info("Universo: %d activos | %s → %s", len(tickers), start_date, end_date)
    logger.info("=" * 65)

    # Instanciar el loader una sola vez (una sola lectura del .env y del YAML)
    try:
        loader = TiingoLoader()
    except ValueError as e:
        logger.error("Error al inicializar TiingoLoader: %s", e)
        return

    all_datasets  = []
    summary_rows  = []
    failed_tickers = []

    for i, ticker in enumerate(tickers, 1):
        logger.info("[%d/%d] ─── %s ───", i, len(tickers), ticker)
        df_ticker = process_ticker(
            ticker, start_date, end_date, loader, force_download, delay=2.0
        )

        if not df_ticker.empty:
            all_datasets.append(df_ticker)
            summary_rows.append({
                "Ticker"    : ticker,
                "Señales"   : len(df_ticker),
                "Ganadoras" : int(df_ticker["Label"].sum()),
                "Win Rate"  : f"{df_ticker['Label'].mean() * 100:.1f}%",
                "Tier A"    : int((df_ticker["Tier"] == "A").sum()),
                "Tier B"    : int((df_ticker["Tier"] == "B").sum()),
                "Tier C"    : int((df_ticker["Tier"] == "C").sum()),
            })
        else:
            failed_tickers.append(ticker)

    if not all_datasets:
        logger.error("No se generaron datos para ningún activo. Revisa la API key de Tiingo.")
        return

    # Concatenar todos los datasets en uno único
    df_final = pd.concat(all_datasets, axis=0)
    df_final.sort_index(inplace=True)

    # Exportar el dataset consolidado
    out_path = Path(output_dir) / "MULTI_LABELED_DATASET.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(out_path)

    # -------------------------------------------------------------------------
    # RESUMEN EJECUTIVO
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  RESUMEN FINAL — MULTI-ASSET LABELED DATASET")
    print("=" * 70)
    print(f"  Activos procesados con éxito : {len(all_datasets):>3} / {len(tickers)}")
    if failed_tickers:
        print(f"  Activos fallidos             : {', '.join(failed_tickers)}")
    print(f"  Total señales etiquetadas    : {len(df_final):>8}")
    print(f"  Ganadoras (Label=1, SL4+TP4): {int(df_final['Label'].sum()):>8}")
    print(f"  Perdedoras (Label=0, SL4+TP4): {int((df_final['Label']==0).sum()):>8}")
    print(f"  Win Rate Global              : {df_final['Label'].mean()*100:>7.1f}%")

    print("\n  --- Desglose por Activo ---")
    df_summary = pd.DataFrame(summary_rows)
    print(df_summary.to_string(index=False))

    print("\n  --- Win Rate por Tier (dataset global) ---")
    if "Tier" in df_final.columns:
        tier_stats = df_final.groupby("Tier")["Label"].agg(["count", "mean"])
        for tier, row in tier_stats.iterrows():
            print(f"    Tier {tier:<10}: {row['mean']*100:.1f}%  ({int(row['count'])} señales)")

    print(f"\n  Dataset guardado en: {out_path}")
    print("=" * 70)

    return df_final


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Genera un dataset etiquetado multi-activo para los modelos de ML."
    )
    parser.add_argument(
        "--start", type=str, default="2018-01-01",
        help="Fecha de inicio (YYYY-MM-DD). Por defecto: 2018-01-01."
    )
    parser.add_argument(
        "--end", type=str, default=datetime.today().strftime("%Y-%m-%d"),
        help="Fecha de fin (YYYY-MM-DD). Por defecto: hoy."
    )
    parser.add_argument(
        "--tickers", type=str, nargs="+", default=None,
        help="Lista de tickers (opcional). Por defecto: universo de 25 activos."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Fuerza la re-descarga aunque los CSVs ya existan en disco."
    )
    args = parser.parse_args()

    tickers_to_use = args.tickers if args.tickers else load_tickers_from_settings()
    run(
        tickers=tickers_to_use,
        start_date=args.start,
        end_date=args.end,
        force_download=args.force,
    )
