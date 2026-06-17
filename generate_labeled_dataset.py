"""
Script Orquestador: Generación del Dataset Etiquetado para Machine Learning.

Ejecuta el pipeline completo:
    1. Carga el dataset limpio y escalado (V_ML_READY.csv).
    2. Evalúa señales con el TierEvaluator.
    3. Pasa las señales por el StrictBacktester (7 SL × 6 TP).
    4. Exporta el dataset etiquetado final (V_LABELED_DATASET.csv).

Uso:
    python generate_labeled_dataset.py
    python generate_labeled_dataset.py --ticker SPY
"""

import argparse
import logging
import pandas as pd
from pathlib import Path

from src.models.tier_evaluator import TierEvaluator
from src.environment.backtester import StrictBacktester

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("GenerateDataset")


def run(ticker: str = "V", data_dir: str = "data/processed"):
    """
    Ejecuta el pipeline completo de generación de dataset etiquetado.

    Args:
        ticker   (str): Símbolo del activo.
        data_dir (str): Directorio con los CSV procesados.
    """
    data_path = Path(data_dir)

    # -----------------------------------------------------------------------
    # 1. Cargar el dataset limpio (sin escalar el precio, escaladas las features)
    # -----------------------------------------------------------------------
    # Usamos el MTF sin escalar para que el backtester trabaje con precios reales
    ml_ready_path = data_path / f"{ticker}_4H_MTF.csv"
    if not ml_ready_path.exists():
        logger.error("No se encuentra el dataset MTF: %s", ml_ready_path)
        return

    df = pd.read_csv(ml_ready_path, index_col="datetime", parse_dates=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    logger.info("Dataset MTF cargado: %d filas, %d columnas.", len(df), len(df.columns))

    # -----------------------------------------------------------------------
    # 2. Añadir patrones de Price Action (necesarios para TierEvaluator y fractales)
    # -----------------------------------------------------------------------
    from src.features.patterns import add_price_action_features
    df = add_price_action_features(df)

    # -----------------------------------------------------------------------
    # 3. Evaluar Tiers
    # -----------------------------------------------------------------------
    evaluator = TierEvaluator()
    df = evaluator.evaluate_dataframe(df)

    signals_df = df[df["Tier"].notna()].copy()
    logger.info("Señales detectadas por TierEvaluator: %d", len(signals_df))

    if signals_df.empty:
        logger.warning("No se detectaron señales. Revisa los parámetros del TierEvaluator.")
        return

    # -----------------------------------------------------------------------
    # 4. Ejecutar el Backtester sobre las señales
    # -----------------------------------------------------------------------
    backtester = StrictBacktester()
    df_labeled = backtester.run(df_prices=df, signals_df=signals_df)

    if df_labeled.empty:
        logger.error("El backtester no generó resultados.")
        return

    # -----------------------------------------------------------------------
    # 5. Exportar el dataset etiquetado
    # -----------------------------------------------------------------------
    out_path = data_path / f"{ticker}_LABELED_DATASET.csv"
    df_labeled.to_csv(out_path)

    logger.info("Dataset etiquetado guardado en: %s", out_path)
    logger.info("Dimensiones finales: %s", df_labeled.shape)

    # Resumen ejecutivo para la Memoria del TFM
    print("\n" + "="*60)
    print(f"  RESUMEN DEL DATASET ETIQUETADO - {ticker}")
    print("="*60)
    print(f"  Señales etiquetadas                 : {len(df_labeled):>8}")
    print(f"  Ganadoras (Label=1, SL4+TP4)        : {int(df_labeled['Label'].sum()):>8}")
    print(f"  Perdedoras (Label=0, SL4+TP4)       : {int((df_labeled['Label']==0).sum()):>8}")
    print(f"  Win Rate Global (SL4+TP4)           : {df_labeled['Label'].mean()*100:>7.1f}%")

    # Win Rate por Stop Loss (promedio de todos los TPs disponibles)
    print("\n  --- Win Rate por Stop Loss (promedio de TPs) ---")
    sl_names = ["SL1_ATR", "SL2_Fractal", "SL3_SMA50", "SL4_SMA200",
                "SL5_Fib382", "SL6_Fib500", "SL7_Fib618"]
    for sl in sl_names:
        cols = [c for c in df_labeled.columns if c.startswith(f"label_{sl}_")]
        if cols:
            wr = df_labeled[cols].mean().mean() * 100
            print(f"    {sl:<15} : {wr:.1f}%  ({len(cols)} TPs)")

    # Win Rate por Take Profit (promedio de todos los SLs disponibles)
    print("\n  --- Win Rate por Take Profit (promedio de SLs) ---")
    tp_names = ["TP1_2R", "TP2_25R", "TP3_3R", "TP4_Fib100", "TP5_Fib1618", "TP6_Fib2618"]
    for tp in tp_names:
        cols = [c for c in df_labeled.columns if c.endswith(f"_{tp}")]
        if cols:
            wr = df_labeled[cols].mean().mean() * 100
            print(f"    {tp:<15} : {wr:.1f}%  ({len(cols)} SLs)")

    print("="*60)


    return df_labeled


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genera el dataset etiquetado para ML.")
    parser.add_argument("--ticker", type=str, default="V", help="Símbolo del activo (ej: V, SPY).")
    args = parser.parse_args()

    run(ticker=args.ticker)
