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

    # Win Rate Desglosado por Tier (usando SL4_SMA200 + TP4_Fib100)
    print("\n  --- Win Rate por Tier (Label principal: SL4+TP4) ---")
    if 'Tier' in df_labeled.columns:
        tier_stats = df_labeled.groupby('Tier')['Label'].agg(['count', 'mean'])
        for tier, row in tier_stats.iterrows():
            print(f"    Tier {tier:<10} : {row['mean']*100:.1f}%  ({int(row['count'])} señales)")

    print("\n  --- Ejemplos Detallados para Análisis Visual en Gráfico (Año >= 2025) ---")
    if 'Tier' in df_labeled.columns:
        # Añadir temporalmente columna de año
        df_labeled['Year'] = df_labeled.index.year
        
        # Filtrar solo a partir de 2025 y agrupar por Año y Tier
        df_recent = df_labeled[df_labeled['Year'] >= 2025]
        grouped = df_recent.groupby(['Year', 'Tier'])
        
        for name, group in grouped:
            year, tier = name
            print(f"\n    [ Año {year} | Tier {tier} ]")
            # Tomar 2 ejemplos aleatorios (o los primeros si hay menos)
            samples = group.head(2)
            for idx, row in samples.iterrows():
                # Obtener qué nivel se tocó primero (Label = 1 si TP, 0 si SL)
                is_win = row.get('Label', 0) == 1
                sl_vela = row.get('SL4_SMA200_vela')
                tp_vela = row.get('TP4_Fib100_vela')
                
                # Determinar vela de salida (la menor si ambas no son nulas)
                if pd.notna(sl_vela) and pd.notna(tp_vela):
                    exit_offset = int(min(sl_vela, tp_vela))
                elif pd.notna(sl_vela):
                    exit_offset = int(sl_vela)
                elif pd.notna(tp_vela):
                    exit_offset = int(tp_vela)
                else:
                    exit_offset = 0

                # Calcular la fecha exacta de salida basándose en el índice original de precios
                entry_pos = df.index.get_loc(idx)
                exit_pos = entry_pos + exit_offset
                if exit_pos < len(df):
                    exit_date = df.index[exit_pos].strftime('%Y-%m-%d %H:%M')
                else:
                    exit_date = "No finalizó"

                resultado = "GANADORA " if is_win else "PERDEDORA"
                
                # Obtener el precio de entrada (cierre de esa vela) y de salida
                entry_price = df.loc[idx, 'close']
                atr = df.loc[idx, 'ATR_14'] if 'ATR_14' in df.columns else 0
                sma50 = df.loc[idx, 'SMA_50'] if 'SMA_50' in df.columns else 0
                sma200 = df.loc[idx, 'SMA_200'] if 'SMA_200' in df.columns else 0
                
                # Buscar el precio del último fractal de soporte propagado hasta esta vela
                if 'is_support_fractal' in df.columns:
                    fractal_series = df['low'].where(df['is_support_fractal'] == 1).shift(1).ffill()
                    fractal_val = fractal_series.loc[idx]
                else:
                    fractal_val = float('nan')

                if is_win:
                    salida_tipo = "TP (Fib100)"
                    exit_price = row.get('TP4_Fib100_precio', 0)
                else:
                    salida_tipo = "SL (SMA200)"
                    exit_price = row.get('SL4_SMA200_precio', 0)
                
                print(f"      -> ENTRADA: {idx.strftime('%Y-%m-%d %H:%M')} (Precio: ${entry_price:.2f})")
                print(f"         [Valores Base de los Indicadores en la Vela de Entrada]")
                print(f"           - ATR(14)      : ${atr:.2f}")
                print(f"           - SMA(50)      : ${sma50:.2f}")
                print(f"           - SMA(200)     : ${sma200:.2f}")
                print(f"           - Suelo Fractal: ${fractal_val:.2f}")
                print(f"         [Niveles Calculados de Stop Loss]")
                print(f"           - SL1 (ATR)    : ${row.get('SL1_ATR_precio', 0):.2f}")
                print(f"           - SL2 (Fractal): ${row.get('SL2_Fractal_precio', 0):.2f}")
                print(f"           - SL3 (SMA50)  : ${row.get('SL3_SMA50_precio', 0):.2f}")
                print(f"           - SL4 (SMA200) : ${row.get('SL4_SMA200_precio', 0):.2f}")
                print(f"           - SL5 (Fib382) : ${row.get('SL5_Fib382_precio', 0):.2f}")
                print(f"           - SL6 (Fib500) : ${row.get('SL6_Fib500_precio', 0):.2f}")
                print(f"           - SL7 (Fib618) : ${row.get('SL7_Fib618_precio', 0):.2f}")
                print(f"         [Niveles Calculados de Take Profit]")
                print(f"           - TP1 (2R)     : ${row.get('TP1_2R_precio', 0):.2f}")
                print(f"           - TP2 (2.5R)   : ${row.get('TP2_25R_precio', 0):.2f}")
                print(f"           - TP3 (3R)     : ${row.get('TP3_3R_precio', 0):.2f}")
                print(f"           - TP4 (Fib100) : ${row.get('TP4_Fib100_precio', 0):.2f}")
                print(f"           - TP5 (Fib161) : ${row.get('TP5_Fib1618_precio', 0):.2f}")
                print(f"           - TP6 (Fib261) : ${row.get('TP6_Fib2618_precio', 0):.2f}")
                print(f"         SALIDA REALIZADA:  {exit_date} (Precio: ${exit_price:.2f}) | {resultado} por {salida_tipo}")
                print("         " + "-"*60)
                
        # Limpiar la columna temporal
        df_labeled.drop(columns=['Year'], inplace=True)

    print("="*60)


    return df_labeled


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genera el dataset etiquetado para ML.")
    parser.add_argument("--ticker", type=str, default="V", help="Símbolo del activo (ej: V, SPY).")
    args = parser.parse_args()

    run(ticker=args.ticker)
