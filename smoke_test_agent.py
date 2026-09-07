"""
Smoke Test — Tier C (Breakout) Batch Processor

Ejecuta el smoke test de Tier C para todas las acciones descargadas en data/raw/,
utilizando SL1_ATR y TP5_Fib1618, y muestra un resumen agregado de rentabilidad.
"""

import pandas as pd
import numpy as np
import glob
import os
from pathlib import Path

from src.data.mtf_builder import MTFBuilder
from src.features.patterns import add_price_action_features as add_pa_features
from src.models.tier_evaluator import TierEvaluator
from src.environment.risk_manager import RiskManager
from src.environment.backtester import StrictBacktester

import warnings
warnings.filterwarnings('ignore') # Ocultar warnings de pandas para output limpio

def run_tier_c_test(ticker: str, verbose: bool = False) -> dict:
    
    # ------------------------------------------------------------------
    # 1. Construir dataset MTF con indicadores técnicos y patrones
    # ------------------------------------------------------------------
    builder = MTFBuilder(ticker=ticker, data_dir="data/raw")
    # Silenciar logs del builder si no es verbose
    import logging
    logger = logging.getLogger()
    old_level = logger.level
    if not verbose:
        logger.setLevel(logging.ERROR)
        
    df = builder.build(guardar_csv=False)
    
    if not verbose:
        logger.setLevel(old_level)
        
    if df is None or df.empty:
        return {"ticker": ticker, "total": 0, "wins": 0, "net_pnl": 0}
        
    df = add_pa_features(df)

    # ------------------------------------------------------------------
    # 2. Evaluar Tiers
    # ------------------------------------------------------------------
    evaluator = TierEvaluator(fib_tolerance=1.5, max_sma_dist=5.0)
    df_eval = evaluator.evaluate_dataframe(df)

    # Extraer TODAS las señales válidas (Tier A, B y C)
    df_c = df_eval[df_eval["Tier"].isin(["A", "B", "C"])].copy()

    # ------------------------------------------------------------------
    # Imprimir todos los fractales de resistencia encontrados (verbose)
    # ------------------------------------------------------------------
    if verbose:
        fractales = df_eval[df_eval["is_resistance_fractal"] == 1].copy()
        print(f"\n{'─'*60}")
        print(f"  FRACTALES DE RESISTENCIA DETECTADOS ({len(fractales)} en total)")
        print(f"{'─'*60}")
        # Pre-calcular la edad de cada fractal (velas desde su formación hasta el fin del dataset)
        bars = pd.Series(range(len(df_eval)), index=df_eval.index)
        for fecha_fractal, row_f in fractales.iterrows():
            precio_fractal = row_f["high"]
            bar_idx = df_eval.index.get_loc(fecha_fractal)
            velas_desde_formacion = len(df_eval) - bar_idx
            print(f"  📌 {fecha_fractal.date()} | Resistencia: ${precio_fractal:.2f} | ~{velas_desde_formacion} velas desde formación")
        print(f"{'─'*60}\n")

    
    if df_c.empty:
        return {"ticker": ticker, "total": 0, "wins": 0, "net_pnl": 0}

    # ------------------------------------------------------------------
    # 3. Backtester para etiquetar cada señal (SL1_ATR + TP5_Fib1618)
    # ------------------------------------------------------------------
    backtester = StrictBacktester()
    df_labeled = backtester.run(df_prices=df_eval, signals_df=df_c)

    if df_labeled.empty:
        return {"ticker": ticker, "total": 0, "wins": 0, "net_pnl": 0}

    label_col = "label_SL1_ATR_TP3_3R"
    if label_col not in df_labeled.columns:
        label_col = "Label"

    wins  = int(df_labeled[label_col].sum())
    total = len(df_labeled)

    # ------------------------------------------------------------------
    # 4. Calcular P&L para todas las señales
    # ------------------------------------------------------------------
    rm = RiskManager(capital_inicial=100_000.0)
    total_net_pnl = 0.0

    # Pre-calcular el nivel de la última resistencia
    last_resistance = df_eval["high"].where(
        df_eval.get("is_resistance_fractal", pd.Series(0, index=df_eval.index)) == 1
    ).ffill()
    bars = pd.Series(np.arange(len(df_eval)), index=df_eval.index)
    last_res_bar = bars.where(
        df_eval.get("is_resistance_fractal", pd.Series(0, index=df_eval.index)) == 1
    ).ffill()
    fractal_age = bars - last_res_bar

    # Pre-calcular fractal de soporte para el impulso
    fractal_support_price = (
        df_eval["low"]
        .where(df_eval.get("is_support_fractal", pd.Series(0, index=df_eval.index)) == 1)
        .shift(1).ffill()
    )

    trades_list = []

    for fecha, row in df_labeled.iterrows():
        precio   = row["close"]
        import math
        resultado = row.get(label_col, None)
        
        sl1_dataset = row.get("SL1_ATR_precio", None)
        if sl1_dataset is not None and math.isnan(sl1_dataset): sl1_dataset = None
        tp3_dataset = row.get("TP3_3R_SL1_ATR_precio", None)
        if tp3_dataset is not None and math.isnan(tp3_dataset): tp3_dataset = None
        tp5_dataset = row.get("TP5_Fib1618_precio", None)

        riesgo_por_accion = precio - sl1_dataset if sl1_dataset else np.nan
        riesgo_max = 100_000 * 0.010  # Tier C = 1.0% del capital (1000$)
        num_acciones = int(riesgo_max / riesgo_por_accion) if riesgo_por_accion and riesgo_por_accion > 0 else 0

        if resultado == 1:
            pnl_bruto = (tp3_dataset - precio) * num_acciones if tp3_dataset is not None else 0
        elif resultado == 0:
            if row.get("BE_Hit", 0) == 1:
                pnl_bruto = 0  # Break even! El stop saltó en precio de entrada
            else:
                pnl_bruto = (sl1_dataset - precio) * num_acciones if sl1_dataset is not None else 0
        else:
            pnl_bruto = None

        comision = precio * num_acciones * 0.001 * 2 if num_acciones else 0  # 0.1% entrada + salida
        pnl_neto = pnl_bruto - comision if pnl_bruto is not None else None
        
        if pnl_neto is not None:
            total_net_pnl += pnl_neto
            
            # Aproximar fecha de cierre para el Drawdown global
            vela_cierre = row.get("TP3_3R_SL1_ATR_vela", None) if resultado == 1 else row.get("SL1_ATR_vela", None)
            try:
                idx_entrada = df_eval.index.get_loc(fecha)
                idx_cierre  = int(idx_entrada + vela_cierre) if vela_cierre else None
                fecha_cierre = df_eval.index[idx_cierre] if idx_cierre and idx_cierre < len(df_eval) else fecha
            except Exception:
                fecha_cierre = fecha
            
            # Guardamos el P&L, el resultado (1=win, 0=loss) y el Tier
            is_win = 1 if pnl_neto > 0 else 0
            trades_list.append((fecha_cierre, pnl_neto, is_win, row.get("Tier", "Unknown")))
            
        if verbose:
            emoji_res = "✅ WIN" if resultado == 1 else "❌ LOSS" if resultado == 0 else "❓"
            emoji_pnl = f"+${pnl_neto:,.0f}" if pnl_neto and pnl_neto > 0 else f"-${abs(pnl_neto):,.0f}" if pnl_neto else "?"
            
            # Fecha cierre
            if resultado == 1:
                vela_cierre = row.get("TP5_Fib1618_vela", None)
            else:
                vela_cierre = row.get("SL1_ATR_vela", None)
            try:
                idx_entrada = df_eval.index.get_loc(fecha)
                idx_cierre  = int(idx_entrada + vela_cierre) if vela_cierre and not pd.isna(vela_cierre) else None
                fecha_cierre = df_eval.index[idx_cierre].date() if idx_cierre and idx_cierre < len(df_eval) else "?"
            except Exception:
                fecha_cierre = "?"
                
            dias_abierta = int(vela_cierre / 6) if not pd.isna(vela_cierre) else "?"
            
            # MFE (Máximo beneficio latente antes de cerrar)
            mfe_str = ""
            if vela_cierre and not pd.isna(vela_cierre):
                try:
                    future_slice = df_eval.iloc[idx_entrada + 1 : idx_entrada + int(vela_cierre) + 1]
                    if not future_slice.empty:
                        max_high = future_slice["high"].max()
                        if max_high > precio:
                            mfe_pct = (max_high - precio) / precio * 100
                            max_profit = (max_high - precio) * num_acciones
                            mfe_str = f"   📈 Llegó a ganar: +${max_profit:,.0f} (Máx: ${max_high:.2f} | +{mfe_pct:.2f}%)"
                except Exception:
                    pass
                    
            # Resistencia
            try:
                res_price = last_resistance.iloc[idx_entrada]
                res_age = int(fractal_age.iloc[idx_entrada])
            except:
                res_price = 0.0
                res_age = 0

            print(f"\n📅 Entrada: {fecha.date()} → Cierre: {fecha_cierre} ({dias_abierta} días)")
            print(f"   Precio: ${precio:.2f} | 💥 Breakout Resistencia: ${res_price:.2f} ({res_age} velas) | {emoji_res} | P&L: {emoji_pnl}")
            sl1_str = f"${sl1_dataset:.2f}" if sl1_dataset else "N/A"
            tp5_str = f"${tp5_dataset:.2f}" if tp5_dataset else "N/A"
            print(f"   SL1: {sl1_str} | TP5: {tp5_str}")
            if mfe_str:
                print(mfe_str)

        # Imprimir resumen final para esta acción
    win_rate = (wins / total) * 100 if total > 0 else 0
    pnl_str = f"+${total_net_pnl:,.0f}" if total_net_pnl > 0 else f"-${abs(total_net_pnl):,.0f}"
    if verbose:
        print(f"\n{'='*70}")
        print(f"  RESUMEN FINAL — {ticker}")
        print(f"  Total Señales : {total}")
        print(f"  Wins          : {wins} ✅")
        print(f"  Win Rate      : {win_rate:.1f}%")
        print(f"  P&L Neto      : {pnl_str}")
        print(f"{'='*70}\n")

    return {
        "ticker": ticker,
        "total": total,
        "wins": wins,
        "net_pnl": total_net_pnl,
        "trades": trades_list
    }

def run_all_tickers():
    print("=" * 80)
    print("  SMOKE TEST BATCH — TIER C (BREAKOUT) | SL1_ATR + TP3_3R + BE | RIESGO 1%")
    print("=" * 80)
    print("Buscando acciones descargadas...")
    
    archivos_4h = glob.glob("data/raw/*_4Hour.csv")
    tickers = [os.path.basename(f).replace("_4Hour.csv", "") for f in archivos_4h]
    tickers = sorted(tickers)
    
    print(f"Se encontraron {len(tickers)} acciones. Evaluando...\n")
    
    resultados = []
    
    # Formato de cabecera de tabla
    print("Calculando y ordenando por P&L Neto...")
    
    total_signals = 0
    total_wins = 0
    total_pnl = 0.0
    all_global_trades = []
    
    for ticker in tickers:
        res = run_tier_c_test(ticker, verbose=False)
        if res["total"] > 0:
            resultados.append(res)
            
    # Ordenar por P&L Neto descendente
    resultados.sort(key=lambda x: x["net_pnl"], reverse=True)
    
    print(f"\n{'TICKER':<8} | {'SEÑALES':<8} | {'WINS':<6} | {'WIN RATE':<10} | {'P&L NETO':<12}")
    print("-" * 55)
    
    for res in resultados:
        ticker = res["ticker"]
        win_rate = (res["wins"] / res["total"]) * 100
        pnl_str = f"+${res['net_pnl']:,.0f}" if res['net_pnl'] > 0 else f"-${abs(res['net_pnl']):,.0f}"
        
        print(f"{ticker:<8} | {res['total']:<8} | {res['wins']:<6} | {win_rate:>5.1f}%     | {pnl_str:>10}")
        
        total_signals += res["total"]
        total_wins += res["wins"]
        total_pnl += res["net_pnl"]
        all_global_trades.extend(res["trades"])
            
    print("-" * 55)
    if total_signals > 0:
        avg_win_rate = (total_wins / total_signals) * 100
        total_pnl_str = f"+${total_pnl:,.0f}" if total_pnl > 0 else f"-${abs(total_pnl):,.0f}"
        print(f"{'TOTAL':<8} | {total_signals:<8} | {total_wins:<6} | {avg_win_rate:>5.1f}%     | {total_pnl_str:>10}")
        
        # Calcular Drawdown Global
        all_global_trades.sort(key=lambda x: x[0])  # Ordenar por fecha de cierre
        current_equity = 100_000  # Capital inicial asumido para %
        peak_equity = 100_000
        max_dd_abs = 0.0
        
        # Breakdown por Tiers
        tiers_stats = {}
        # En all_global_trades guardamos ahora tuplas de 4: (fecha, pnl, is_win, tier)
        # O de 2: (fecha, pnl) en caso de que vinieran de corridas antiguas (para compatibilidad si lo llamamos fuera)
        for trade in all_global_trades:
            if len(trade) >= 4:
                tier = trade[3]
                pnl = trade[1]
                is_win = trade[2]
            else:
                tier = "Unknown"
                pnl = trade[1]
                is_win = 1 if pnl > 0 else 0
                
            if tier not in tiers_stats:
                tiers_stats[tier] = {
                    "total": 0, "wins": 0, "pnl": 0.0,
                    "gross_profit": 0.0, "gross_loss": 0.0,
                    "current_streak": 0, "max_losing_streak": 0,
                    "current_equity": 100_000, "peak_equity": 100_000, "max_dd_rel": 0.0
                }
            
            ts = tiers_stats[tier]
            ts["total"] += 1
            ts["wins"] += is_win
            ts["pnl"] += pnl
            
            if pnl > 0:
                ts["gross_profit"] += pnl
                ts["current_streak"] = 0
            else:
                ts["gross_loss"] += abs(pnl)
                ts["current_streak"] += 1
                if ts["current_streak"] > ts["max_losing_streak"]:
                    ts["max_losing_streak"] = ts["current_streak"]
                    
            ts["current_equity"] += pnl
            if ts["current_equity"] > ts["peak_equity"]:
                ts["peak_equity"] = ts["current_equity"]
            
            dd_rel = ((ts["peak_equity"] - ts["current_equity"]) / ts["peak_equity"]) * 100 if ts["peak_equity"] > 0 else 0
            if dd_rel > ts["max_dd_rel"]:
                ts["max_dd_rel"] = dd_rel
            
        print("-" * 55)
        print("  DESGLOSE POR TIERS (Advanced):")
        for t in sorted(tiers_stats.keys()):
            s = tiers_stats[t]
            if s["total"] > 0:
                wr = (s["wins"] / s["total"]) * 100
                pnl_str = f"+${s['pnl']:,.0f}" if s['pnl'] >= 0 else f"-${abs(s['pnl']):,.0f}"
                pf = (s["gross_profit"] / s["gross_loss"]) if s["gross_loss"] > 0 else 999.0
                print(f"    Tier {t}: {s['total']} ops | Win Rate: {wr:.1f}% | P&L: {pnl_str} | Max DD: -{s['max_dd_rel']:.2f}% | Profit Factor: {pf:.2f} | Racha Max: {s['max_losing_streak']}")

        print("-" * 55)
        
        # Calculate max drawdown
        max_dd_rel = 0.0
        current_equity = 100_000
        peak_equity = 100_000
        for trade in all_global_trades:
            current_equity += trade[1]
            if current_equity > peak_equity:
                peak_equity = current_equity
            
            dd_abs = peak_equity - current_equity
            if dd_rel > max_dd_rel: max_dd_rel = dd_rel
            
        max_dd_abs_pct = (max_dd_abs / 100_000) * 100
        # Calculate Sharpe
        import numpy as np
        returns = np.array([trade[1] for trade in all_global_trades])
        tpy = len(returns) / 10.0  # ~10 years
        mean_return = np.mean(returns) if len(returns) > 0 else 0
        std_return = np.std(returns) if len(returns) > 0 else 0
        sharpe = (mean_return / std_return) * np.sqrt(tpy) if std_return > 0 else 0
        
        downside_returns = returns[returns < 0]
        std_downside = np.std(downside_returns) if len(downside_returns) > 0 else 0
        sortino = (mean_return / std_downside) * np.sqrt(tpy) if std_downside > 0 else 0
        
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        current_streak = 0
        max_streak = 0
        for r in returns:
            if r < 0:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            elif r > 0:
                current_streak = 0

        # Breakdown por Tiers
        tiers_stats = {}
        # En all_global_trades guardamos ahora tuplas de 4: (fecha, pnl, is_win, tier)
        # O de 2: (fecha, pnl) en caso de que vinieran de corridas antiguas (para compatibilidad si lo llamamos fuera)
        for trade in all_global_trades:
            if len(trade) >= 4:
                tier = trade[3]
                pnl = trade[1]
                is_win = trade[2]
            else:
                tier = "Unknown"
                pnl = trade[1]
                is_win = 1 if pnl > 0 else 0
                
            if tier not in tiers_stats:
                tiers_stats[tier] = {"total": 0, "wins": 0, "pnl": 0.0}
            tiers_stats[tier]["total"] += 1
            tiers_stats[tier]["wins"] += is_win
            tiers_stats[tier]["pnl"] += pnl
            
        print("-" * 55)
        print("  DESGLOSE POR TIERS:")
        for t in sorted(tiers_stats.keys()):
            s = tiers_stats[t]
            if s["total"] > 0:
                wr = (s["wins"] / s["total"]) * 100
                pnl_str = f"+${s['pnl']:,.0f}" if s['pnl'] >= 0 else f"-${abs(s['pnl']):,.0f}"
                print(f"    Tier {t}: {s['total']} ops | Win Rate: {wr:.1f}% | P&L: {pnl_str}")

        print("-" * 55)
        print(f"  MAX DRAWDOWN (Absoluto) : -{max_dd_abs_pct:.2f}% (-${max_dd_abs:,.0f})")
        print(f"  MAX DRAWDOWN (Dinámico) : -{max_dd_rel:.2f}%")
        print(f"  PROFIT FACTOR           : {profit_factor:.2f}")
        print(f"  SHARPE RATIO (Trades)   : {sharpe:.2f}")
        print(f"  SORTINO RATIO (Trades)  : {sortino:.2f}")
        print(f"  MAX RACHA PÉRDIDAS      : {max_streak} operaciones")
        
        # Breakdown por periodos de 2 años
        periodos = {}
        import pandas as pd
        for trade in all_global_trades:
            fecha, pnl = trade[0], trade[1]
            try:
                year = pd.to_datetime(fecha).year
                if str(year) not in periodos:
                    periodos[str(year)] = 0.0
                periodos[str(year)] += pnl
            except Exception:
                pass
            
        print("-" * 55)
        print("  DESGLOSE P&L POR PERIODOS (AÑO POR AÑO):")
        for periodo in sorted(periodos.keys()):
            pnl_periodo = periodos[periodo]
            pnl_str_per = f"+${pnl_periodo:,.0f}" if pnl_periodo >= 0 else f"-${abs(pnl_periodo):,.0f}"
            print(f"    {periodo} : {pnl_str_per:>10}")
        
    else:
        print("No se generaron señales.")
        
    print("=" * 80)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--lin":
        print("\n======================================================================")
        print("  ANÁLISIS DETALLADO TIER B — LIN (LINDE) | RIESGO 1% | SL1+TP3")
        run_tier_c_test("LIN", verbose=True)
    
    elif len(sys.argv) > 1 and sys.argv[1] == "--vis":
        print("\n======================================================================")
        print("  ANÁLISIS DETALLADO TIER B — V (VISA) | RIESGO 1% | SL1+TP3")
        run_tier_c_test("V", verbose=True)
    else:
        run_all_tickers()