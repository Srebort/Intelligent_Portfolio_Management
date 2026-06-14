"""
Script de validación para el Agente (Tier Evaluator y Risk Manager).
Carga el histórico procesado, evalúa los Tiers y muestra el sizing de las señales.
"""
import pandas as pd
from pathlib import Path

from src.models.tier_evaluator import TierEvaluator
from src.environment.risk_manager import RiskManager

def run_test():
    print("=== SMOKE-TEST: Agente de Trading (Tiers & Risk Management) ===")
    
    ruta_datos = Path("data/raw/V_4Hour.csv")
    if not ruta_datos.exists():
        print("ERROR: No se encuentra data/raw/V_4Hour.csv")
        return
        
    print("\n1. Construyendo dataset Multi-Timeframe (MTF) y calculando patrones...")
    
    from src.data.mtf_builder import MTFBuilder
    from src.features.patterns import add_price_action_features as add_pa_features
    
    # Construye el dataset MTF propagando 1D y 1W hacia las velas de 4H
    builder = MTFBuilder(ticker="V", data_dir="data/raw")
    df = builder.build(guardar_csv=False)
    
    if df.empty:
        print("ERROR: Fallo al construir el dataset MTF. Revisa que existan los CSVs de 4H, 1D y 1W.")
        return
        
    df = add_pa_features(df)
    
    print(f"Dataset listo: {df.shape}")
    
    print("\n2. Evaluando Tiers...")
    evaluator = TierEvaluator(fib_tolerance=2.0, max_sma_dist=10.0)
    df_eval = evaluator.evaluate_dataframe(df)
    
    # Filtrar solo donde hay señal
    df_signals = df_eval[df_eval['Tier'].notna()]
    print(f"Se encontraron {len(df_signals)} oportunidades de Trading en el histórico.")
    
    if len(df_signals) > 0:
        print("\n3. Calculando Position Sizing para las últimas 5 oportunidades...")
        rm = RiskManager(capital_inicial=100000.0)
        
        for idx, row in df_signals.tail(5).iterrows():
            params = rm.calculate_trade_parameters(row)
            if not params:
                continue
                
            print("-" * 50)
            print(f"Fecha   : {idx}")
            print(f"Tier    : {params['tier']} (Riesgo max: {rm.tier_risk_allocation[params['tier']]*100}%)")
            print(f"Entrada : ${params['precio_entrada']}")
            print(f"StopLoss: ${params['stop_loss']} (Riesgo por acción: ${params['riesgo_por_accion']})")
            print(f"Riesgo total permitido: ${params['riesgo_monetario_max']}")
            print(f"Acciones a comprar    : {params['num_acciones']} acciones")
            print(f"Capital requerido     : ${params['capital_requerido']} ({params['porcentaje_cartera_usado']}% de la cartera)")
            
    else:
        print("No se generaron señales con los parámetros actuales.")

if __name__ == "__main__":
    run_test()
