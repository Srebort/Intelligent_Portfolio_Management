"""
Module for the Autonomous Agent logic deciding portfolio allocations.
"""

import os
import logging
import joblib
import pandas as pd
from src.environment.risk_manager import RiskManager

logger = logging.getLogger("PortfolioAgent")
logging.basicConfig(level=logging.INFO, format="%(message)s")

class PortfolioAgent:
    """
    Autonomous Agent that decides portfolio weights, rebalancing, and risk control.
    """
    
    def __init__(self, model_path: str = "models/best_model.pkl", scaler_path: str = "models/scaler.pkl", prob_threshold: float = 0.75):
        """
        Initializes the Portfolio Agent by loading the ML models.
        
        Args:
            model_path (str): Path to the trained ML model.
            scaler_path (str): Path to the trained scaler.
            prob_threshold (float): Minimum probability to accept a trade.
        """
        self.prob_threshold = prob_threshold
        self.risk_manager = RiskManager()
        
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            logger.info("Modelo y scaler cargados correctamente en el Agente.")
        except Exception as e:
            logger.error(f"Error cargando los modelos en {model_path}: {e}")
            self.model = None
            self.scaler = None

    def calculate_weights(self, predictions: dict, current_portfolio: dict, risk_metrics: dict) -> dict:
        """
        Decides capital allocation for each asset based on model predictions and risk metrics.
        (Mantenido por compatibilidad histórica)
        """
        pass

    def process_signals(self, signals_df: pd.DataFrame, features_cols: list, current_equity: float) -> list:
        """
        Process a batch of signals for a given day.
        
        1. Escala las features de la señal.
        2. Obtiene la probabilidad de éxito de XGBoost/RF.
        3. Filtra las señales con probabilidad >= prob_threshold.
        4. Si es aprobada, consulta al RiskManager el Position Sizing basado en el equity actual.
        
        Args:
            signals_df (pd.DataFrame): DataFrame con las señales técnicas generadas por el TierEvaluator.
            features_cols (list): Lista de columnas que el modelo necesita para predecir.
            current_equity (float): Capital total actual de la cartera para aplicar interés compuesto.
            
        Returns:
            list: Lista de diccionarios con las órdenes de compra a enviar al Portfolio.
        """
        if signals_df.empty or self.model is None or self.scaler is None:
            return []
            
        # Actualizamos el capital del RiskManager al equity real actual
        self.risk_manager.capital = current_equity
        approved_orders = []
        
        for index, row in signals_df.iterrows():
            # Extraer features para el modelo
            try:
                features = row[features_cols].to_frame().T
                features_scaled = self.scaler.transform(features)
                
                # Obtener probabilidad de la clase 1 (Take Profit)
                prob = self.model.predict_proba(features_scaled)[0][1]
            except Exception as e:
                logger.warning(f"Error al predecir señal: {e}")
                continue
                
            ticker = row.get('ticker', f"Signal_{index}")
            
            # Filtro probabilístico
            if prob >= self.prob_threshold:
                logger.info(f" Señal APROBADA para {ticker} con probabilidad {prob*100:.1f}%")
                
                # Pedir al RiskManager el tamaño de la posición
                trade_params = self.risk_manager.calculate_trade_parameters(row)
                
                if trade_params and trade_params.get("num_acciones", 0) > 0:
                    order = {
                        "ticker": ticker,
                        "action": "BUY",
                        "quantity": trade_params["num_acciones"],
                        "price": trade_params["precio_entrada"],
                        "stop_loss": trade_params["stop_loss"],
                        "probability": prob,
                        "tier": trade_params["tier"]
                    }
                    approved_orders.append(order)
                else:
                    logger.info(f"Señal para {ticker} descartada por el RiskManager (distancia al SL incorrecta o falta de liquidez matemática).")
            else:
                logger.info(f"Señal RECHAZADA para {ticker}. Probabilidad: {prob*100:.1f}% < {self.prob_threshold*100}%")
                
        # Ranking de señales: ordenar de mayor a menor probabilidad para ejecutar primero las mejores
        approved_orders.sort(key=lambda x: x["probability"], reverse=True)
        
        return approved_orders
