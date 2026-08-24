"""
Módulo para el modelo predictivo avanzado utilizando XGBoost.
Implementa un clasificador fuertemente regularizado para evitar el sobreajuste
en conjuntos de datos financieros.
"""

import logging
from typing import Dict, Any, Tuple
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, brier_score_loss

logger = logging.getLogger("TradeSelectorXGB")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")

class TradeSelectorXGB:
    def __init__(self, **kwargs):
        """
        Clasificador avanzado basado en XGBoost.
        Configurado por defecto con fuerte regularización L1/L2 (alpha/lambda)
        y profundidad máxima baja para evitar sobreajuste en datasets financieros pequeños.
        """
        # Hiperparámetros por defecto optimizados para evitar overfitting
        default_params = {
            'n_estimators': 200,         # Número de árboles óptimo
            'max_depth': 5,              # Árboles con profundidad óptima
            'learning_rate': 0.1,        # Aprendizaje óptimo
            'subsample': 1.0,            # Usa el 100% de las filas por árbol
            'colsample_bytree': 0.6,     # Usa el 60% de las columnas por árbol
            'reg_alpha': 1.0,            # Regularización L1 óptima
            'reg_lambda': 1.0,           # Regularización L2 óptima
            'scale_pos_weight': 1.0,     # Ajustar si hay desbalanceo severo
            'random_state': 42,
            'eval_metric': 'logloss'
        }
        
        # Sobrescribir con los kwargs proporcionados
        self.params = {**default_params, **kwargs}
        self.model = xgb.XGBClassifier(**self.params)
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray = None, y_val: np.ndarray = None):
        """
        Entrena el modelo XGBoost.
        """
        logger.info(f"Entrenando TradeSelectorXGB (N_train={len(y_train)})...")
        
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
            
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )
        
    def predict(self, X_test: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predice la clase binaria usando un umbral personalizado.
        """
        probs = self.predict_proba(X_test)[:, 1]
        return (probs >= threshold).astype(int)
        
    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """
        Devuelve las probabilidades [P(Loss), P(Win)]
        """
        return self.model.predict_proba(X_test)
        
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
        """
        Evalúa el rendimiento del modelo con un umbral dado.
        """
        preds = self.predict(X_test, threshold)
        probs = self.predict_proba(X_test)
        
        cm = confusion_matrix(y_test, preds)
        tn, fp, fn, tp = cm.ravel() if len(cm.ravel()) == 4 else (0,0,0,0)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0),
            "specificity": specificity,
            "confusion_matrix": cm.tolist()
        }
        
        if probs is not None and probs.shape[1] == 2:
            metrics["roc_auc"] = roc_auc_score(y_test, probs[:, 1])
            metrics["brier_score"] = brier_score_loss(y_test, probs[:, 1])
        else:
            metrics["roc_auc"] = 0.5
            metrics["brier_score"] = 0.0
            
        return metrics

if __name__ == "__main__":
    from src.models.ml_pipeline import MLPipeline
    
    print("=" * 60)
    print("INICIANDO EVALUACIÓN DE TRADE SELECTOR XGBOOST")
    print("=" * 60)
    
    pipeline = MLPipeline()
    df = pipeline.load_data()
    
    if not df.empty:
        pipeline.filter_leakage_columns()
        pipeline.prepare_features_and_target()
        
        # Iterar sobre las particiones cronológicas
        for fold, (X_train, X_test, y_train, y_test) in enumerate(pipeline.split_and_scale(n_splits=3)):
            if fold == 2:  # Evaluar en el último fold
                logger.info("--- EVALUANDO EN ÚLTIMO FOLD (Train Size: %d, Test Size: %d) ---", len(y_train), len(y_test))
                
                # Inicializar TradeSelectorXGB
                xgb_model = TradeSelectorXGB()
                
                # Entrenar pasándole el test como validación
                xgb_model.train(X_train, y_train, X_test, y_test)
                
                # Evaluar
                metrics = xgb_model.evaluate(X_test, y_test)
                
                print(f"\n--- TradeSelectorXGB ---")
                print(f"  -> Accuracy:  {metrics['accuracy']:.3f}")
                print(f"  -> Precision: {metrics['precision']:.3f}")
                print(f"  -> Recall:    {metrics['recall']:.3f}")
                print(f"  -> F1 Score:  {metrics['f1']:.3f}")
                
                cm = metrics['confusion_matrix']
                print(f"  -> Matriz de Confusión:")
                print(f"       [TN: {cm[0][0]:2d} | FP: {cm[0][1]:2d}]  (0=Perdedoras)")
                print(f"       [FN: {cm[1][0]:2d} | TP: {cm[1][1]:2d}]  (1=Ganadoras)")
                
                # Probabilidades de los primeros 5 ejemplos
                probs = xgb_model.predict_proba(X_test[:5])
                print("\n  -> Probabilidades (primeras 5 operaciones del futuro):")
                for i, p in enumerate(probs):
                    real_label = "Ganadora" if y_test.iloc[i] == 1 else "Perdedora"
                    print(f"       Op {i+1}: Loss {p[0]*100:5.1f}% | Win {p[1]*100:5.1f}%  (Real: {real_label})")
                    
        print("\n" + "=" * 60)
