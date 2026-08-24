"""
Módulo con la implementación de los modelos predictivos base.
Incluye Regresión Logística, Support Vector Machine (SVM) y Random Forest.
Proporciona una clase base estandarizada para facilitar el entrenamiento y evaluación.
"""

import logging
from typing import Dict, Any
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, brier_score_loss, confusion_matrix

logger = logging.getLogger("BaselineModels")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")

class BaseModel:
    def __init__(self, name: str, model_instance):
        self.name = name
        self.model = model_instance
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        logger.info(f"Entrenando modelo: {self.name} (N_train={len(y_train)})")
        self.model.fit(X_train, y_train)
        
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return self.model.predict(X_test)
        
    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X_test)
        return None
        
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        preds = self.predict(X_test)
        probs = self.predict_proba(X_test)
        
        cm = confusion_matrix(y_test, preds)
        tn, fp, fn, tp = cm.ravel() if len(cm.ravel()) == 4 else (0,0,0,0)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0),
            "specificity": specificity
        }
        
        if probs is not None and probs.shape[1] == 2:
            metrics["roc_auc"] = roc_auc_score(y_test, probs[:, 1])
            metrics["brier_score"] = brier_score_loss(y_test, probs[:, 1])
        else:
            metrics["roc_auc"] = 0.5
            metrics["brier_score"] = 0.0
            
        return metrics

class LogRegModel(BaseModel):
    def __init__(self, C=1.0, class_weight='balanced', penalty='l2', **kwargs):
        """
        Regresión Logística con parámetros óptimos de GridSearchCV.
        """
        model = LogisticRegression(C=C, penalty=penalty, class_weight=class_weight, random_state=42, max_iter=2000, **kwargs)
        super().__init__("Logistic Regression", model)

class SVMModel(BaseModel):
    def __init__(self, C=5.0, kernel='rbf', probability=True, class_weight='balanced', gamma='auto', **kwargs):
        """
        Support Vector Machine con parámetros óptimos de GridSearchCV.
        """
        model = SVC(C=C, kernel=kernel, probability=probability, class_weight=class_weight, gamma=gamma, random_state=42, **kwargs)
        super().__init__("Support Vector Machine", model)

class RandomForestModel(BaseModel):
    def __init__(self, n_estimators=200, max_depth=None, min_samples_split=2, min_samples_leaf=1, class_weight='balanced', **kwargs):
        """
        Random Forest con parámetros óptimos de GridSearchCV.
        """
        model = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            random_state=42, 
            **kwargs
        )
        super().__init__("Random Forest", model)

if __name__ == "__main__":
    from src.models.ml_pipeline import MLPipeline
    
    print("=" * 60)
    print("INICIANDO EVALUACIÓN DE MODELOS BASE (SPRINT 4)")
    print("=" * 60)
    
    # 1. Cargar y preparar datos
    pipeline = MLPipeline()
    df = pipeline.load_data()
    
    if not df.empty:
        pipeline.filter_leakage_columns()
        pipeline.prepare_features_and_target()
        
        # 2. Inicializar modelos
        models = [
            LogRegModel(C=0.05),
            SVMModel(C=0.5, kernel='rbf'),
            RandomForestModel(n_estimators=50, max_depth=3)
        ]
        
        # 3. Entrenar y evaluar
        # Usamos el último fold (partición 3/3) porque nos da la mayor cantidad
        # de datos de entrenamiento (100 filas) y testea en los más recientes (33 filas).
        for fold, (X_train, X_test, y_train, y_test) in enumerate(pipeline.split_and_scale(n_splits=3)):
            if fold == 2:  # Último fold cronológico
                logger.info("--- EVALUANDO EN ÚLTIMO FOLD (Train Size: %d, Test Size: %d) ---", len(y_train), len(y_test))
                
                for model in models:
                    model.train(X_train, y_train)
                    metrics = model.evaluate(X_test, y_test)
                    
                    print(f"\n--- {model.name} ---")
                    print(f"  -> Accuracy:  {metrics['accuracy']:.3f}")
                    print(f"  -> Precision: {metrics['precision']:.3f}")
                    print(f"  -> Recall:    {metrics['recall']:.3f}")
                    print(f"  -> F1 Score:  {metrics['f1']:.3f}")
                    
                    # Mostrar algunas probabilidades reales del test set
                    probs = model.predict_proba(X_test[:5])
                    if probs is not None:
                        print("  -> Probabilidades (primeras 5 operaciones del futuro):")
                        for i, p in enumerate(probs):
                            real_label = "Ganadora" if y_test.iloc[i] == 1 else "Perdedora"
                            print(f"       Op {i+1}: Loss {p[0]*100:5.1f}% | Win {p[1]*100:5.1f}%  (Real: {real_label})")
                
                print("\n" + "=" * 60)
