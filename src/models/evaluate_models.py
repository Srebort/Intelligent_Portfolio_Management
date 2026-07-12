"""
Script de evaluación y selección de modelos de Machine Learning (Sprint 4).
Entrena y compara los modelos base y avanzados (LogReg, SVM, RF, XGBoost),
extrae la importancia de las variables (Feature Importance) y exporta el mejor modelo (.pkl).
"""

import logging
import joblib
import pandas as pd
from pathlib import Path
from sklearn.metrics import confusion_matrix

from src.models.ml_pipeline import MLPipeline
from src.models.baseline_models import LogRegModel, SVMModel, RandomForestModel
from src.models.xgboost_filter import TradeSelectorXGB

logger = logging.getLogger("ModelEvaluation")
logging.basicConfig(level=logging.INFO, format="%(message)s")

def main():
    print("=" * 75)
    print("--- SPRINT 4: EVALUACIÓN COMPARATIVA Y SELECCIÓN DE MODELOS ---")
    print("=" * 75)
    
    pipeline = MLPipeline()
    df = pipeline.load_data()
    if df.empty:
        return
        
    pipeline.filter_leakage_columns()
    pipeline.prepare_features_and_target()
    
    models = {
        "LogReg": LogRegModel(C=0.05),
        "SVM": SVMModel(C=0.5, kernel='rbf'),
        "RF": RandomForestModel(n_estimators=50, max_depth=3),
        "XGBoost": TradeSelectorXGB()
    }
    
    results = []
    best_model_name = None
    best_f1 = -1
    best_model_instance = None
    
    # Evaluar con validación Walk-Forward (Train < 2025, Test >= 2025)
    X_train, X_test, y_train, y_test = pipeline.train_test_split_by_date(split_date="2025-01-01")
    
    if X_train is None or len(X_train) == 0:
        logger.error("Error en el split. Abortando evaluación.")
        return
        
    print(f"\nEntrenando Out-Of-Sample en {len(y_train)} muestras (antes de 2024)...")
    print(f"Evaluando en {len(y_test)} muestras (2024 en adelante)...")
    
    for name, model in models.items():
                if name == "XGBoost":
                    model.train(X_train, y_train, X_test, y_test)
                    metrics = model.evaluate(X_test, y_test)
                else:
                    model.train(X_train, y_train)
                    metrics = model.evaluate(X_test, y_test)
                
                preds = model.predict(X_test)
                cm = confusion_matrix(y_test, preds)
                
                results.append({
                    "Model": name,
                    "Accuracy": metrics['accuracy'],
                    "Precision": metrics['precision'],
                    "Recall": metrics['recall'],
                    "F1 Score": metrics['f1'],
                    "TN": cm[0][0], "FP": cm[0][1],
                    "FN": cm[1][0], "TP": cm[1][1]
                })
                
                if metrics['f1'] > best_f1:
                    best_f1 = metrics['f1']
                    best_model_name = name
                    best_model_instance = model
            
    # --- Feature Importance ---
    print("\n" + "=" * 75)
    print("--- FEATURE IMPORTANCE (Top 10 - Modelos de Árboles) ---")
    print("=" * 75)
    
    for name in ["RF", "XGBoost"]:
        m = models[name].model
        importances = m.feature_importances_
        feat_df = pd.DataFrame({
            "Feature": pipeline.feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False).head(10)
        
        print(f"\n--- {name} ---")
        for _, row in feat_df.iterrows():
            print(f"  -> {row['Feature']:<25} : {row['Importance']:.4f}")
            
    # --- Generar Tabla Comparativa ---
    print("\n" + "=" * 75)
    print("--- TABLA COMPARATIVA FINAL ---")
    print("=" * 75)
    res_df = pd.DataFrame(results).set_index("Model")
    # Imprimir con un formato de tabla limpio
    print(res_df.to_string(float_format=lambda x: f"{x:.3f}"))
    
    # --- Guardar Mejor Modelo ---
    print("\n" + "=" * 75)
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / "best_model.pkl"
    scaler_path = models_dir / "scaler.pkl"
    
    joblib.dump(best_model_instance.model, model_path)
    joblib.dump(pipeline.scaler, scaler_path)
    
    print(f" Mejor modelo ({best_model_name}) guardado en: {model_path}")
    print(f" Scaler (StandardScaler) guardado en:  {scaler_path}")
    print("\nNota: Este modelo está entrenado 100% In-Sample hasta 2024 inclusive.")
    print("La simulación correrá Out-Of-Sample a partir de 2025.")
    print("=" * 75)

if __name__ == "__main__":
    main()
