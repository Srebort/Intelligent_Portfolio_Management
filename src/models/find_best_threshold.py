import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import confusion_matrix
import joblib

# Modelos
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from ml_pipeline import MLPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M",
)
logger = logging.getLogger("ThresholdTuning")

def find_best_threshold(data_path: str = "../../data/processed/MULTI_LABELED_DATASET.csv"):
    # Nos aseguramos de estar en la raíz para las rutas relativas
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    os.chdir(root_dir)
    data_path = "data/processed/MULTI_LABELED_DATASET.csv"
    
    from src.models.ml_pipeline import MLPipeline
    pipeline = MLPipeline(data_path)
    pipeline.load_data()
    pipeline.prepare_features_and_target()
    X = pipeline.df[pipeline.feature_names] # Use df directly so mask slicing works
    y = pipeline.target
    df_clean = pipeline.df
    
    # Split: aislar 2025 para NO contaminar (Data Leakage)
    split_date = pd.to_datetime("2025-01-01", utc=True)
    mask_train = df_clean["fecha_entrada"] < split_date
    
    X_train = X[mask_train]
    y_train = y[mask_train]
    df_train = df_clean[mask_train].copy()
    
    # Escalado
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Definir modelos con sus hiperparámetros óptimos
    models = {
        "LogReg": LogisticRegression(C=1.0, random_state=42, max_iter=1000),
        "SVM": SVC(kernel='rbf', C=5.0, gamma='auto', probability=True, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=200, min_samples_split=2, random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=1.0,
            colsample_bytree=0.6,
            reg_alpha=1.0,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1
        )
    }

    # Time Series Split (para validar el pasado como si fuera futuro)
    tscv = TimeSeriesSplit(n_splits=5)
    
    thresholds = np.arange(0.05, 1.00, 0.05)
    results = {name: [] for name in models.keys()}

    logger.info("Iniciando optimización de umbrales con validación cruzada temporal...")
    
    for name, model in models.items():
        logger.info(f"Evaluando umbrales para {name}...")
        
        # Guardaremos todas las predicciones y etiquetas reales de las 5 validaciones
        y_true_all = []
        y_prob_all = []
        
        for train_idx, val_idx in tscv.split(X_train_scaled):
            X_t, X_v = X_train_scaled[train_idx], X_train_scaled[val_idx]
            y_t, y_v = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            model.fit(X_t, y_t)
            probs = model.predict_proba(X_v)[:, 1]
            
            y_true_all.extend(y_v.values)
            y_prob_all.extend(probs)
            
        y_true_all = np.array(y_true_all)
        y_prob_all = np.array(y_prob_all)
        
        # Probar cada umbral
        for th in thresholds:
            y_pred_th = (y_prob_all >= th).astype(int)
            
            # Calcular matriz de confusión
            tn, fp, fn, tp = confusion_matrix(y_true_all, y_pred_th).ravel()
            
            # Utilidad Financiera Teórica (Expected Value)
            # TP = Gana +3R
            # FP = Pierde -1R
            # TN = Ahorra -1R (es 0R, no perdemos nada)
            # FN = Pierde oportunidad de ganar (es 0R)
            expected_value_r = (tp * 3) - (fp * 1)
            
            results[name].append(expected_value_r)
            
    # Graficar resultados
    plt.figure(figsize=(12, 6))
    for name, ev_list in results.items():
        plt.plot(thresholds, ev_list, label=name, marker='o')
        
        # Encontrar el umbral óptimo
        best_idx = np.argmax(ev_list)
        best_th = thresholds[best_idx]
        best_ev = ev_list[best_idx]
        logger.info(f"--- {name} ---")
        logger.info(f"Mejor Umbral: {best_th:.2f}")
        logger.info(f"Beneficio Neto Máximo (R): +{best_ev}R")
        
    plt.axhline(0, color='black', linestyle='--', alpha=0.5)
    plt.title('Calibración del Umbral de Probabilidad (Cross-Validation pre-2025)')
    plt.xlabel('Probability Threshold')
    plt.ylabel('Beneficio Neto Teórico (R-Múltiplos)')
    plt.legend()
    plt.grid(alpha=0.3)
    
    os.makedirs('results/figures', exist_ok=True)
    plt.savefig('results/figures/threshold_calibration.png', dpi=300, bbox_inches='tight')
    logger.info("Gráfico guardado en results/figures/threshold_calibration.png")
    
if __name__ == "__main__":
    find_best_threshold()
