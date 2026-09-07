import os
import sys
import json
import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.models.ml_pipeline import MLPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger("Tuner")

def main():
    logger.info("Iniciando Hyperparameter Tuning...")
    
    # 1. Cargar y preparar datos
    pipeline = MLPipeline("data/processed/MULTI_LABELED_DATASET.csv")
    pipeline.load_data()
    pipeline.prepare_features_and_target()
    
    # Extraer el conjunto de entrenamiento (antes de 2025) para el tuning
    split_date = "2025-01-01"
    split_ts = pd.to_datetime(split_date, utc=True)
    dates_utc = pd.to_datetime(pipeline.entry_dates, utc=True)
    train_mask = dates_utc < split_ts
    
    X_train_raw = pipeline.features[train_mask]
    y_train = pipeline.target.iloc[train_mask]
    
    # Escalar X_train_raw para modelos que lo requieran
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    
    # Validation scheme cronológico
    tscv = TimeSeriesSplit(n_splits=5)
    
    best_params_all = {}
    
    # ---------------------------------------------------------
    # 1. Logistic Regression
    # ---------------------------------------------------------
    logger.info("Optimizando Logistic Regression...")
    param_grid_lr = {
        'C': [0.001, 0.01, 0.1, 1.0, 10.0],
        'penalty': ['l2']
    }
    lr = LogisticRegression(class_weight='balanced', random_state=42, max_iter=2000)
    search_lr = RandomizedSearchCV(lr, param_grid_lr, n_iter=5, cv=tscv, scoring='roc_auc', n_jobs=-1, random_state=42)
    search_lr.fit(X_train_scaled, y_train)
    logger.info(f"LogReg Best ROC AUC: {search_lr.best_score_:.4f}")
    best_params_all["Logistic Regression"] = search_lr.best_params_
    
    # ---------------------------------------------------------
    # 2. Support Vector Machine (SVM)
    # ---------------------------------------------------------
    logger.info("Optimizando SVM...")
    param_grid_svm = {
        'C': [0.01, 0.1, 0.5, 1.0, 5.0],
        'gamma': ['scale', 'auto', 0.01, 0.1]
    }
    svm = SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42)
    search_svm = RandomizedSearchCV(svm, param_grid_svm, n_iter=8, cv=tscv, scoring='roc_auc', n_jobs=-1, random_state=42)
    search_svm.fit(X_train_scaled, y_train)
    logger.info(f"SVM Best ROC AUC: {search_svm.best_score_:.4f}")
    best_params_all["SVM"] = search_svm.best_params_
    
    # ---------------------------------------------------------
    # 3. Random Forest
    # ---------------------------------------------------------
    logger.info("Optimizando Random Forest...")
    param_grid_rf = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 3, 5]
    }
    rf = RandomForestClassifier(class_weight='balanced', random_state=42)
    search_rf = RandomizedSearchCV(rf, param_grid_rf, n_iter=15, cv=tscv, scoring='roc_auc', n_jobs=-1, random_state=42)
    search_rf.fit(X_train_scaled, y_train)
    logger.info(f"Random Forest Best ROC AUC: {search_rf.best_score_:.4f}")
    best_params_all["Random Forest"] = search_rf.best_params_
    
    # ---------------------------------------------------------
    # 4. XGBoost
    # ---------------------------------------------------------
    logger.info("Optimizando XGBoost...")
    neg_count = sum(y_train == 0)
    pos_count = sum(y_train == 1)
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    
    param_grid_xgb = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'reg_alpha': [0, 0.1, 0.5, 1.0],
        'reg_lambda': [0.1, 1.0, 5.0]
    }
    xgb = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        random_state=42
    )
    search_xgb = RandomizedSearchCV(xgb, param_grid_xgb, n_iter=15, cv=tscv, scoring='roc_auc', n_jobs=-1, random_state=42)
    search_xgb.fit(X_train_scaled, y_train)
    logger.info(f"XGBoost Best ROC AUC: {search_xgb.best_score_:.4f}")
    
    def convert_types(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        return obj

    xgb_best = {k: convert_types(v) for k, v in search_xgb.best_params_.items()}
    best_params_all["XGBoost"] = xgb_best
    
    logger.info("\n--- RESULTADOS DEL TUNING ---")
    print(json.dumps(best_params_all, indent=4))
    
    os.makedirs("results", exist_ok=True)
    with open("results/best_hyperparameters.json", "w") as f:
        json.dump(best_params_all, f, indent=4)
        
    logger.info("Parámetros óptimos guardados en results/best_hyperparameters.json")

if __name__ == "__main__":
    main()
