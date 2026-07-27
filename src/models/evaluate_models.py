"""
Script de evaluación y selección de modelos de Machine Learning (Sprint 4).
Entrena y compara los modelos base y avanzados (LogReg, SVM, RF, XGBoost),
extrae la importancia de las variables (Feature Importance), calcula valores SHAP,
genera curvas de calibración y exporta TODOS los modelos.
"""

import logging
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.calibration import CalibrationDisplay
from sklearn.inspection import permutation_importance

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
        "SVM": SVMModel(C=0.5, kernel='rbf', probability=True),
        "RF": RandomForestModel(n_estimators=50, max_depth=3),
        "XGBoost": TradeSelectorXGB()
    }
    
    results = []
    
    # Evaluar con validación Walk-Forward (Train < 2025, Test >= 2025)
    X_train, X_test, y_train, y_test = pipeline.train_test_split_by_date(split_date="2025-01-01")
    
    if X_train is None or len(X_train) == 0:
        logger.error("Error en el split. Abortando evaluación.")
        return
        
    print(f"\nEntrenando Out-Of-Sample en {len(y_train)} muestras (antes de 2025)...")
    print(f"Evaluando en {len(y_test)} muestras (2025 en adelante)...")
    
    # --- Directorios de resultados ---
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    figures_dir = Path("results/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    fig_calib, ax_calib = plt.subplots(figsize=(8, 6))
    
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
            "Accuracy": metrics.get('accuracy', 0),
            "Precision": metrics.get('precision', 0),
            "Recall": metrics.get('recall', 0),
            "Specificity": metrics.get('specificity', 0),
            "F1 Score": metrics.get('f1', 0),
            "ROC AUC": metrics.get('roc_auc', 0),
            "Brier Score": metrics.get('brier_score', 0),
            "TN": cm[0][0] if cm.shape == (2,2) else 0,
            "FP": cm[0][1] if cm.shape == (2,2) else 0,
            "FN": cm[1][0] if cm.shape == (2,2) else 0,
            "TP": cm[1][1] if cm.shape == (2,2) else 0
        })
        
        # Guardar todos los modelos
        model_path = models_dir / f"{name}.pkl"
        joblib.dump(model.model, model_path)
        print(f"[{name}] Guardado en {model_path}")
        
        # Curva de Calibración
        try:
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_test)[:, 1]
                CalibrationDisplay.from_predictions(y_test, probs, n_bins=10, ax=ax_calib, name=name)
        except Exception as e:
            logger.warning(f"No se pudo generar calibración para {name}: {e}")

    joblib.dump(pipeline.scaler, models_dir / "scaler.pkl")
    
    ax_calib.set_title("Curvas de Calibración (Reliability Diagrams)")
    fig_calib.savefig(figures_dir / "calibration_curves.png")
    plt.close(fig_calib)
            
    # --- Generar Tabla Comparativa ---
    print(f"\n{'Modelo':<10} | {'Acc':<6} | {'Prec':<6} | {'Recall':<6} | {'Spec':<6} | {'F1':<6} | {'ROC AUC':<7} | {'Brier':<7}")
    print("-" * 80)
    sorted_results = sorted(results, key=lambda x: x['F1 Score'], reverse=True)
    for res in sorted_results:
        print(f"{res['Model']:<10} | {res['Accuracy']:.4f} | {res['Precision']:.4f} | {res['Recall']:.4f} | {res['Specificity']:.4f} | {res['F1 Score']:.4f} | {res['ROC AUC']:.4f}  | {res['Brier Score']:.4f}")
    
    # --- Matriz de Confusión Financiera ---
    print("\n" + "=" * 75)
    print("--- MATRIZ DE CONFUSIÓN FINANCIERA (R-Multiples) ---")
    print("Asumiendo: TP = +3R, FP = -1R, TN = 0R (Ahorro de -1R), FN = Coste de Oportunidad")
    print("=" * 75)
    print(f"{'Modelo':<10} | {'TP (Gana 3R)':<15} | {'FP (Pierde 1R)':<15} | {'Balance Neto (R)':<18} | {'TN (Ahorra 1R)':<15}")
    print("-" * 80)
    for res in sorted_results:
        tp_r = res['TP'] * 3
        fp_r = res['FP'] * -1
        net_r = tp_r + fp_r
        print(f"{res['Model']:<10} | {res['TP']} ops (+{tp_r}R)  | {res['FP']} ops ({fp_r}R)   | {net_r:>+5}R             | {res['TN']} ops")

    # --- Feature Importance y SHAP ---
    print("\n" + "=" * 75)
    print("--- XAI: EXPLICABILIDAD Y FEATURE IMPORTANCE ---")
    print("=" * 75)
    
    # 1. SHAP Values para Todos los Modelos
    for name, model in models.items():
        try:
            print(f"\nCalculando SHAP Values para {name} (sobre Test Set)...")
            m = model.model
            
            X_plot = X_test
            
            if name in ["XGBoost", "RF"]:
                explainer = shap.TreeExplainer(m)
                shap_values = explainer.shap_values(X_test)
            elif name == "LogReg":
                explainer = shap.LinearExplainer(m, X_train)
                shap_values = explainer.shap_values(X_test)
            elif name == "SVM":
                print("   (KernelExplainer en SVM es lento: usando un subset de 100 muestras...)")
                background = shap.kmeans(X_train, 10)
                # m.predict_proba es necesario en SVM para KernelExplainer si es clasificación
                explainer = shap.KernelExplainer(m.predict_proba, background)
                np.random.seed(42)
                sample_idx = np.random.choice(X_test.shape[0], 100, replace=False)
                X_plot = X_test[sample_idx]
                shap_values = explainer.shap_values(X_plot)
            
            # Formatear el array de SHAP values (Lista, 3D o 2D)
            if isinstance(shap_values, list):
                shap_values_to_plot = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
                shap_values_to_plot = shap_values[:, :, 1]
            else:
                shap_values_to_plot = shap_values
                
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values_to_plot, X_plot, feature_names=pipeline.feature_names, show=False)
            plt.title(f"SHAP Summary Plot - {name}")
            plt.tight_layout()
            plt.savefig(figures_dir / f"shap_summary_{name.lower()}.png")
            plt.close()
            print(f" -> Gráfico SHAP guardado en {figures_dir / f'shap_summary_{name.lower()}.png'}")
        except Exception as e:
            print(f"Error calculando SHAP para {name}: {e}")
        
    # 2. Permutation Importance
    print("\nCalculando Permutation Importance (Test Set) para Random Forest...")
    try:
        rf_model = models["RF"].model
        perm_importance = permutation_importance(rf_model, X_test, y_test, n_repeats=10, random_state=42, scoring='f1')
        
        perm_df = pd.DataFrame({
            "Feature": pipeline.feature_names,
            "Importance_Mean": perm_importance.importances_mean,
            "Importance_Std": perm_importance.importances_std
        }).sort_values(by="Importance_Mean", ascending=False).head(10)
        
        for _, row in perm_df.iterrows():
            print(f"  -> {row['Feature']:<25} : {row['Importance_Mean']:.4f} (+/- {row['Importance_Std']:.4f})")
    except Exception as e:
        print(f"Error calculando Permutation Importance: {e}")

    # --- Curva de Capital (Simulación Walk-Forward) ---
    print("\n" + "=" * 75)
    print("--- CURVA DE CAPITAL COMPARATIVA (Walk-Forward 2025+) ---")
    print("=" * 75)
    
    try:
        # Array con los resultados reales de cada operación (1=+3R, 0=-1R)
        y_test_r = np.where(y_test == 1, 3, -1)
        
        # Baseline: Opera todo
        equity_baseline = np.cumsum(y_test_r)
        
        plt.figure(figsize=(12, 7))
        plt.plot(equity_baseline, label=f"Baseline (Todo) [Final: {equity_baseline[-1]}R]", color='gray', alpha=0.5, linestyle='--')
        
        colors = {"LogReg": "orange", "SVM": "purple", "RF": "blue", "XGBoost": "green"}
        
        # Iterar sobre todos los modelos para pintar su curva
        for name, model in models.items():
            # Usar .predict() estándar para que cuadre exactamente con la Matriz de Confusión
            preds = model.predict(X_test)
            equity_model = np.cumsum(np.where(preds == 1, y_test_r, 0))
            
            line_w = 2.5 if name == "XGBoost" else 1.5
            plt.plot(equity_model, label=f"{name} [Final: {equity_model[-1]}R]", 
                     color=colors.get(name, "black"), linewidth=line_w)
        
        plt.title("Curva de Capital Walk-Forward Comparativa (Acumulación R-Múltiplos)")
        plt.xlabel("Señales Categóricas en Test Set (2025+)")
        plt.ylabel("R-Múltiplos Acumulados")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(figures_dir / "equity_curve_comparison.png")
        plt.close()
        print(f" -> Gráfico de Curva de Capital guardado en {figures_dir / 'equity_curve_comparison.png'}")
        
    except Exception as e:
        print(f"Error generando Curva de Capital: {e}")

    print("\nEvaluación y exportación completadas exitosamente.")
    print("=" * 75)

if __name__ == "__main__":
    main()
