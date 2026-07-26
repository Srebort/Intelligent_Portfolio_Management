"""
Módulo para el pipeline de Machine Learning.
Se encarga de cargar el dataset, limpiar variables del futuro (evitar data leakage),
y preparar los datos (partición cronológica y escalado) para el entrenamiento de los modelos predictivos.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("ML_Pipeline")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")

class MLPipeline:
    def __init__(self, data_path: str = "data/processed/MULTI_LABELED_DATASET.csv"):
        """
        Inicializa el pipeline de Machine Learning.
        """
        self.data_path = Path(data_path)
        self.df = None
        self.features = None
        self.target = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.entry_dates = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Carga el dataset etiquetado.
        """
        if not self.data_path.exists():
            logger.error(f"Archivo no encontrado: {self.data_path}")
            return pd.DataFrame()
            
        logger.info(f"Cargando dataset desde {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        
        if 'fecha_entrada' in self.df.columns:
            self.df['fecha_entrada'] = pd.to_datetime(self.df['fecha_entrada'])
            # Ordenar por fecha_entrada es crucial para evitar data leakage en TimeSeriesSplit
            self.df = self.df.sort_values(by=['fecha_entrada', 'Ticker']).reset_index(drop=True)
            
        logger.info(f"Dataset cargado con éxito. Dimensiones originales: {self.df.shape}")
        return self.df
        
    def filter_leakage_columns(self):
        """
        Elimina las variables del futuro o aquellas que revelarían el resultado de la operación
        (data leakage), manteniendo sólo las variables conocidas en el momento de la entrada.
        """
        if self.df is None or self.df.empty:
            logger.error("Dataset no cargado. Llama a load_data() primero.")
            return

        cols_to_drop = []
        
        for col in self.df.columns:
            # Eliminar columnas de precios de TP/SL, velas de ejecución, hits y etiquetas alternativas
            if col.endswith('_precio') or col.endswith('_vela') or col.endswith('_hit'):
                cols_to_drop.append(col)
            elif col.startswith('label_'):
                cols_to_drop.append(col)
                
        # Mantener solo 'Label' como objetivo
        cols_to_drop = list(set(cols_to_drop))
        
        logger.info(f"Eliminando {len(cols_to_drop)} columnas para evitar Data Leakage.")
        self.df = self.df.drop(columns=cols_to_drop, errors='ignore')
        logger.info(f"Dimensiones tras filtro anti-leakage: {self.df.shape}")
        
    def prepare_features_and_target(self, target_col: str = 'Label', drop_cols: list = None):
        """
        Separa el DataFrame en matriz de características (X) y vector objetivo (y).
        Maneja valores nulos y variables categóricas (como 'Tier').
        """
        if drop_cols is None:
            drop_cols = ['fecha_entrada', 'Ticker']
            
        # Eliminar NaNs
        before_drop = len(self.df)
        self.df = self.df.dropna()
        after_drop = len(self.df)
        if before_drop != after_drop:
            logger.info(f"Filas con NaNs eliminadas: {before_drop - after_drop}")
            
        # One-Hot Encoding para variables categóricas si las hay
        if 'Tier' in self.df.columns:
            self.df = pd.get_dummies(self.df, columns=['Tier'], drop_first=False)
            
        # Separar features y target
        self.target = self.df[target_col].astype(int)
        
        features_df = self.df.drop(columns=[target_col] + drop_cols, errors='ignore')
        
        # Guardar lista de features para el feature importance luego
        self.feature_names = features_df.columns.tolist()
        self.features = features_df.values
        
        # Guardar fechas para el split cronológico manual
        if 'fecha_entrada' in self.df.columns:
            self.entry_dates = self.df['fecha_entrada'].copy()
        
        logger.info(f"Features preparadas: {self.features.shape[1]} columnas.")
        
    def split_and_scale(self, n_splits: int = 5):
        """
        Aplica TimeSeriesSplit para la partición del dataset respetando la cronología,
        y escala las variables numéricas (crucial para SVM, Logistic Regression).
        
        Yields:
            X_train_scaled, X_test_scaled, y_train, y_test
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        for fold, (train_index, test_index) in enumerate(tscv.split(self.features)):
            X_train, X_test = self.features[train_index], self.features[test_index]
            y_train, y_test = self.target.iloc[train_index], self.target.iloc[test_index]
            
            # Escalar. El scaler se ajusta SOLO a los datos de entrenamiento
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            yield X_train_scaled, X_test_scaled, y_train, y_test

    def train_test_split_by_date(self, split_date: str = "2024-01-01"):
        """
        Divide el dataset en Train (antes del split_date) y Test (después).
        Escala los datos ajustando el scaler SOLO con el conjunto de Train (Out-of-Sample).
        """
        if self.entry_dates is None:
            logger.error("No se encontraron fechas (entry_dates) para hacer el split.")
            return None, None, None, None

        split_ts = pd.to_datetime(split_date, utc=True)
        # Convertir entry_dates a UTC si no lo están para poder comparar
        dates_utc = pd.to_datetime(self.entry_dates, utc=True)
        
        train_mask = dates_utc < split_ts
        test_mask = dates_utc >= split_ts
        
        train_mask = (dates_utc < split_ts).values
        test_mask = (dates_utc >= split_ts).values
        
        X_train, X_test = self.features[train_mask], self.features[test_mask]
        y_train, y_test = self.target.iloc[train_mask], self.target.iloc[test_mask]
        
        # Escalar SOLO ajustando en Train
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        logger.info(f"Split por Fecha ({split_date}): Train={X_train.shape[0]} filas | Test={X_test.shape[0]} filas")
        return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == "__main__":
    # Prueba rápida del pipeline
    pipeline = MLPipeline()
    pipeline.load_data()
    pipeline.filter_leakage_columns()
    pipeline.prepare_features_and_target()
    
    if pipeline.features is not None:
        X_train, X_test, y_train, y_test = pipeline.train_test_split_by_date()
        logger.info(f"OOS Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
