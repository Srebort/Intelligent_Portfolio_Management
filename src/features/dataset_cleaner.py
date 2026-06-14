"""
Pipeline de Limpieza y Normalización del Dataset Final.

Toma el dataset resultante del MTF (con las velas de 4H y los indicadores
técnicos de 1D y 1W propagados), aplica los patrones de Price Action y 
purifica el conjunto de datos para que sea ingerible por modelos de Machine Learning.

Pasos del pipeline:
1. Calcular features de Price Action (patterns.py).
2. Purgar NaNs originados por el periodo de precalentamiento (ej. SMA 200).
3. Neutralizar valores infinitos (inf) procedentes de divisiones entre cero.
4. Escalar variables matemáticas usando StandardScaler, excluyendo precios base
   y variables lógicas/booleanas.
5. Guardar el dataset limpio y listo para el entrenamiento.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("DatasetCleaner")

class DatasetCleaner:
    def __init__(self, ticker: str, data_dir: str = "data/processed"):
        """
        Inicializa el limpiador de datos.

        Args:
            ticker (str): Símbolo del activo (ej. 'V' o 'SPY').
            data_dir (str): Directorio donde se encuentra el dataset MTF inicial
                            y donde se guardará el final.
        """
        self.ticker = ticker
        self.data_dir = Path(data_dir)
        self.scaler = StandardScaler()

    def _load_mtf_dataset(self) -> pd.DataFrame:
        """Carga el dataset generado por el MTFBuilder."""
        filepath = self.data_dir / f"{self.ticker}_4H_MTF.csv"
        if not filepath.exists():
            logger.error("No se encuentra el dataset MTF en: %s", filepath)
            return pd.DataFrame()
            
        df = pd.read_csv(filepath, index_col="datetime", parse_dates=True)
        # Asegurar que el índice tiene zona horaria UTC
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
            
        logger.info("Dataset MTF cargado (%d filas, %d columnas).", len(df), len(df.columns))
        return df

    def _add_price_action_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica la detección de patrones de Price Action."""
        from src.features.patterns import add_price_action_features
        df_pa = add_price_action_features(df)
        logger.info("Patrones de Price Action aplicados. Total columnas: %d", len(df_pa.columns))
        return df_pa

    def handle_infinites(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reemplaza valores infinitos (np.inf o -np.inf) originados por
        posibles divisiones entre cero por np.nan para que luego sean
        purgados o imputados.
        """
        inf_count = np.isinf(df.select_dtypes(include=[np.number])).values.sum()
        if inf_count > 0:
            logger.warning("Detectados %d valores infinitos. Reemplazando por NaN...", inf_count)
            df = df.replace([np.inf, -np.inf], np.nan)
        else:
            logger.info("No se detectaron valores infinitos.")
        return df

    def drop_nans(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Elimina todas las filas que contengan algún NaN. 
        Generalmente purgará las primeras 200 velas necesarias para la SMA 200.
        """
        initial_rows = len(df)
        df_clean = df.dropna()
        dropped = initial_rows - len(df_clean)
        
        logger.info("Eliminadas %d filas con NaNs (Periodo de precalentamiento). Filas útiles: %d", 
                    dropped, len(df_clean))
        return df_clean

    def scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Escala las características matemáticas usando StandardScaler.
        Ignora los precios OHLCV puros y las variables booleanas/discretas.
        """
        df_scaled = df.copy()
        
        # 1. Definir columnas intocables (Precios, volumen y las de timeframes superiores)
        base_cols = ['open', 'high', 'low', 'close', 'volume', 
                     'close_1D', 'close_1W']
                     
        # 2. Identificar columnas booleanas (0.0 / 1.0) o de patrones lógicos
        bool_cols = [col for col in df.columns if df[col].nunique() <= 2 or col.startswith('is_') or col.startswith('pattern_')]
        
        # 3. Columnas a excluir del escalado
        exclude_cols = set(base_cols + bool_cols)
        
        # 4. Columnas a escalar (todas las numéricas que no estén en excluidas)
        numeric_cols = df_scaled.select_dtypes(include=[np.number]).columns.tolist()
        scale_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        if not scale_cols:
            logger.warning("No se encontraron columnas matemáticas para escalar.")
            return df_scaled
            
        logger.info("Escalando %d características matemáticas (StandardScaler)...", len(scale_cols))
        
        # Aplicar scaler y mantener los nombres de columna e índice
        scaled_values = self.scaler.fit_transform(df_scaled[scale_cols])
        df_scaled[scale_cols] = scaled_values
        
        return df_scaled

    def build_final_dataset(self) -> pd.DataFrame:
        """
        Ejecuta todo el pipeline de limpieza y normalización.
        """
        logger.info("=== Iniciando Pipeline de Limpieza para '%s' ===", self.ticker)
        
        df = self._load_mtf_dataset()
        if df.empty:
            return df
            
        df = self._add_price_action_patterns(df)
        df = self.handle_infinites(df)
        df = self.drop_nans(df)
        df = self.scale_features(df)
        
        # Exportar resultado
        out_path = self.data_dir / f"{self.ticker}_ML_READY.csv"
        df.to_csv(out_path)
        logger.info("Dataset limpio y escalado guardado en: %s", out_path)
        
        return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, 
                        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
    
    print("\n=== SMOKE-TEST: Dataset Cleaner para V ===")
    cleaner = DatasetCleaner(ticker="V")
    df_clean = cleaner.build_final_dataset()
    
    if not df_clean.empty:
        print(f"\nDimensiones finales: {df_clean.shape}")
        print("\nMuestra de características escaladas (RSI, dist_SMA_200):")
        if 'RSI_14' in df_clean.columns and 'dist_SMA_200' in df_clean.columns:
            print(df_clean[['close', 'RSI_14', 'dist_SMA_200', 'is_bullish_divergence']].tail())
