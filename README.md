# Intelligent Portfolio Management (TFM)

**Intelligent Portfolio Management** es un sistema algorítmico avanzado de gestión de carteras diseñado como Trabajo de Fin de Máster (TFM). Combina el poder predictivo del Machine Learning (XGBoost) con reglas mecánicas e institucionales de gestión de riesgo para operar en los mercados financieros con una ventaja probabilística demostrable (Edge).

A diferencia de los sistemas de trading tradicionales, este proyecto se centra en la asimetría del riesgo/beneficio (+3R vs -1R) y en la protección del capital frente a la volatilidad del mercado, demostrando que la estadística pura de la IA bate a las decisiones basadas en emociones humanas.

---

## 🚀 Resultados Clave (Out-Of-Sample 2025)

El simulador evaluado sobre datos nunca antes vistos por el modelo (Test Set 2025) arrojó los siguientes resultados contra el mercado (Benchmark compuesto por SPY/QQQ):

- **Rentabilidad Total:** `+44.7%`
- **Max Drawdown:** `-16.0%`
- **Win Rate:** `45.5%`
- **Profit Factor:** `1.62`
- **Alpha (Anualizado):** `+27.83%` sobre el Benchmark.

---

## 🧠 Arquitectura del Sistema

El proyecto se divide en tres pilares fundamentales:

1. **Ingeniería de Datos (Data Pipeline):** Extracción, limpieza y generación de 44 variables técnicas (features), incluyendo indicadores de momento (RSI, ATR), medias móviles múltiples (SMA/EMA) en varios timeframes (1D, 1W), niveles de Fibonacci y patrones de Price Action.
2. **Motor Predictivo (Machine Learning):** Se entrenaron y evaluaron 4 algoritmos (Random Forest, Regresión Logística, SVM y XGBoost). **XGBoost** fue coronado como el "Rey Institucional" gracias a su altísima especificidad (66.59%), capaz de filtrar el ruido del mercado y atrapar tendencias sólidas.
3. **Simulador de Cartera Institucional (Environment):** Un entorno de backtesting tick-a-tick que simula el mundo real (comisiones, slippage y gestión de riesgo cruzada).

---

## ⚙️ La Configuración Maestra (El "Edge")

Tras realizar múltiples estudios de ablación técnica, se descubrió una **Sinergia Total** contraintuitiva. La configuración que extrae el Alfa absoluto del mercado es:

1. **Umbral Predictivo (0.32):** Se exige a la IA un 32% de probabilidad pura para entrar, filtrando señales de baja calidad sin caer en la parálisis por análisis.
2. **Gestión de Tendencia (Sin Break-Even):** *La Paradoja del Break-Even.* La IA rechaza el miedo humano. Mover el Stop Loss a cero asfixia al sistema; se debe dejar respirar el precio asumiendo una asimetría rígida de +3R o -1R.
3. **Gestión del Tiempo (Time-Stop a 30 días):** Libera el capital atrapado en operaciones "zombis" que no terminan de explotar.
4. **Gestión de la Frustración (Cooldown a 60 días):** Impide el *revenge trading* bloqueando operativas reiteradas en activos que han fallado, forzando la rotación de capital hacia activos sanos.

---

## 📂 Estructura del Directorio

```text
Intelligent_Portfolio_Management/
├── data/
│   ├── raw/               # Archivos CSV diarios originales
│   └── processed/         # Dataset etiquetado (MULTI_LABELED_DATASET.csv)
├── src/
│   ├── data/              # Scripts de descarga y procesamiento (Feature Engineering)
│   ├── models/            # Pipeline ML, agente lógico (agent_logic.py) y XGBoost
│   ├── environment/       # Simulador (run_simulation.py), Risk & Exit Managers
│   └── evaluation/        # Generación de métricas y gráficos comparativos
├── models/                # Modelos entrenados exportados (.pkl)
├── results/               # Logs de operaciones y figuras (SHAP, Benchmarks)
└── README.md              # Este archivo
```

---

## 🛠️ Instalación y Ejecución

### 1. Requisitos
Asegúrate de tener Python 3.10+ y crear un entorno virtual:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Entrenamiento del Modelo (Opcional)
Si deseas re-entrenar el modelo XGBoost desde cero con los datos procesados:
```bash
PYTHONPATH=. venv/bin/python3 src/models/ml_pipeline.py
```

### 3. Ejecutar la Simulación (Test Out-Of-Sample)
Para lanzar el simulador de la cartera institucional con las reglas maestras de riesgo:
```bash
PYTHONPATH=. venv/bin/python3 src/environment/run_simulation.py
```
> *Los resultados estadísticos y el registro de operaciones (Trade Log) se guardarán automáticamente en `results/logs/`.*

### 4. Generación de Gráficos (Evaluación)
Para generar los gráficos comparativos contra los Benchmarks y evaluar las métricas institucionales (Sharpe, Sortino, etc.):
```bash
PYTHONPATH=. venv/bin/python3 src/evaluation/plotting.py
PYTHONPATH=. venv/bin/python3 src/evaluation/metrics.py
```

---

## ⚠️ Limitaciones Conocidas (Trabajo Futuro)
- **Dependencia de Escala (Scale Dependency):** El modelo actual incorpora variables de precio absolutas (ej. `SMA_200`) normalizadas globalmente. Para futuras iteraciones, se recomienda usar exclusivamente variables de distancia relativas (`dist_SMA_200`) para aislar completamente el patrón de la escala de precio de la acción.

---
*Desarrollado para la investigación académica en Gestión Cuantitativa de Carteras y Machine Learning Financiero.*