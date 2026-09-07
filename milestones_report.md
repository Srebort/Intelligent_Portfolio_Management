# 📋 Milestones e Issues — Intelligent_Portfolio_Management

> Generado el 03/08/2026 a las 19:32  
> Repositorio: [Srebort/Intelligent_Portfolio_Management](https://github.com/Srebort/Intelligent_Portfolio_Management)

---

## Índice de Milestones

- ✅ Sprint 1: Infraestructura de Datos
- ✅ Sprint 2: Ingeniería de Características
- ✅ Sprint 3: Backtester y Generación de Etiquetas
- ✅ Sprint 4: Entrenamiento de Modelos de Machine Learning
- ✅ Sprint 5: Entorno de Simulación, Cartera y Riesgo
- ✅ Sprint 6: Motor de Simulación y Rotación de Cartera
- ✅ Sprint 7: Análisis Cuantitativo y Visualización
- 🔵 Sprint 8: Redacción de la memoria y corrección de fallos

---

## Sprint 1: Infraestructura de Datos

| Campo            | Valor |
|------------------|-------|
| **Estado**       | ✅ Cerrada |
| **Fecha límite** | 30/05/2026 |
| **Cerrada el**   | 04/06/2026 |
| **Progreso**     | 7/7 issues completadas (100%) |

### Descripción

Conseguir que el proyecto se conecte a las APIs, descargue los datos y los guarde limpios en el disco duro para no volver a depender de internet.

**Progreso:** `[████████████████████]` 100%

### Issues (4)

#### ✅ \#1 — Configuración de la arquitectura base del proyecto y variables de entorno

| Campo   | Valor |
|---------|-------|
| Estado  | ✅ Cerrada |
| Labels  | `enhancement` |
| Creada  | 17/05/2026 |
| Cerrada | 27/05/2026 |
| URL     | [https://github.com/Srebort/Intelligent_Portfolio_Management/issues/1](https://github.com/Srebort/Intelligent_Portfolio_Management/issues/1) |

> ###  Objetivo Establecer la estructura de directorios inicial del TFM, configurar la gestión de dependencias y aislar las credenciales de las APIs por seguridad.

#### ✅ \#2 — Desarrollo del módulo de extracción de Tiingo

| Campo   | Valor |
|---------|-------|
| Estado  | ✅ Cerrada |
| Labels  | `enhancement` |
| Creada  | 17/05/2026 |
| Cerrada | 03/06/2026 |
| URL     | [https://github.com/Srebort/Intelligent_Portfolio_Management/issues/2](https://github.com/Srebort/Intelligent_Portfolio_Management/issues/2) |

> ###  Objetivo Programar el script encargado de conectarse a la API de Tiingo (endpoint IEX) para descargar el histórico de precios intradiarios de los activos definidos en la configuración.

#### ✅ \#3 — Desarrollo del módulo de extracción de la FRED

| Campo   | Valor |
|---------|-------|
| Estado  | ✅ Cerrada |
| Labels  | `enhancement` |
| Creada  | 17/05/2026 |
| Cerrada | 03/06/2026 |
| URL     | [https://github.com/Srebort/Intelligent_Portfolio_Management/issues/3](https://github.com/Srebort/Intelligent_Portfolio_Management/issues/3) |

> ###  Objetivo Programar el script para descargar el contexto macroeconómico y de sentimiento de mercado desde la base de datos del Banco Central de EE.UU. (FRED).

#### ✅ \#4 — Desarrollo del pipeline de fusión de datos

| Campo   | Valor |
|---------|-------|
| Estado  | ✅ Cerrada |
| Labels  | `enhancement` |
| Creada  | 17/05/2026 |
| Cerrada | 03/06/2026 |
| URL     | [https://github.com/Srebort/Intelligent_Portfolio_Management/issues/4](https://github.com/Srebort/Intelligent_Portfolio_Management/issues/4) |

> ###  Objetivo Alinear temporalmente las velas intradiarias (4H) con los indicadores macroeconómicos (diarios/mensuales) para generar el dataset definitivo que consumirá el modelo de Machine Learning.

---

## Sprint 2: Ingeniería de Características

| Campo            | Valor |
|------------------|-------|
| **Estado**       | ✅ Cerrada |
| **Fecha límite** | 13/06/2026 |
| **Cerrada el**   | 16/06/2026 |
| **Progreso**     | 9/9 issues completadas (100%) |

### Descripción

Traducir los precios puros a matemáticas e indicadores que el modelo pueda entender.

**Progreso:** `[████████████████████]` 100%

### Issues (4)

#### ✅ \#10 — Desarrollo de Indicadores Técnicos Clásicos (RSI y Medias Móviles)

| Campo   | Valor |
|---------|-------|
| Estado  | ✅ Cerrada |
| Labels  | `enhancement` |
| Creada  | 03/06/2026 |
| Cerrada | 04/06/2026 |
| URL     | [https://github.com/Srebort/Intelligent_Portfolio_Management/issues/10](https://github.com/Srebort/Intelligent_Portfolio_Management/issues/10) |

> ## Transformar la serie temporal de precios en características (*features*) matemáticas estandarizadas que ayuden al modelo a identificar la tendencia y el momentum del mercado. ###  Criterios de Acep…

#### ✅ \#11 — Implementación de métricas de Volatilidad (ATR)

| Campo   | Valor |
|---------|-------|
| Estado  | ✅ Cerrada |
| Labels  | `enhancement` |
| Creada  | 03/06/2026 |
| Cerrada | 04/06/2026 |
| URL     | [https://github.com/Srebort/Intelligent_Portfolio_Management/issues/11](https://github.com/Srebort/Intelligent_Portfolio_Management/issues/11) |

> ### Calcular el *Average True Range* (ATR) para cuantificar el ruido y la volatilidad de cada activo. Esta métrica no se usará tanto para predecir, sino que será el pilar fundamental del módulo de `Ri…

#### ✅ \#12 — Desarrollo del algoritmo de detección de patrones (Fractales y Wick Reclaims)

| Campo   | Valor |
|---------|-------|
| Estado  | ✅ Cerrada |
| Labels  | `enhancement` |
| Creada  | 03/06/2026 |
| Cerrada | 14/06/2026 |
| URL     | [https://github.com/Srebort/Intelligent_Portfolio_Management/issues/12](https://github.com/Srebort/Intelligent_Portfolio_Management/issues/12) |

> ### Objetivo Traducir conceptos visuales de *Price Action* (Acción del Precio) a lógica de código duro. El modelo necesita saber matemáticamente cuándo se ha formado un suelo/techo local o cuándo el p…

#### ✅ \#13 — Pipeline de Limpieza y Normalización del Dataset Final

| Campo   | Valor |
|---------|-------|
| Estado  | ✅ Cerrada |
| Labels  | `enhancement` |
| Creada  | 03/06/2026 |
| Cerrada | 14/06/2026 |
| URL     | [https://github.com/Srebort/Intelligent_Portfolio_Management/issues/13](https://github.com/Srebort/Intelligent_Portfolio_Management/issues/13) |

> ### Objetivo Ensamblar todas las características calculadas junto con los precios base y purgar el dataset de valores nulos o infinitos para que esté 100% listo para ser ingerido por el algoritmo de M…

---

## Sprint 3: Backtester y Generación de Etiquetas

| Campo            | Valor |
|------------------|-------|
| **Estado**       | ✅ Cerrada |
| **Fecha límite** | 28/06/2026 |
| **Cerrada el**   | 01/07/2026 |
| **Progreso**     | 6/6 issues completadas (100%) |

### Descripción

Este Sprint se centra en construir el motor de simulación histórico (Backtester) que recorrerá las señales emitidas por el sistema para determinar si resultaron en Ganancia (1) o Pérdida (0). Se implementará una lógica de negocio estricta para resolver la gestión del riesgo dinámico, los bloqueos temporales y los casos límite, asegurando que el dataset resultante (ML_READY_LABELED.csv) esté perfectamente etiquetado para entrenar a los modelos de Inteligencia Artificial sin sesgos ni fugas de información.

**Progreso:** `[████████████████████]` 100%

### Issues (4)

#### ✅ \#17 — Implementar la Lógica Core del Backtester (Gestión de Riesgo y Ratios)

| Campo   | Valor |
|---------|-------|
| Estado  | ✅ Cerrada |
| Labels  | `enhancement` |
| Creada  | 14/06/2026 |
| Cerrada | 17/06/2026 |
| URL     | [https://github.com/Srebort/Intelligent_Portfolio_Management/issues/17](https://github.com/Srebort/Intelligent_Portfolio_Management/issues/17) |

> Crear el módulo base src/environment/backtester.py encargado de simular el futuro de cada señal y calcular si la operación es ganadora o perdedora: - [x] Crear la estructura de la clase StrictBacktest…

#### ✅ \#18 — Resolución de Conflictos Temporales y Casos Límite

| Campo   | Valor |
|---------|-------|
| Estado  | ✅ Cerrada |
| Labels  | `enhancement` |
| Creada  | 14/06/2026 |
| Cerrada | 01/07/2026 |
| URL     | [https://github.com/Srebort/Intelligent_Portfolio_Management/issues/18](https://github.com/Srebort/Intelligent_Portfolio_Management/issues/18) |

> Añadir reglas de validación pesimistas al Backtester para satisfacer los requisitos académicos del tribunal y evitar sobreestimar los resultados: - [x] Implementar la regla de Time-Out: si la operació…

#### ✅ \#19 — Integración y Generación del Dataset Etiquetado

| Campo   | Valor |
|---------|-------|
| Estado  | ✅ Cerrada |
| Labels  | `enhancement` |
| Creada  | 14/06/2026 |
| Cerrada | 01/07/2026 |
| URL     | [https://github.com/Srebort/Intelligent_Portfolio_Management/issues/19](https://github.com/Srebort/Intelligent_Portfolio_Management/issues/19) |

> Conectar los módulos analíticos para producir el archivo final etiquetado sobre el que aprenderá el Machine Learning: - [x] Crear el script orquestador generate_labeled_dataset.py.

#### ✅ \#20 — Redacción de la Memoria (Metodología, Datos y Arquitectura)

| Campo   | Valor |
|---------|-------|
| Estado  | ✅ Cerrada |
| Labels  | `documentation` |
| Creada  | 14/06/2026 |
| Cerrada | 01/07/2026 |
| URL     | [https://github.com/Srebort/Intelligent_Portfolio_Management/issues/20](https://github.com/Srebort/Intelligent_Portfolio_Management/issues/20) |

> Empezar la redacción formal de los primeros capítulos del TFM siguiendo las indicaciones del tribunal, aprovechando que la Fase 1 (Ingesta de Datos) y la Fase 2 (Ingeniería de Características) están f…

---

## Sprint 4: Entrenamiento de Modelos de Machine Learning

| Campo            | Valor |
|------------------|-------|
| **Estado**       | ✅ Cerrada |
| **Fecha límite** | 09/07/2026 |
| **Cerrada el**   | 03/07/2026 |
| **Progreso**     | 7/7 issues completadas (100%) |

### Descripción

Desarrollo e integración de la capa de inteligencia predictiva del bot. Utilizando el dataset etiquetado de la Verdad Fundamental (Sprint 3), se entrenarán múltiples modelos supervisados (Logistic Regression, SVM, Random Forest y XGBoost) para clasificar la viabilidad de las operaciones detectadas por Acción del Precio. 
El objetivo es reducir los falsos positivos descubriendo patrones ocultos no lineales entre los indicadores técnicos (RSI, ATR, Medias) y el contexto macroeconómico, logrando que el sistema rechace operaciones matemáticamente desfavorables antes de ejecutarlas.

**Progreso:** `[████████████████████]` 100%

### Issues (4)

#### ✅ \#24 — Desarrollar módulo `ml_pipeline.py` para la carga y preparación de datos.

| Campo   | Valor |
|---------|-------|
| Estado  | ✅ Cerrada |
| Labels  | `enhancement` |
| Creada  | 01/07/2026 |
| Cerrada | 01/07/2026 |
| URL     | [https://github.com/Srebort/Intelligent_Portfolio_Management/issues/24](https://github.com/Srebort/Intelligent_Portfolio_Management/issues/24) |

> - [x] Cargar `V_LABELED_DATASET.csv`. - [x] Implementar lógica de eliminación de variables futuras (evitar data leakage).

#### ✅ \#25 — Implementar clasificadores predictivos base (LogReg, SVM, RandomForest).

| Campo   | Valor |
|---------|-------|
| Estado  | ✅ Cerrada |
| Labels  | `enhancement` |
| Creada  | 01/07/2026 |
| Cerrada | 01/07/2026 |
| URL     | [https://github.com/Srebort/Intelligent_Portfolio_Management/issues/25](https://github.com/Srebort/Intelligent_Portfolio_Management/issues/25) |

> - [x] Crear las clases correspondientes en el módulo de modelos. - [x] Entrenar los modelos con los datos procesados, configurando hiperparámetros iniciales para evitar el sobreajuste (overfitting) de…

#### ✅ \#26 — Implementar clasificador predictivo avanzado basado en XGBoost.

| Campo   | Valor |
|---------|-------|
| Estado  | ✅ Cerrada |
| Labels  | `enhancement` |
| Creada  | 01/07/2026 |
| Cerrada | 03/07/2026 |
| URL     | [https://github.com/Srebort/Intelligent_Portfolio_Management/issues/26](https://github.com/Srebort/Intelligent_Portfolio_Management/issues/26) |

> - [x] Rellenar el esqueleto de la clase `TradeSelectorXGB` en `xgboost_filter.py`. - [x] Entrenar el modelo aplicando regularización fuerte (L1/L2) para maximizar la generalización en conjuntos de dat…

#### ✅ \#27 — Evaluar el rendimiento, comparar los 4 modelos y analizar el Feature Importance.

| Campo   | Valor |
|---------|-------|
| Estado  | ✅ Cerrada |
| Labels  | `enhancement` |
| Creada  | 01/07/2026 |
| Cerrada | 03/07/2026 |
| URL     | [https://github.com/Srebort/Intelligent_Portfolio_Management/issues/27](https://github.com/Srebort/Intelligent_Portfolio_Management/issues/27) |

> - [x] Calcular Matriz de Confusión, Precision, Recall y F1-Score para los 4 modelos (LogReg, SVM, Random Forest, XGBoost) y generar una tabla comparativa. - [x] Extraer y visualizar las 10 característ…

---

## Sprint 5: Entorno de Simulación, Cartera y Riesgo

| Campo            | Valor |
|------------------|-------|
| **Estado**       | ✅ Cerrada |
| **Fecha límite** | 09/07/2026 |
| **Cerrada el**   | 07/07/2026 |
| **Progreso**     | 5/5 issues completadas (100%) |

### Descripción

El objetivo de este Sprint es construir el "Cerebro Financiero" del sistema. Conectaremos las predicciones de la IA (Machine Learning) con la simulación de la Billetera (Portfolio), y aplicaremos fórmulas matemáticas de gestión monetaria (RiskManager) para decidir automáticamente qué operaciones realizar y qué porcentaje exacto de capital invertir sin arriesgar jamás más del 2% de la cartera.

**Progreso:** `[████████████████████]` 100%

### Issues (3)

#### ✅ \#31 — Completar la Clase Portfolio (Billetera Virtual)

| Campo   | Valor |
|---------|-------|
| Estado  | ✅ Cerrada |
| Labels  | `enhancement` |
| Creada  | 03/07/2026 |
| Cerrada | 03/07/2026 |
| URL     | [https://github.com/Srebort/Intelligent_Portfolio_Management/issues/31](https://github.com/Srebort/Intelligent_Portfolio_Management/issues/31) |

> Implementar la lógica de ejecución comercial en el Portfolio. Rellenar la lógica interna del simulador financiero para registrar las compras, ventas y calcular el balance diario. El archivo 'src/envir…

#### ✅ \#32 — Construir el PortfolioAgent

| Campo   | Valor |
|---------|-------|
| Estado  | ✅ Cerrada |
| Labels  | `enhancement` |
| Creada  | 03/07/2026 |
| Cerrada | 07/07/2026 |
| URL     | [https://github.com/Srebort/Intelligent_Portfolio_Management/issues/32](https://github.com/Srebort/Intelligent_Portfolio_Management/issues/32) |

> Integrar IA Predictiva y RiskManager en el Agente de Cartera. Convertir el esqueleto vacío de agent_logic.py en el orquestador principal que toma la decisión final de inversión. El Agente debe conecta…

#### ✅ \#33 — Estrategia de Rebalanceo Dinámico

| Campo   | Valor |
|---------|-------|
| Estado  | ✅ Cerrada |
| Labels  | `enhancement` |
| Creada  | 03/07/2026 |
| Cerrada | 07/07/2026 |
| URL     | [https://github.com/Srebort/Intelligent_Portfolio_Management/issues/33](https://github.com/Srebort/Intelligent_Portfolio_Management/issues/33) |

> Añadir lógicas defensivas (Time-Stop y Volatilidad). Dotar al Agente de reglas de protección de capital pasivas para mercados estancados o de pánico. Dado que es un sistema Swing Trading, la cartera n…

---

## Sprint 6: Motor de Simulación y Rotación de Cartera

| Campo            | Valor |
|------------------|-------|
| **Estado**       | ✅ Cerrada |
| **Fecha límite** | 15/07/2026 |
| **Cerrada el**   | 12/07/2026 |
| **Progreso**     | 5/5 issues completadas (100%) |

### Descripción

n el Sprint anterior dotamos de inteligencia al sistema (IA Predictiva + Risk Management). En este Sprint vamos a construir la "Máquina del Tiempo" que hará que todo cobre vida. El objetivo es desarrollar el bucle principal de simulación (Historical Loop) que recorrerá el dataset día a día, comprobando si debemos cerrar operaciones abiertas (Stop Loss, Take Profit o Time-Stop), actualizando los precios del mercado en la Cartera, procesando nuevas señales de compra y, lo más importante, registrando la curva de ganancias diaria (Equity Curve) para tener las métricas finales.

**Progreso:** `[████████████████████]` 100%

### Issues (3)

#### ✅ \#36 — Lógicas de Cierre (Ventas)

| Campo   | Valor |
|---------|-------|
| Estado  | ✅ Cerrada |
| Labels  | `enhancement` |
| Creada  | 05/07/2026 |
| Cerrada | 12/07/2026 |
| URL     | [https://github.com/Srebort/Intelligent_Portfolio_Management/issues/36](https://github.com/Srebort/Intelligent_Portfolio_Management/issues/36) |

> Desarrollar funciones de gestión de salidas y cierre de operaciones. Implementar la lógica que revisa diariamente las operaciones abiertas para decidir si deben venderse y devolver el capital al Portf…

#### ✅ \#37 — Bucle Principal de Simulación (Historical Loop)

| Campo   | Valor |
|---------|-------|
| Estado  | ✅ Cerrada |
| Labels  | `enhancement` |
| Creada  | 05/07/2026 |
| Cerrada | 12/07/2026 |
| URL     | [https://github.com/Srebort/Intelligent_Portfolio_Management/issues/37](https://github.com/Srebort/Intelligent_Portfolio_Management/issues/37) |

> Construir el Orquestador Temporal de Backtesting. Desarrollar el loop cronológico que simula el paso del tiempo en el mercado financiero día a día. Este es el motor central del TFM. Debe procesar toda…

#### ✅ \#38 — Tracking Histórico de la Cartera (Equity Curve)

| Campo   | Valor |
|---------|-------|
| Estado  | ✅ Cerrada |
| Labels  | `enhancement` |
| Creada  | 05/07/2026 |
| Cerrada | 12/07/2026 |
| URL     | [https://github.com/Srebort/Intelligent_Portfolio_Management/issues/38](https://github.com/Srebort/Intelligent_Portfolio_Management/issues/38) |

> Implementar el registro estadístico de la cuenta. Guardar el estado de la billetera virtual cada día simulado para poder generar las métricas de rentabilidad. Sin un registro milimétrico, no podremos …

---

## Sprint 7: Análisis Cuantitativo y Visualización

| Campo            | Valor |
|------------------|-------|
| **Estado**       | ✅ Cerrada |
| **Fecha límite** | 19/07/2026 |
| **Cerrada el**   | 27/07/2026 |
| **Progreso**     | 6/6 issues completadas (100%) |

### Descripción

Desarrollar el motor de cálculo de métricas institucionales (Sharpe, Max Drawdown real) y generar los gráficos de rendimiento comparando el comportamiento del bot contra el mercado (Benchmark S&P 500). Esto es el núcleo para la defensa de resultados del TFM.

**Progreso:** `[████████████████████]` 100%

### Issues (4)

#### ✅ \#40 — Corregir bug del Drawdown irreal y estructurar los logs

| Campo   | Valor |
|---------|-------|
| Estado  | ✅ Cerrada |
| Labels  | `enhancement` |
| Creada  | 12/07/2026 |
| Cerrada | 13/07/2026 |
| URL     | [https://github.com/Srebort/Intelligent_Portfolio_Management/issues/40](https://github.com/Srebort/Intelligent_Portfolio_Management/issues/40) |

> Solucionar el error matemático en la simulación por el cual la cuenta cae artificialmente cuando una posición abierta no registra un precio de cierre en días en los que no hay señales nuevas. Además, …

#### ✅ \#41 — Script de Integración del Benchmark S&P 500

| Campo   | Valor |
|---------|-------|
| Estado  | ✅ Cerrada |
| Labels  | `enhancement` |
| Creada  | 12/07/2026 |
| Cerrada | 13/07/2026 |
| URL     | [https://github.com/Srebort/Intelligent_Portfolio_Management/issues/41](https://github.com/Srebort/Intelligent_Portfolio_Management/issues/41) |

> Desarrollar un script para descargar de forma automática el histórico de precios del mercado (S&P 500) para el periodo exacto que duró la simulación Out-Of-Sample. Esto servirá de base (Buy & Hold) pa…

#### ✅ \#42 — Motor de Métricas Financieras (Sharpe, Alpha, Beta)

| Campo   | Valor |
|---------|-------|
| Estado  | ✅ Cerrada |
| Labels  | `enhancement` |
| Creada  | 12/07/2026 |
| Cerrada | 13/07/2026 |
| URL     | [https://github.com/Srebort/Intelligent_Portfolio_Management/issues/42](https://github.com/Srebort/Intelligent_Portfolio_Management/issues/42) |

> Construir el módulo matemático que ingerirá los CSVs de results/logs/ y calculará las métricas institucionales de rendimiento y riesgo de la estrategia algorítmica frente al mercado. - [x] Crear el sc…

#### ✅ \#43 — Dashboard de Visualización (Plotting)

| Campo   | Valor |
|---------|-------|
| Estado  | ✅ Cerrada |
| Labels  | `enhancement` |
| Creada  | 12/07/2026 |
| Cerrada | 27/07/2026 |
| URL     | [https://github.com/Srebort/Intelligent_Portfolio_Management/issues/43](https://github.com/Srebort/Intelligent_Portfolio_Management/issues/43) |

> Implementar el renderizado visual con librerías de gráficos para mostrar de un vistazo la superioridad (o inferioridad) del bot frente a dejar el dinero quieto en el S&P 500. Se guardarán como imágene…

---

## Sprint 8: Redacción de la memoria y corrección de fallos

| Campo            | Valor |
|------------------|-------|
| **Estado**       | 🔵 Abierta |
| **Fecha límite** | 06/08/2026 |
| **Cerrada el**   | — |
| **Progreso**     | 0/1 issues completadas (0%) |

### Descripción

Redactar los capítulos que faltan de la memoria y corregir fallos que surjan a lo largo de la redacción ya sean de código o en la memoria.

**Progreso:** `[░░░░░░░░░░░░░░░░░░░░]` 0%

### Issues (1)

#### 🔵 \#48 — Redactar Memoria

| Campo   | Valor |
|---------|-------|
| Estado  | 🔵 Abierta |
| Labels  | `documentation` |
| Creada  | 27/07/2026 |
| Cerrada | — |
| URL     | [https://github.com/Srebort/Intelligent_Portfolio_Management/issues/48](https://github.com/Srebort/Intelligent_Portfolio_Management/issues/48) |

> Se procederá a la redacción de la memoria para su entrega: - [ ] Redactar los capitulos que faltan

---
