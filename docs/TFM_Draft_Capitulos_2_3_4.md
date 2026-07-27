# BORRADOR DE REDACCIÓN — TFM: Intelligent Portfolio Management
# Capítulos: Estado del Arte, Análisis del Problema, Arquitectura e Ingeniería de Características
# Autor: Sergio Rebollo Puerto | MUIINF — Universitat Politècnica de València
# Fecha: Junio 2026
#
# INSTRUCCIONES: Copiar cada sección en la plantilla Word (plantillaTFM_MUIINF.docx)
# respetando los estilos de título definidos en la plantilla.
# ==================================================================================


# ===========================================================================
# CAPÍTULO 1: INTRODUCCIÓN
# ===========================================================================

Este capítulo presenta la visión global del Trabajo Fin de Máster (TFM), estableciendo los motivos que impulsan su desarrollo, los objetivos que se persiguen, la metodología de trabajo adoptada y el impacto esperado de la solución propuesta.

## 1.1 Motivación

En la última década, los mercados financieros han experimentado una transformación estructural hacia la automatización. Mientras que los inversores institucionales utilizan complejos modelos cuantitativos y algoritmos de alta frecuencia para gestionar su exposición al riesgo, el inversor particular sigue dependiendo mayoritariamente de decisiones discrecionales, sesgos emocionales y análisis técnico manual. 

La motivación principal de este proyecto radica en la necesidad de cerrar esta brecha tecnológica. Se pretende demostrar que, mediante la integración de Análisis Multi-Timeframe (MTF) riguroso, algoritmos de detección de *Price Action* y modelos de Machine Learning (como XGBoost o Random Forest), es posible construir un sistema de gestión cuantitativa de carteras autónomo, robusto y con esperanza matemática positiva, accesible sin requerir la infraestructura de un fondo de inversión.

A nivel personal y académico, este TFM representa la oportunidad de aplicar de manera práctica e integrada los conocimientos adquiridos durante el Máster Universitario en Ingeniería Informática (MUIINF), abarcando disciplinas complejas como la ingeniería de datos, el aprendizaje automático, el diseño de arquitecturas de software y la teoría financiera de gestión de riesgos.

## 1.2 Objetivos

El propósito fundamental de este trabajo es diseñar, implementar y validar un sistema algorítmico autónomo capaz de gestionar una cartera de renta variable estadounidense, maximizando el rendimiento ajustado al riesgo mediante el uso de inteligencia artificial.

De este propósito se derivan los siguientes objetivos específicos:

1.  **Desarrollar una infraestructura de datos robusta:** Construir un pipeline capaz de extraer, limpiar y fusionar datos de múltiples temporalidades (intradiario, diario y semanal) evitando la filtración de información futura (*lookahead bias*).
2.  **Codificar reglas expertas de análisis técnico:** Traducir patrones visuales complejos (como fractales de Bill Williams, divergencias de RSI y *wick reclaims*) en un sistema algorítmico jerárquico de evaluación de señales (Tiers).
3.  **Implementar un filtro predictivo de Machine Learning:** Entrenar modelos basados en árboles (Random Forest y XGBoost) para clasificar y descartar señales de baja probabilidad, optimizando la precisión de las entradas al mercado.
4.  **Diseñar un simulador de gestión de riesgos:** Construir un módulo de gestión de cartera que implemente estrategias de dimensionamiento de posición (*position sizing*) basadas en riesgo fijo fraccional, controlando estrictamente la exposición máxima del capital.
5.  **Evaluar empíricamente el rendimiento:** Realizar un *backtest* exhaustivo de la estrategia frente a un *benchmark* pasivo (como el índice S&P 500), utilizando métricas financieras estándar como el Ratio de Sharpe y el Maximum Drawdown.

## 1.3 Impacto Esperado

El resultado de este trabajo supondrá el desarrollo de un motor cuantitativo integral (*Intelligent Portfolio Management*) capaz de operar de manera autónoma. Los beneficiarios directos de esta tecnología podrían ser inversores particulares avanzados o pequeñas firmas de gestión de capital que busquen sistematizar su operativa para eliminar el estrés psicológico del trading manual.

Desde una perspectiva académica, este TFM aporta un marco de validación estricto para modelos de Machine Learning aplicados a finanzas, demostrando cómo el etiquetado basado en reglas de *backtesting* estricto resuelve el problema clásico de los *datasets* financieros ruidosos y desbalanceados.

## 1.4 Metodología

El desarrollo del proyecto se ha regido por una adaptación de metodologías ágiles, estructurando el trabajo en *Sprints* iterativos e incrementales. Cada iteración ha estado orientada a entregar un componente funcional de la arquitectura:

*   **Sprints de Datos e Ingeniería de Características:** Centrados en la conexión a APIs (Tiingo, FRED) y la extracción matemática de indicadores.
*   **Sprints de Modelado y Etiquetado:** Dedicados a la construcción del *Backtester* y al entrenamiento de los modelos predictivos de Machine Learning.
*   **Sprints de Simulación Financiera:** Enfocados en la lógica del Agente Gestor de Cartera y las reglas de riesgo (*Risk Management*).

Para la implementación tecnológica se ha empleado Python como lenguaje base, utilizando librerías estándar de la industria de datos como `pandas`, `numpy`, `scikit-learn` y `xgboost`, garantizando así la mantenibilidad y escalabilidad del código.

## 1.5 Estructura de la Memoria

El presente documento se estructura de la siguiente manera:
*   El **Capítulo 2** expone el Estado del Arte, revisando la literatura sobre trading algorítmico y Machine Learning, e identificando el espacio de innovación del trabajo.
*   El **Capítulo 3** realiza el Análisis del Problema, definiendo el universo de activos, las fuentes de datos y las consideraciones legales y de seguridad.
*   El **Capítulo 4** detalla la Solución Propuesta, abarcando desde la arquitectura del pipeline de datos y la codificación de patrones técnicos, hasta el diseño y evaluación del filtro predictivo de inteligencia artificial.
*   El **Capítulo 5** presentará la Implantación y los Resultados, mostrando el rendimiento del sistema en simulaciones históricas (*backtesting*).
*   El **Capítulo 6** recoge las Conclusiones finales, la relación del trabajo con los estudios cursados y propone futuras líneas de investigación.


# ===========================================================================
# CAPÍTULO 2: ESTADO DEL ARTE
# ===========================================================================

## 2.1 Trading Algorítmico y Sistemas de Gestión Cuantitativa de Carteras

El trading algorítmico, definido como la ejecución automatizada de órdenes de compraventa
mediante reglas computacionales predeterminadas, ha experimentado un crecimiento exponencial
desde la década de los noventa. Según datos de la U.S. Securities and Exchange Commission (SEC),
los algoritmos generan actualmente más del 70% del volumen negociado en las principales bolsas
americanas [1]. Esta automatización, históricamente reservada a fondos de inversión
institucionales con acceso a infraestructuras tecnológicas de alto coste, ha comenzado a
democratizarse gracias a la proliferación de APIs de datos de mercado y librerías de machine
learning de código abierto.

En el ámbito académico, el trabajo fundacional de Markowitz (1952) [2] sobre la Teoría Moderna
de Carteras estableció los principios matemáticos de la diversificación óptima. Posteriormente,
Fama y French (1993) [3] demostraron que los mercados financieros no son perfectamente eficientes
y que determinados factores sistemáticos (valor, tamaño, momentum) permiten obtener rentabilidades
ajustadas al riesgo superiores al mercado. Este hallazgo abrió la puerta al desarrollo de
estrategias de inversión basadas en reglas cuantitativas.

Los sistemas modernos de gestión algorítmica de carteras pueden clasificarse en dos grandes
familias: (i) los sistemas basados exclusivamente en reglas de análisis técnico, donde las
señales de compraventa se generan mediante la combinación de indicadores clásicos como medias
móviles, osciladores de momentum y patrones de velas; y (ii) los sistemas híbridos o de
aprendizaje automático, donde un modelo estadístico aprende a identificar patrones rentables
a partir de datos históricos etiquetados.

El presente TFM se sitúa en esta segunda familia, combinando un sistema de reglas expertas
estructuradas jerárquicamente (los Tiers de probabilidad) con un clasificador de Machine
Learning (XGBoost) entrenado sobre los resultados reales del backtester.


## 2.2 Análisis Multi-Timeframe (MTF): Justificación y Ventajas

El Análisis Multi-Timeframe (MTF) es una metodología ampliamente utilizada por traders
institucionales que consiste en tomar decisiones de trading en una temporalidad operativa
(corto plazo) únicamente cuando la tendencia en temporalidades superiores (largo plazo) es
favorable a la dirección de la operación. Elder (1993) [4], en su obra seminal "Trading for a
Living", formalizó esta idea bajo la denominación "Sistema de la Triple Pantalla", donde
se utilizan tres temporalidades con relación aproximada 1:5 entre sí.

La justificación matemática del MTF descansa en la teoría de ondas de mercado: los precios
se mueven en tendencias anidadas de distintos órdenes de magnitud. Una vela de 4 horas alcista
no es contradictoria con una tendencia semanal bajista; sin embargo, operar a favor de la
tendencia de mayor orden estadísticamente aumenta la tasa de acierto y mejora el ratio
riesgo-beneficio de las operaciones [4].

Trabajos como el de Appel (2005) [5] y Murphy (1999) [6] documentaron empíricamente que la
confirmación de la tendencia en temporalidades superiores antes de ejecutar una entrada en
la temporalidad operativa reduce la frecuencia de señales falsas en un 30-40% respecto a
sistemas que operan en una única temporalidad. Esta reducción de señales falsas es
especialmente relevante en estrategias Long-Only como la desarrollada en este trabajo, donde
cada señal incorrecta implica un coste de oportunidad y un capital inmovilizado.

### 2.2.1 El Problema del Lookahead Bias en Sistemas MTF

La principal amenaza en la implementación de un sistema MTF es el denominado lookahead bias
o fuga de información temporal. Este sesgo ocurre cuando el modelo de aprendizaje o el
sistema de señales utiliza, implícitamente, información del futuro para generar señales en el
pasado, produciendo resultados de backtesting artificialmente optimistas que no se replicarán
en trading real.

En el contexto de los sistemas MTF, el lookahead bias aparece frecuentemente cuando se
propagan valores de temporalidades superiores (ej. el cierre diario del lunes 10 de enero)
a todas las velas de 4H del mismo día, incluidas las anteriores al cierre de mercado. Este
error implica que una vela de 4H de las 8:00h "conoce" el cierre de mercado de las 16:00h.

El sistema desarrollado en este TFM resuelve este problema mediante el uso de
`pandas.merge_asof` con `direction='backward'`: la propagación de valores de temporalidades
superiores a las velas de 4H utiliza exclusivamente el último valor disponible en el momento
de cada vela, nunca valores futuros. Esta garantía se verifica automáticamente en cada
ejecución del pipeline, registrando en el log del sistema el rango temporal de los datos
propagados.

Tabla 2.1: Comparativa entre sistemas de análisis de mercado
| Característica | Sistema Mono-Timeframe | Sistema MTF (este TFM) |
|---|---|---|
| Señales falsas | Alta frecuencia | Reducidas por filtro multi-nivel |
| Lookahead bias | Riesgo moderado | Eliminado mediante merge_asof backward |
| Complejidad | Baja | Media-Alta |
| Adaptación al contexto macro | No | Sí (filtro 1D y 1W) |
| Robustez a cambios de régimen | Baja | Alta (tendencia macro como permiso) |


## 2.3 Machine Learning en la Predicción de Señales de Trading

La aplicación de técnicas de aprendizaje automático a la predicción de movimientos bursátiles
cuenta con una extensa literatura. Los primeros trabajos, como los de Kimoto et al. (1990) [7]
y Refenes et al. (1994) [8], utilizaron redes neuronales feed-forward para predecir el índice
Nikkei y el S&P 500, respectivamente, con resultados prometedores pero difícilmente
replicables por el problema del sobreajuste (overfitting).

El advenimiento de los métodos de ensamblado (ensemble methods) supuso un avance significativo
en la robustez de los modelos de predicción financiera. En particular:

- **Random Forest** (Breiman, 2001) [9]: Genera múltiples árboles de decisión sobre submuestras
  aleatorias del dataset (técnica de bagging) y promedia sus predicciones, reduciendo la
  varianza sin incrementar el sesgo. Su robustez a variables irrelevantes y su capacidad
  para manejar interacciones no lineales lo convierten en una línea base sólida para problemas
  de clasificación financiera.

- **XGBoost** (Chen y Guestrin, 2016) [10]: Implementación optimizada del Gradient Boosting que
  construye árboles de forma secuencial, donde cada árbol corrige los errores del anterior
  (técnica de boosting). Ha demostrado resultados estado del arte en múltiples competencias
  de predicción financiera (Kaggle) y en estudios académicos sobre clasificación de señales
  de mercado [10].

La recomendación específica del tribunal de incluir una comparativa entre Regresión Logística,
Random Forest y XGBoost responde a la necesidad académica de validar que la complejidad
adicional del modelo boosted aporta mejoras estadísticamente significativas sobre el baseline
estadístico clásico.


## 2.4 Crítica al Estado del Arte y Contribución Original

La mayor parte de los trabajos de Machine Learning aplicado a trading presentan una limitación
metodológica fundamental: entrenan los modelos directamente sobre los cambios de precio
(clasificando si el precio sube o baja), sin incorporar información sobre la calidad del
setup de entrada. Este enfoque ignora el contexto técnico en el que se produce cada movimiento,
generando datasets desbalanceados y con baja señal-ruido.

El sistema propuesto en este TFM adopta un enfoque alternativo: el clasificador
XGBoost no predice si el precio subirá genéricamente, sino si una señal técnica específica
(ya filtrada por el sistema de Tiers) resultará rentable dadas las condiciones de gestión de
riesgo definidas (Stop Loss y Take Profit). Esta formulación, denominada "confirmación de
señal" en la literatura especializada [11], presenta varias ventajas:

1. El dataset de entrenamiento es balanceado naturalmente (las señales ganadoras y perdedoras
   tienen frecuencias comparables).
2. Las features (variables de entrada) están directamente relacionadas con la calidad del
   setup, no con el ruido genérico del mercado.
3. El backtester actúa como generador de etiquetas matemáticamente rigurosas, basadas en
   resultados reales de precio y no en predicciones subjetivas.


# ===========================================================================
# CAPÍTULO 3: ANÁLISIS DEL PROBLEMA Y DATOS
# ===========================================================================

## 3.1 Definición del Problema

La gestión cuantitativa de carteras de renta variable enfrenta tres desafíos fundamentales
que este TFM aborda de forma integrada:

**Desafío 1 — Selección de señales de calidad:** Los mercados financieros generan
continuamente patrones técnicos, pero la mayoría de ellos son ruido aleatorio sin valor
predictivo. Se necesita un sistema de filtrado capaz de distinguir las oportunidades de alta
probabilidad del ruido.

**Desafío 2 — Gestión dinámica del riesgo:** Asignar el mismo capital a todas las operaciones,
independientemente de su calidad, es subóptimo. Se requiere un mecanismo de dimensionamiento
de posición proporcional a la confianza en la señal.

**Desafío 3 — Control de exposición global:** En una cartera con múltiples activos simultáneos,
la correlación entre posiciones puede concentrar el riesgo de forma inadvertida. Es necesario
un gestor de cartera que limite la exposición total y priorice las mejores oportunidades.


## 3.2 Universo de Activos

El universo de inversión del sistema comprende más de 50 activos de renta variable estadounidense,
organizados en los siguientes sectores:

Tabla 3.1: Universo de activos del sistema (45 acciones evaluadas)
| Sector | Activos (Tickers) |
|---|---|
| **Tecnología y Semiconductores** | AAPL, MSFT, NVDA, GOOGL, META, AMD |
| **Ciberseguridad y Nube** | CRWD, PANW, SNOW, PLTR |
| **Servicios Financieros** | JPM, BAC, V |
| **Salud y Biotecnología** | JNJ, UNH, LLY, ABBV, ISRG |
| **Consumo Discrecional y Viajes** | AMZN, HD, MCD, TSLA, F, NFLX, ABNB, UBER, DAL |
| **Consumo Defensivo** | WMT, PG, KO |
| **Energía e Industria** | XOM, CVX, CAT, UNP, GE |
| **Aeroespacial y Defensa** | LMT, RTX, BA |
| **Materiales Básicos** | FCX, NEM, LIN |
| **Bienes Raíces (REITs) y Utilities** | O, PLD, AMT, NEE |

La selección de los 45 activos de renta variable americana como universo principal no es arbitraria ni puramente retrospectiva. Responde a criterios predefinidos y objetivos establecidos antes de iniciar el ciclo de simulación:
1.  **Capitalización y Liquidez Extrema:** Se seleccionaron exclusivamente activos pertenecientes al S&P 500 y Nasdaq-100 con un volumen medio de negociación diario superior a 10 millones de acciones. Este umbral es crítico para garantizar la ejecutabilidad real de las órdenes institucionales sin generar un impacto de mercado adverso (*slippage* severo).
2.  **Disponibilidad Histórica Ininterrumpida:** Activos que mantengan un historial de cotización robusto y continuo desde al menos el año 2010, permitiendo una ventana temporal de entrenamiento estadísticamente significativa para capturar múltiples regímenes macroeconómicos.
3.  **Representatividad Sectorial Cruzada:** Se impuso la inclusión deliberada de activos pertenecientes a 10 macro-sectores económicos distintos. Esto asegura que el modelo de Machine Learning aprenda dinámicas universales de la estructura del precio (*Price Action*), evitando el sobreajuste a la idiosincrasia direccional de un único sector (ej. el sesgo alcista secular del sector tecnológico).

*Limitaciones Metodológicas:* Al acotar el universo utilizando activos consolidados (megacapitalización del S&P 500 y Nasdaq), se mitigan los riesgos sistémicos, pero se reconoce la existencia potencial de un sesgo de supervivencia (*Survivorship Bias*). Dado que la selección se basó en empresas listadas y líderes a fecha de 2024, se filtraron inadvertidamente aquellas entidades que quebraron o fueron deslistadas en la última década. Para un despliegue en producción exhaustivo, el universo debe necesariamente abarcar activos purgados (*delisted*) para cimentar la validez absoluta del backtesting.


## 3.3 Fuentes de Datos

### 3.3.1 Datos de Precio Intradiario: Tiingo IEX

Los datos de precio OHLCV (Open, High, Low, Close, Volume) se obtienen del endpoint IEX de
la API de Tiingo (https://api.tiingo.com), que proporciona series históricas con granularidad
desde 1 minuto hasta 1 semana.

Para este TFM se descargan tres resoluciones temporales:
- **4 Horas (4H):** Temporalidad operativa principal, desde 2018 hasta 2026.
- **1 Día (1D):** Temporalidad de tendencia de medio plazo, desde 2018 hasta 2026.
- **1 Semana (1W):** Temporalidad de tendencia macro, desde 2018 hasta 2026.

El módulo `tiingo_loader.py` implementa la clase `TiingoLoader`, que gestiona la autenticación
mediante clave API almacenada en un fichero `.env` (nunca expuesta en el repositorio), aplica
un control de tasa de solicitudes (rate limiting) para respetar los límites de la API, y
exporta los datos descargados a ficheros CSV en `data/raw/` para eliminar la dependencia
de internet en ejecuciones posteriores.

### 3.3.2 El Problema del Precalentamiento de la Media Móvil de 200 Períodos

La Media Móvil Simple (SMA) de 200 períodos semanales (SMA_200_1W) es el indicador de
tendencia macro más empleado por inversores institucionales. Sin embargo, su cálculo correcto
requiere exactamente 200 velas semanales previas (200 semanas ≈ 4 años) antes de producir
un valor válido.

Si el histórico disponible comenzara en enero de 2018, las primeras 200 semanas (hasta
aproximadamente diciembre de 2021) generarían valores NaN en la SMA_200_1W, invalidando el
filtro de tendencia macro del sistema y descartando todas las señales de ese período.

Para resolver este problema, el módulo `tiingo_loader.py` incorpora la función
`download_daily_historical_ticker`, que descarga datos End-of-Day (EOD) desde el 1 de enero
de 2010 para todos los activos del universo. Estos datos de "precalentamiento" no se utilizan
directamente en el entrenamiento del modelo (cuyo período comienza en 2018), sino
exclusivamente para garantizar que la SMA_200_1W esté completamente calculada cuando se
inicia el período de operación.

La figura 3.1 ilustra este efecto: con datos desde 2010, la SMA_200_1W está disponible desde
enero de 2014, cubriendo con holgura el período operativo desde 2018.

### 3.3.3 Datos Macroeconómicos: FRED

El sistema incorpora variables macroeconómicas procedentes del Federal Reserve Economic Data
(FRED), base de datos pública mantenida por el Banco de la Reserva Federal de St. Louis.
Las series descargadas son:

Tabla 3.2: Variables macroeconómicas del sistema
| Serie FRED | Variable | Frecuencia | Interpretación |
|---|---|---|---|
| FEDFUNDS | Tipo de interés Fed | Mensual | Coste del dinero (hawkish vs dovish) |
| CPIAUCSL | Índice de Precios al Consumo | Mensual | Presión inflacionaria |
| UNRATE | Tasa de desempleo | Mensual | Salud del mercado laboral |
| T10Y2Y | Spread bonos 10Y-2Y | Diaria | Indicador adelantado de recesión |
| VIXCLS | VIX (volatility index) | Diaria | Índice del miedo del mercado |

El módulo `fred_loader.py` descarga estas series mediante la API pública de FRED y las
une al dataset principal mediante un merge temporal con relleno hacia adelante (forward fill),
asegurando que cada vela de 4H lleve consigo el último valor conocido de cada variable
macroeconómica sin filtración de información futura.


## 3.4 Pipeline de Ingesta y Transformación de Datos (ETL)

El pipeline de datos sigue una arquitectura modular de tres etapas: Extracción (E),
Transformación (T) y Carga (L), implementada mediante los siguientes módulos:

```
EXTRACCIÓN          TRANSFORMACIÓN              CARGA
tiingo_loader.py → mtf_builder.py → dataset_cleaner.py → V_ML_READY.csv
fred_loader.py  ↗
```

### 3.4.1 Construcción del Dataset Multi-Timeframe: `mtf_builder.py`

El módulo `MTFBuilder` es el núcleo de la transformación de datos. Su función es unificar
las tres temporalidades en un único DataFrame de velas de 4H, preservando la integridad
temporal mediante la técnica de merge asof.

El proceso de construcción sigue los siguientes pasos:

1. **Carga de datos:** Se leen los CSVs de 4H, 1D y 1W previamente descargados por
   `tiingo_loader.py`, convirtiendo el índice temporal a formato UTC para garantizar la
   coherencia entre temporalidades.

2. **Cálculo de indicadores por temporalidad:** Sobre cada DataFrame se calculan los
   indicadores técnicos correspondientes mediante `add_all_features()` (módulo `technical.py`).
   Los indicadores de 1D y 1W reciben el sufijo `_1D` y `_1W` respectivamente para
   distinguirlos de sus equivalentes en 4H.

3. **Propagación hacia adelante sin lookahead bias:** La fusión de temporalidades se realiza
   mediante `pd.merge_asof(direction='backward')`, que para cada vela de 4H busca el último
   valor disponible en 1D y 1W cuyo timestamp sea estrictamente anterior o igual al de la
   vela de 4H. Esto garantiza que, por ejemplo, una vela de las 08:00h del lunes solo tiene
   acceso al cierre del viernes anterior en la temporalidad diaria, no al cierre del lunes.

La salida del `MTFBuilder` es un DataFrame con 35+ columnas que representa la "fotografía
completa" del mercado en cada vela de 4H, incluyendo el contexto de largo plazo.

### 3.4.2 Limpieza y Normalización: `dataset_cleaner.py`

El módulo `DatasetCleaner` aplica el pipeline de limpieza final antes de que los datos sean
consumidos por el modelo de Machine Learning:

1. **Eliminación de NaNs:** Las primeras velas del dataset contienen valores NaN en los
   indicadores de ventanas largas (SMA_200, RSI_14 con precalentamiento insuficiente). Estas
   filas se eliminan, preservando únicamente el período operativo desde 2018.

2. **Neutralización de infinitos:** El cálculo de ratios y distancias porcentuales puede
   producir valores infinitos en casos excepcionales (divisiones por precios nulos en activos
   sin liquidez). Estos valores se reemplazan por el máximo finito de la columna para evitar
   errores de entrenamiento sin eliminar la fila completa.

3. **Normalización selectiva con StandardScaler:** Se aplica normalización Z-score
   (media = 0, desviación típica = 1) exclusivamente sobre los indicadores técnicos continuos
   (RSI, ATR, distancias, pendientes). Los precios OHLCV originales y las variables binarias
   (fractales, patrones de velas) quedan sin normalizar, ya que su significado semántico no
   debe alterarse. La fórmula de normalización es:

   z = (x - μ) / σ

   donde μ es la media del indicador en el período de entrenamiento y σ su desviación típica.

El dataset resultante, exportado como `data/processed/V_ML_READY.csv`, contiene 47 columnas
de características listas para entrenar el clasificador.


# ===========================================================================
# CAPÍTULO 4: ARQUITECTURA DEL SISTEMA E INGENIERÍA DE CARACTERÍSTICAS
# ===========================================================================

## 4.1 Visión General de la Arquitectura

El sistema desarrollado sigue una arquitectura en capas, donde cada capa añade un nivel de
abstracción sobre los datos crudos hasta llegar a la decisión de inversión:

```
Capa 1 — Datos Brutos:     OHLCV 4H/1D/1W + Variables Macro (FRED)
Capa 2 — Features:         Indicadores técnicos + Price Action + MTF
Capa 3 — Señales:          Sistema de Tiers (A*, A, B, C) — TierEvaluator
Capa 4 — Filtro ML:        Clasificador XGBoost — confirmación de señal
Capa 5 — Gestión de riesgo: Stop Loss, Take Profit, Position Sizing — RiskManager
Capa 6 — Cartera:          Ranking, liquidez, exposición máxima — PortfolioAgent
```

## 4.2 Indicadores Técnicos: `technical.py`

El módulo `technical.py` implementa de forma completamente vectorizada (sin bucles for)
todos los indicadores técnicos utilizados como características de entrada del modelo de
Machine Learning. La vectorización mediante operaciones de Pandas y NumPy garantiza la
escalabilidad a millones de velas sin degradación del rendimiento.

Todos los indicadores cumplen la garantía anti-lookahead bias: utilizan exclusivamente
operaciones retrospectivas (rolling backward-looking), es decir, cada valor del indicador
se calcula usando únicamente la información disponible hasta el instante de esa vela.

### 4.2.1 RSI — Relative Strength Index

El RSI, desarrollado por J. Welles Wilder (1978), mide la velocidad y magnitud de los
movimientos de precio recientes, oscilando entre 0 y 100.

Fórmula:
  Δ_t       = close_t - close_{t-1}
  Gain_t    = max(Δ_t, 0)
  Loss_t    = max(-Δ_t, 0)
  RS        = EWM(Gain, α=1/14) / EWM(Loss, α=1/14)
  RSI       = 100 - 100 / (1 + RS)

donde EWM denota la media exponencialmente ponderada con factor de suavizado α = 1/(periodo-1).

Valores por encima de 70 indican sobrecompra (el activo puede revertir a la baja);
valores por debajo de 30 indican sobreventa (posible oportunidad de compra). En el sistema,
el RSI se calcula con período 14 en las temporalidades 4H y 1D.

### 4.2.2 ATR y NATR — Average True Range

El ATR, también propuesto por Wilder (1978), mide la volatilidad real del mercado incorporando
los gaps entre sesiones (saltos de precio de un cierre al apertura del día siguiente):

  True Range_t = max(
    high_t - low_t,                  ← Rango intradía
    |high_t - close_{t-1}|,          ← Gap alcista respecto al cierre anterior
    |low_t  - close_{t-1}|           ← Gap bajista respecto al cierre anterior
  )
  ATR_t = EWM(True Range, período=14)

El ATR tiene dos usos fundamentales en el sistema: (i) como feature para el clasificador ML,
reflejando el régimen de volatilidad actual; y (ii) como parámetro del cálculo del Stop Loss,
donde sirve de "colchón" para evitar que el ruido normal del mercado active la orden de salida.

El NATR (Normalized ATR) es la versión porcentual del ATR, calculada como:
  NATR = (ATR / close) × 100

Esta normalización permite comparar la volatilidad entre activos de distintos rangos de precio.

### 4.2.3 Medias Móviles: SMA y EMA

El sistema calcula medias móviles simples (SMA) y exponenciales (EMA) en cinco ventanas
estándar: 9, 21, 50, 100 y 200 períodos.

  SMA_n = (1/n) × Σ(close_{t-i}, i=0 a n-1)

  EMA_n = close_t × α + EMA_{n,t-1} × (1 - α),   α = 2/(n+1)

Las distancias relativas al precio se calculan como:
  dist_SMA_n = (close - SMA_n) / SMA_n × 100

y las pendientes como:
  slope_SMA_n = (SMA_n,t - SMA_n,t-5) / SMA_n,t-5 × 100

La pendiente de la SMA_200_1W (pendiente semanal) es la feature más importante del filtro
de tendencia macro: si es positiva, el mercado está en tendencia alcista de largo plazo.

### 4.2.4 Compresión de Volatilidad: ATR Squeeze

El ratio de compresión de volatilidad detecta períodos de baja volatilidad que históricamente
preceden a movimientos bruscos de precio (breakouts):

  atr_squeeze = ATR_14 / SMA(ATR_14, ventana=20)

Valores inferiores a 1.0 indican que la volatilidad actual está por debajo de su media de
los últimos 20 períodos (compresión). En el contexto del sistema, señales generadas durante
períodos de compresión tienen mayor probabilidad de producir movimientos explosivos.

### 4.2.5 Divergencias Alcistas del RSI

La divergencia alcista es uno de los patrones de mayor fiabilidad en el análisis técnico.
Se produce cuando el precio forma un mínimo más bajo (lower low) pero el RSI forma un mínimo
más alto (higher low), indicando que el momentum bajista se está agotando.

Algoritmo de detección (vectorizado):
  1. Identificar mínimos locales de precio y de RSI en una ventana de 20 períodos
  2. Comparar el mínimo actual de precio con el anterior: si price_low_t < price_low_{t-prev} → condición 1
  3. Comparar el mínimo actual de RSI con el anterior: si rsi_low_t > rsi_low_{t-prev} → condición 2
  4. is_bullish_divergence = condición_1 AND condición_2

Esta feature es condición necesaria para el Tier A (máxima probabilidad) del sistema.


## 4.3 Price Action Algorítmico: `patterns.py`

El módulo `patterns.py` traduce conceptos visuales del análisis de Price Action en
características matemáticas binarias (1/0) que el modelo de ML puede procesar.

### 4.3.1 Fractales de Bill Williams

Los fractales de Bill Williams identifican mínimos locales (soporte algorítmico) y máximos
locales (resistencia algorítmica) de forma objetiva:

- **Fractal de soporte (Down Fractal):** Una vela cuyo mínimo es estrictamente inferior al
  mínimo de las 2 velas anteriores y las 2 velas posteriores.
  is_support_fractal_t ≡ (low_{t-2} > low_{t-2}, low_{t-1} > low_{t-2}, low_{t+1} > low_{t-2}, low_{t+2} > low_{t-2})

- **Fractal de resistencia (Up Fractal):** Análogo para los máximos.

Garantía anti-lookahead bias: dado que el fractal requiere las 2 velas POSTERIORES para su
confirmación, la señal se emite en la vela t (la actual), no en t-2 (el centro del patrón).
Esto modela fielmente lo que un trader vería en tiempo real: la confirmación solo llega cuando
el mercado ha formado dos velas más bajas que el mínimo central.

### 4.3.2 Patrones de Velas Japonesas

**Martillo (Hammer):** Indica fuerte rechazo del precio hacia abajo. Condiciones:
  - Mecha inferior ≥ 2 × cuerpo del cuerpo
  - Mecha superior ≤ 10% del rango total

**Martillo Invertido (Inverted Hammer):** Indica fuerte rechazo al alza:
  - Mecha superior ≥ 2 × cuerpo
  - Mecha inferior ≤ 10% del rango total

Ambos patrones se calculan vectorizadamente sin ninguna condicional iterativa.

### 4.3.3 Wick Reclaims (Barridos de Liquidez)

El Bullish Wick Reclaim identifica velas que barrieron stops por debajo de un soporte
y luego recuperaron ese nivel dentro de la misma vela, dejando una mecha inferior larga:

  ratio_mecha_inferior = mecha_inferior / rango_total
  is_bullish_wick_reclaim = (ratio_mecha_inferior ≥ 0.60)

Este patrón es fundamental en el Tier A (junto con la divergencia RSI), ya que representa
un barrido de liquidez seguido de absorción institucional.

### 4.3.4 Niveles de Fibonacci en Ventana Deslizante

Los niveles de Fibonacci se calculan sobre una ventana deslizante de 50 períodos,
identificando el máximo (H) y el mínimo (L) del período:

  Rango = H - L
  Fib_38.2% = H - 0.382 × Rango  (retroceso del 38.2%)
  Fib_61.8% = H - 0.618 × Rango  (retroceso del 61.8%)

El uso de una ventana deslizante en lugar de niveles fijos garantiza que los niveles de
Fibonacci se actualicen con el movimiento del mercado, capturando la zona de valor relevante
en cada momento sin filtración de información futura.

Las features resultantes son las distancias relativas del precio actual a cada nivel:
  dist_fib_retr_618 = (close - Fib_61.8%) / close × 100

Un valor cercano a 0 indica que el precio está en ese nivel de Fibonacci, lo que es una
condición de entrada en los Tiers A y A*.


## 4.4 Sistema de Evaluación de Señales: Evolución y Optimización Empírica

El `TierEvaluator` es el motor algorítmico encargado de clasificar cada vela del dataset en una de las tres estrategias principales (Tier A, Tier B, Tier C). Durante la fase de desarrollo, la arquitectura original fue diseñada basándose en heurísticas tradicionales de trading discrecional. Sin embargo, el backtesting exhaustivo sobre más de 100,000 velas históricas demostró que las reglas humanas, al codificarse de manera rígida, resultaban contraproducentes. 

A continuación, se detalla el proceso empírico de optimización que transformó un sistema teóricamente conservador (pero perdedor) en una arquitectura robusta, generando un *dataset* final equilibrado de 3,318 señales limpias para el entrenamiento de la Inteligencia Artificial.

### 4.4.1 Eliminación de la Dependencia Multi-Timeframe (MTF) en todos los Tiers

**El paradigma original:** La teoría clásica dicta operar siempre a favor de la tendencia macro. Por ello, el sistema original exigía que, para tomar cualquier operación (ya fuera Tier A, B o C), el precio debía estar por encima de la SMA 200 no solo en la gráfica operativa (4 Horas), sino también en las gráficas de 1 Día y 1 Semana.

**El problema empírico:** El backtesting demostró que las medias móviles semanales y diarias tienen demasiado *lag* (retraso). Para cuando el precio lograba cruzar la SMA 200 en 1 Semana, el activo ya había subido agresivamente durante semanas. Este filtro obligaba al sistema a comprar en la cima de los movimientos extendidos, justo antes de las correcciones, mientras filtraba los pivotes y rebotes tempranos más rentables.

**La solución:** Se purgaron los filtros macro de forma universal. Ahora, los **Tiers A, B y C** evalúan la tendencia puramente en su entorno operativo, exigiendo únicamente:
  `cond_4h = close_4H > SMA_200_4H`
Esta simplificación permitió al sistema reaccionar de forma ágil a los cambios de momentum en las tres estrategias.

### 4.4.2 Evolución del Tier A: De la Perfección Teórica al Pragmatismo del *Shakeout*

El Tier A fue diseñado para operar *pullbacks* (retrocesos) hacia la media institucional (SMA 200). En su concepción inicial, estaba fuertemente restringido por múltiples candados lógicos.

**Las restricciones eliminadas:**
1. **El nivel exacto de Fibonacci:** Se exigía que el rebote coincidiera milimétricamente con el nivel de retroceso del 61.8% de Fibonacci. Se descubrió empíricamente que esta regla asfixiaba al sistema, descartando rebotes altamente rentables en los niveles 50% o 78.6%. La condición fue eliminada.
2. **La Regla 6/20 (Valid Pullback):** Esta regla dictaba que, de las últimas 20 velas, un máximo de 6 podían haber cerrado por debajo de la SMA 200. Su objetivo era evitar comprar activos cuya tendencia estuviera "rota". Contraintuitivamente, el backtesting demostró que los peores rompimientos temporales (aquellos donde el precio pasa 8 o 10 velas "hundido" bajo la media) son en realidad **cacerías de liquidez (*shakeouts*)** ejecutadas por el dinero institucional. Al eliminar la regla 6/20, permitimos al algoritmo comprar pánico extremo, lo que casi duplicó el beneficio neto (P&L) y redujo el *Drawdown* máximo del -18.7% al -16.6%.

**El problema del *Overtrading* y la solución del Cooldown:**
Al eliminar las restricciones anteriores, el Tier A se volvió muy reactivo. Cuando el precio consolidaba lateralmente sobre la SMA 200, el sistema abría decenas de operaciones simultáneas en la misma zona, asumiendo un riesgo catastrófico (generando un *Drawdown* inaceptable del -109%).
Para mitigarlo, se programó un **Filtro de Cooldown (Enfriamiento) de 12 velas**. Una vez ejecutada una entrada de Tier A, el sistema bloquea cualquier nueva señal en ese mismo activo durante 2 días operativos. Esta simple regla espacial distribuyó el riesgo y estabilizó la curva de capital.

La ecuación final del Tier A prioriza el contacto crudo con la media institucional:
  `Tier_A = cond_4h AND |dist_SMA_200| < 5% AND not_escaped AND cooldown_cleared`

### 4.4.3 Independencia de los Tiers B (Soporte) y C (Breakout)

En versiones anteriores, existía un sistema jerárquico donde una señal de Tier A excluía explícitamente a las de Tier B o C. Esto generaba una pérdida de datos valiosos para el modelo predictivo. Actualmente, las estrategias operan de forma ortogonal, permitiendo que la IA evalúe la calidad de cada patrón por sus propios méritos:
- **Tier B:** Captura el agotamiento de ventas a través de fractales de soporte de Bill Williams. `Tier_B = cond_4h AND is_support_fractal = 1`
- **Tier C:** Busca el momentum mediante la ruptura validada (mínimo 5 velas de antigüedad) de una resistencia. `Tier_C = cond_4h AND close > ultimo_fractal_de_resistencia`

### 4.4.4 Flujo de Prioridad y Asignación de Capital

Aunque los Tiers B y C operan de forma ortogonal, existe la posibilidad matemática de que una misma vela dispare múltiples señales simultáneamente (por ejemplo, un rebote en soporte que además rompe la SMA 200). Para gestionar estas colisiones, el sistema implementa un **flujo de prioridad en cascada ascendente**:
1. El sistema evalúa primero la condición de **Tier C**.
2. A continuación, evalúa el **Tier B**. Si se cumple, sobrescribe la señal del Tier C.
3. Por último, evalúa el **Tier A** (máxima probabilidad). Si se cumple, sobrescribe cualquier señal anterior.

Esta jerarquía garantiza que, ante un evento técnico complejo, la señal se etiquete siempre con el nivel de probabilidad más alto posible. 

Este nivel de probabilidad dicta directamente la **Asignación de Capital** mediante una aproximación conservadora del *Criterio de Kelly*, arriesgando un porcentaje mayor de la cartera en las señales más fiables:
- **Tier A:** 1.5% del capital total en riesgo.
- **Tier B:** 1.0% del capital total en riesgo.
- **Tier C:** 0.5% del capital total en riesgo.

### 4.4.5 La Clave de la Supervivencia: Gestión de Riesgo Asimétrica y *Break Even* al 1R (Universal)

Por muy depuradas que estén las señales de entrada, la estocasticidad del mercado garantiza rachas de pérdidas. El descubrimiento más importante del proceso de backtesting fue que la supervivencia del sistema no dependía del *Win Rate* de las entradas, sino de la arquitectura de salida de la operación.

El sistema fue reescrito para utilizar una **Gestión de Riesgo Dinámica basada en Volatilidad (ATR) aplicada universalmente a los Tiers A, B y C:**
1. **Stop Loss Variable (1 ATR):** En lugar de un porcentaje fijo, el riesgo se adapta al "ruido" real del mercado en ese momento exacto.
2. **Take Profit Asimétrico (3R):** Se exige que la operación pague tres veces el riesgo asumido, creando una esperanza matemática positiva incluso con *Win Rates* sub-óptimos.
3. **El Escudo del *Break Even* Dinámico (1R):** El salvavidas definitivo de la estrategia. Independientemente del Tier que genere la señal, si la operación alcanza un beneficio flotante equivalente al riesgo inicial ($+1R$), el Stop Loss se traslada algorítmicamente al precio exacto de entrada. 

**Impacto empírico:** El *Break Even* dinámico redujo la rentabilidad bruta de las operaciones que sufrían alta volatilidad antes de explotar, pero su efecto protector sobre el capital fue transformador en **todos los niveles**. Rescató las tres estrategias de la bancarrota técnica, convirtiendo Drawdowns letales en escenarios de desgaste controlado, permitiendo al sistema acumular una base de datos de entrenamiento inmensa (3,318 señales) de forma completamente segura.

### 4.4.6 Informe de Rendimiento Individual por Tier

Para validar empíricamente la efectividad de las tres estrategias operando de forma ortogonal, se sometieron a un riguroso *backtesting* individual sobre un universo de 45 activos y un periodo de 10 años. Todas fueron evaluadas bajo el mismo criterio estricto: **Riesgo Variable (1 ATR), Take Profit Fijo (3R) y Break Even Dinámico al +1R**.

A continuación, se detalla el rendimiento individual de cada Tier:

| Métrica | Tier A (SMA 200 Pullback) | Tier B (Soporte Fractal) | Tier C (Ruptura Madura) |
| :--- | :---: | :---: | :---: |
| **Operaciones Totales** | 989 | 899 | 1,430 |
| **Win Rate (Asimétrico)** | 26.7% | 26.8% | 24.5% |
| **Beneficio Neto (P&L)** | +$437,702 | +$456,821 | +$712,650 |
| **Max Drawdown (Dinámico)** | -8.73% | -4.37% | -5.52% |
| **Max Drawdown (Absoluto)** | -16.60% | -14.13% | -13.66% |
| **Profit Factor** | 2.33 | 2.89 | 3.36 |
| **Sharpe Ratio (Trade)** | 2.85 | 3.22 | 4.25 |
| **Sortino Ratio (Trade)** | 9.12 | 10.83 | 15.53 |
| **Racha Máxima Pérdidas** | 26 ops | 24 ops | 21 ops |

**Análisis de Resultados:**
1. **Volumen de Datos:** La eliminación de la jerarquía restrictiva permitió recuperar miles de ejemplos (especialmente 1,430 rupturas de Tier C que antes eran descartadas por colisiones), logrando las **3,318 señales** necesarias para entrenar modelos predictivos robustos.
2. **Consistencia:** El *Win Rate* de los tres Tiers converge de forma natural entre el 24.5% y el 26.8%. Aunque parezca bajo para el estándar tradicional, recordemos que el *Break Even* dinámico sacrifica *Win Rate* (anulando operaciones ganadoras tardías) a cambio de proteger el capital a toda costa.
3. **Esperanza Matemática:** Gracias al ratio Riesgo:Beneficio de 1:3, la rentabilidad neta agregada supera el millón y medio de dólares hipotéticos, lo que demuestra que la "lógica pura" es matemáticamente ganadora incluso antes de aplicar filtros de Inteligencia Artificial.


## 4.5 El Filtro Predictivo (Machine Learning)

A pesar de la rigurosidad matemática del sistema de Tiers basado en *Price Action* descrito en la sección anterior, los mercados financieros presentan un alto grado de estocasticidad que produce, inevitablemente, un porcentaje significativo de señales falsas (operaciones perdedoras que tocan el Stop Loss). Para mitigar este problema, se ha integrado una capa adicional de inteligencia artificial: un filtro predictivo de Machine Learning.

El objetivo de este modelo no es predecir genéricamente la dirección del mercado, sino actuar como un "segundo juez". Toma como entrada exclusivamente las operaciones que el `TierEvaluator` ya ha validado, analiza sus características técnicas (features) en el momento exacto de la señal, y predice la probabilidad matemática de que dicha operación alcance el Take Profit antes que el Stop Loss.

### 4.5.1 Prevención de Data Leakage y Pipeline de Datos

El desafío técnico más crítico en el modelado financiero predictivo es la prevención de la filtración de información futura (*data leakage*). Dado que el dataset fue etiquetado matemáticamente por el backtester utilizando precios futuros reales (hit de TP o SL), estas columnas revelan directamente el resultado de la operación.

Para evitar esto, se ha implementado la clase `MLPipeline` (`src/models/ml_pipeline.py`), que realiza las siguientes operaciones de seguridad de forma automatizada:
1.  **Purga de variables del futuro:** Elimina programáticamente 81 columnas del dataset, incluyendo cualquier variable terminada en `_precio`, `_vela` o `_hit`, conservando estrictamente los indicadores técnicos e índices de *Price Action* calculados hasta el momento de la entrada.
2.  **Partición Cronológica Estricta (TimeSeriesSplit):** A diferencia de un problema de clasificación tradicional, los datos financieros poseen una fuerte dependencia temporal asimétrica. Se estableció una barrera temporal dura el 1 de enero de 2025. El ajuste de hiperparámetros de los modelos y la calibración empírica del umbral de decisión probabilístico (60%) se realizaron utilizando **exclusivamente** el conjunto de Entrenamiento (2010-2024) mediante los *folds* internos de la validación cruzada `TimeSeriesSplit`. El conjunto de Test (2025 en adelante) se mantuvo **totalmente bloqueado (*locked*)** hasta la fase final del proyecto, garantizando que su uso se limitó estrictamente a la evaluación *Out-Of-Sample* definitiva, previniendo la contaminación de la selección del modelo.
3.  **Escalado:** Las variables numéricas son estandarizadas mediante `StandardScaler` (ajustado exclusivamente sobre los datos de entrenamiento) para garantizar un aprendizaje estable en algoritmos sensibles a la magnitud, como Support Vector Machines.

### 4.5.2 Modelos Predictivos Base y Control del Sobreajuste

Al trabajar con series temporales financieras y, especialmente en las fases iniciales del desarrollo con un dataset limitado, el riesgo de sobreajuste (*overfitting*) es severo. Si se permite que el modelo memorice el "ruido" del mercado, su capacidad de generalización en operaciones futuras reales se desploma.

Por ello, se establecieron tres modelos base (líneas base o *baselines*) fuertemente regularizados:
-   **Regresión Logística (LogReg):** Configurada con una regularización L2 agresiva (`C=0.05`). Actúa como el baseline lineal del sistema.
-   **Support Vector Machine (SVM):** Utilizando un kernel Gaussiano (RBF) con un margen de regularización suave (`C=0.5`).
-   **Random Forest (RF):** Ensamblaje de 50 árboles de decisión, severamente limitados en profundidad (`max_depth=3`) y con exigencia de al menos 3 muestras por hoja (`min_samples_leaf=3`) para forzar la abstracción.

### 4.5.3 Clasificador Avanzado: XGBoost y Justificación contra Modelos Secuenciales (LSTM)

Como modelo final, se implementó `TradeSelectorXGB` basado en XGBoost (*eXtreme Gradient Boosting*). Este algoritmo construye árboles de decisión secuencialmente para minimizar los errores de sus predecesores. 

Para su configuración financiera, se aplicó una regularización combinada: `reg_alpha=0.5` (Lasso) para forzar dispersión reduciendo a cero los pesos de características irrelevantes, y `reg_lambda=1.0` (Ridge) para penalizar ponderaciones excesivas. Adicionalmente, el ratio de aprendizaje se redujo a `learning_rate=0.05` y la profundidad máxima a 3, obligando al modelo a aprender patrones sutiles y robustos en lugar de particularidades del dataset.

**¿Por qué no se utilizaron Redes Neuronales Recurrentes (LSTM)?**
Aunque las arquitecturas LSTM son un estándar en la predicción de series temporales continuas, resultan arquitectónicamente incompatibles con este sistema. Las LSTMs requieren secuencias temporales densas (velas de precios consecutivas). Sin embargo, el dataset predictivo generado (3.318 muestras) no son velas consecutivas, sino "eventos de trading" esporádicos dispersos en el tiempo (las señales filtradas por los Tiers A/B/C). Modelar la dependencia temporal entre un evento en *Apple* en enero y otro en *Microsoft* en abril carece de validez financiera. Además, con la relativa escasez de datos (cientos de eventos por activo), el riesgo de sobreajuste de una red neuronal profunda es drásticamente superior al de un modelo basado en árboles estrictamente regularizado.

### 4.5.4 Evaluación Exploratoria y Transición hacia Métricas Financieras

En las fases exploratorias iniciales del proyecto, los cuatro modelos compitieron directamente sobre un subconjunto de datos reducido, priorizando el uso del **F1-Score** como métrica clásica de validación en aprendizaje automático.

**Resultados Preliminares (Fase Exploratoria Inicial):**
-   **Random Forest:** Logró el mejor desempeño inicial, con un F1-Score de **0.778**, un Accuracy del 87.9% y tan solo 3 Falsos Positivos.
-   **SVM:** Obtuvo un F1-Score de **0.737** (Accuracy 84.8%).
-   **XGBoost:** Mostró un Recall perfecto del 100%, resultando en un F1-Score inicial de **0.640**.
-   **Regresión Logística:** Quedó rezagada con un F1-Score de **0.522**.

**Análisis de Importancia de Variables (Feature Importance):**
El análisis paramétrico inicial reveló qué características técnicas tienen mayor poder predictivo. Destacan significativamente:
1.  **Volatilidad (`ATR_14`):** Una de las variables con mayor poder predictivo. Entornos de alta volatilidad aumentan drásticamente la probabilidad matemática de que el ruido del mercado alcance el Stop Loss.
2.  **Momentum (`impulso`):** El tamaño relativo de la vela de entrada indica la convicción institucional.
3.  **Sobreextensión del Precio (`dist_SMA_50` y `dist_SMA_200`):** Crítico para capturar el principio de "reversión a la media".

*Nota metodológica crucial:* Aunque Random Forest despuntó en esta primera fase puramente estadística (y el F1-Score alto sugirió su viabilidad inicial), **este resultado probó ser temporal y engañoso al escalar el modelo**. Como se demostrará con rigor empírico en la sección 4.7.2 (Matriz de Confusión Financiera) mediante validación Walk-Forward sobre el dataset completo definitivo (3.318 operaciones), las métricas clásicas demostraron ser insuficientes para el dominio financiero. Al evaluar el impacto económico asimétrico (R-múltiplos), **XGBoost demostró una superioridad aplastante** (+106R neto frente a +6R de Random Forest), justificando objetivamente su selección definitiva como el motor predictivo (`best_model.pkl`) desplegado en el agente gestor.


## 4.6 El Agente Gestor de Cartera: `PortfolioAgent`

Una vez que el filtro predictivo de Machine Learning aprueba una señal técnica, la decisión de inversión entra en su fase más crítica: ¿cuánto capital comprometer, cuándo salir y qué hacer si el mercado entra en un régimen adverso? Estas responsabilidades recaen sobre el `PortfolioAgent`, implementado en `src/models/agent_logic.py`.

### 4.6.1. Definición Formal del Agente (Arquitectura PEAS)

A efectos de la Inteligencia Artificial, este sistema se define formalmente como un **Agente Reactivo Basado en Modelos** (*Model-based Reflex Agent*). No se ha empleado ningún *framework* genérico de agentes (como Mesa o LangChain); el agente ha sido diseñado y programado desde cero mediante una arquitectura modular estricta en Python.

Utilizando el estándar de especificación PEAS (*Performance, Environment, Actuators, Sensors*), la ontología del agente se define como sigue:

- **Medida de Rendimiento (*Performance Objective*):** El objetivo primordial del agente no es maximizar el retorno absoluto, sino maximizar el **Sharpe Ratio** (rentabilidad ajustada por riesgo), manteniendo estrictamente el *Max Drawdown* histórico por debajo del -15% institucional.
- **Entorno (*Environment*):** El mercado financiero de renta variable estadounidense (S&P 500 y Nasdaq-100). Es un entorno parcialmente observable, estocástico, secuencial, dinámico, continuo y multiagente.
- **Sensores (*Perceptions*):** El agente no interactúa con los precios crudos. Sus percepciones de entrada son: (1) las señales estructurales generadas por el sistema de Tiers, (2) la probabilidad de éxito $P(TP)$ que retorna XGBoost, (3) el índice de volatilidad macroeconómica (VIX), y (4) el porcentaje de amplitud del mercado global. Adicionalmente, posee un **Estado Interno** en memoria (la curva de capital histórica, el *Peak Equity* máximo alcanzado, el capital disponible y el historial temporal de operaciones) necesario para lidiar con la observabilidad parcial del mercado.
- **Actuadores (*Actions*):** El agente puede invocar los siguientes comandos sobre el módulo `Portfolio`: `Comprar_Activo(volumen_calculado)`, `Vender_Activo(motivo_salida)`, y `Bloquear_Mercado(días_cooldown)`.

El diseño y proceso de decisión del Agente responde directamente a los desafíos planteados:
1.  **Evaluación probabilística:** El Agente aplica un umbral estricto para filtrar las predicciones de XGBoost.
2.  **Gestión dinámica del riesgo:** Interacciona con el `RiskManager` para dimensionar el volumen de compra según algoritmos financieros.
3.  **Control de exposición:** Ejecuta filtros defensivos (actuadores de bloqueo) ante cambios bruscos en las percepciones macro (VIX/Drawdown).

### 4.6.1 Filtro Probabilístico de la Inteligencia Artificial

El primer filtro que aplica el `PortfolioAgent` es la comprobación de la probabilidad de éxito predicha por el modelo. Cada señal técnica validada por el `TierEvaluator` es escalada mediante el `scaler.pkl` y pasada al modelo `best_model.pkl` para obtener su probabilidad de alcanzar el Take Profit (`P(TP)`).

El umbral de aceptación está configurado por defecto en **P(TP) ≥ 0.60** (60%). Este valor fue determinado empíricamente durante la evaluación del Sprint 4: por debajo de este umbral, el número de Falsos Positivos en el conjunto de test aumenta significativamente, erosionando la rentabilidad esperada de la cartera.

Las señales por encima del umbral son ordenadas de mayor a menor probabilidad (ranking), garantizando que si la liquidez de la cartera no permite ejecutar todas las órdenes del día, el sistema invertirá primero en las oportunidades de mayor convicción estadística.

### 4.6.2 Reglas de Rebalanceo Dinámico

A diferencia de los sistemas de gestión pasiva de carteras (como la inversión en índices), el Agente implementa un conjunto de reglas activas de rebalanceo basadas en la literatura académica de finanzas cuantitativas. Estas reglas operan en cascada, de modo que si cualquiera de las comprobaciones globales falla, el Agente bloquea todas las nuevas compras del día sin necesidad de evaluar las señales individuales.

#### Regla 1: Filtro de Pánico Macroeconómico — Filtro VIX

Los mercados financieros no operan bajo una distribución estadística constante, sino bajo diferentes regímenes de volatilidad. Ang y Bekaert (2002) demostraron empíricamente mediante modelos de *Regime-Switching* que, cuando la volatilidad implícita se dispara, las correlaciones entre activos tienden a uno (todo cae simultáneamente), invalidando los supuestos de diversificación del modelo de Markowitz.

El índice VIX (*CBOE Volatility Index*) es el indicador estándar de la industria para medir el "miedo" del mercado, calculado como la volatilidad implícita de las opciones sobre el S&P 500 a 30 días. Un VIX por encima de 30 puntos señala un régimen de pánico (*Risk-Off*), donde los modelos entrenados en periodos de normalidad pierden su validez estadística.

El `PortfolioAgent` incorpora este principio como su primera comprobación: si el valor del VIX del día (extraído del dataset FRED como `VIXCLS`) supera el umbral de 30, el Agente no abre ninguna nueva posición independientemente de lo que prediga el modelo de Machine Learning, trasladándose a liquidez (cash) como estrategia defensiva.

*Fuentes: [12] [13]*

#### Regla 2: Kill-Switch Global — Interruptor de Emergencia por Máximo Drawdown

Los stop-loss individuales de cada operación protegen contra el fracaso aislado de una empresa, pero no contra el colapso sistémico del mercado (un *Black Swan* o Cisne Negro). La Teoría de Protección de Carteras (*Constant Proportion Portfolio Insurance*, CPPI), formalizada por Black y Jones (1987), establece que debe existir un mecanismo global de última línea de defensa que suspenda la operativa cuando la pérdida acumulada de la cartera supere un umbral crítico.

El `PortfolioAgent` implementa este principio mediante el **Kill-Switch Global**: si el valor total de la cartera (Total Equity) cae más de un **10% desde su máximo histórico** (parámetro `max_drawdown_limit = -0.10`), el Agente activa automáticamente un **período de enfriamiento de 30 días** (*cooling-off period*), durante el cual bloquea cualquier nueva apertura de posiciones. Este mecanismo garantiza que el sistema no continúe acumulando pérdidas en un mercado bajista estructural, esperando a que las condiciones se estabilicen antes de reanudar la operativa.

*Fuente: [14]*

#### Regla 3: Time-Stop — El Método de la Triple Barrera

La literatura clásica de gestión de riesgo contempla únicamente dos barreras de salida: el Stop Loss (protección contra pérdidas) y el Take Profit (realización de ganancias). Sin embargo, López de Prado (2018), en su obra *Advances in Financial Machine Learning*, formaliza una tercera barrera: el **Time-Stop** o barrera temporal.

El concepto académico subyacente es el de la *deriva del modelo* (*Model Drift*): las predicciones de un modelo de Machine Learning tienen una ventana de validez estadística. Si el mercado no se mueve en la dirección esperada dentro de un plazo razonable, la hipótesis del modelo sobre ese activo ha perdido su vigencia y mantener el capital inmovilizado genera un **Coste de Oportunidad** que erosiona la rentabilidad de la cartera.

En el presente sistema, el Time-Stop opera con una doble condición de cierre: se cierra la posición si han transcurrido más de **30 días** desde la entrada (un mes de mercado) sin resultado, o si el retorno flotante de la posición ha caído por debajo del **−3%** sin haber alcanzado el Stop Loss definido por el `RiskManager`. Esta segunda condición permite capturar posiciones que se deterioran lentamente y que, de no cerrarse, acabarían tocando el Stop Loss en peores condiciones.

*Fuente: [15]*

#### Regla 4: Kelly Fraccional — Ajuste Dinámico del Riesgo

La asignación fija de capital por Tier (Tier A = 2%, Tier B = 1.5%, etc.) presenta una limitación: no se adapta a los periodos en los que el modelo de Machine Learning está en una racha de errores. El Criterio de Kelly, formalizado matemáticamente por Kelly (1956) y ampliado por MacLean, Thorp y Ziemba (2011), proporciona la fracción óptima del capital a invertir en función de la tasa de aciertos y el ratio ganancia/pérdida esperado:

```
f* = W − (1 − W) / R
```

Donde `f*` es la fracción óptima a invertir, `W` es la tasa de aciertos (*Win Rate*) del modelo y `R` es el ratio medio de ganancia sobre pérdida (*Payoff Ratio*).

Dado que el Kelly Puro es matemáticamente agresivo (puede recomendar apostar el 50% del capital en una sola operación), el sistema implementa el **Half-Kelly** (`f* × 0.5`), estrategia estándar en los fondos cuantitativos institucionales, que mantiene un perfil de riesgo más conservador preservando la esencia del ajuste dinámico. El `PortfolioAgent` reevalúa el Kelly Factor cada 50 operaciones cerradas, reduciendo automáticamente el riesgo base de todos los Tiers si el modelo está atravesando un período de menor precisión.

*Fuente: [16]*

#### Regla 5: Límite de Correlación — Diversificación Cuantitativa (Markowitz)

Cuando el mercado entra en una tendencia alcista sectorial, el modelo de Machine Learning puede emitir señales de compra simultáneas sobre varios activos del mismo sector. Sin embargo, activos de un mismo sector presentan un coeficiente de correlación de Pearson elevado, lo que implica que, ante una noticia adversa del sector, todas las posiciones caerían simultáneamente, concentrando el riesgo en lugar de diversificarlo.

La Teoría Moderna de Carteras de Markowitz (1952) establece formalmente que la diversificación óptima requiere invertir en activos con correlaciones bajas entre sí, reduciendo el riesgo total de la cartera sin sacrificar la rentabilidad esperada.

El `PortfolioAgent` implementa este principio calculando el coeficiente de correlación de Pearson entre el historial de precios del nuevo activo y los ya aprobados para comprar ese día. Si la correlación supera el umbral de **0.80**, la señal es descartada para evitar la concentración de riesgo sectorial.

*Fuente: [2]*

#### Regla 6: Filtro de Amplitud de Mercado — Market Breadth Filter

La observación del nivel del S&P 500 como indicador de la salud del mercado puede ser engañosa: el índice puede subir impulsado únicamente por los 5 o 10 valores de mayor capitalización (los "Magnificent 7" tecnológicos), mientras que el 80% de las empresas están en tendencia bajista. Esta condición, conocida como *Bull Trap* o trampa alcista, genera señales de compra técnicas en un mercado estructuralmente débil.

Faber (2007) formalizó el concepto de *Market Breadth* (*Amplitud de Mercado*) como filtro de posicionamiento táctico: el porcentaje de activos del universo de inversión que cotizan por encima de su Media Móvil de 200 períodos es un indicador robusto de la salud subyacente del mercado.

El `PortfolioAgent` aplica este principio bloqueando nuevas compras cuando menos del **40% del universo de activos** cotiza por encima de su SMA_200. Este filtro funciona como una capa macroeconómica complementaria al Filtro VIX: mientras el VIX captura el pánico puntual (alta volatilidad implícita), el Breadth Filter captura el deterioro estructural gradual del mercado.

*Fuente: [17]*

### 4.6.3 Filtro de Liquidez y Slippage (Illiquidity Penalty)

Una limitación fundamental de la mayoría de los backtests publicados en la literatura académica es la asunción de ejecución perfecta: se asume que el inversor puede comprar y vender exactamente al precio de cierre de la vela. En la realidad, esto no es posible por dos razones:

1.  **Slippage (deslizamiento):** La diferencia entre el precio teórico de la orden y el precio real de ejecución, causada por el movimiento del mercado entre la decisión y la ejecución.
2.  **Iliquidez:** En activos de bajo volumen de negociación, el *bid-ask spread* (diferencia entre el precio de compra y venta más favorable disponible) puede consumir una fracción significativa de los beneficios esperados.

Amihud (2002) demostró empíricamente la existencia de una prima de iliquidez significativa en los mercados de renta variable: los activos con menor ratio de volumen sobre cambio de precio ofrecen mayores rentabilidades esperadas precisamente como compensación por su menor liquidez, pero generan costes de transacción implícitos que erosionan los resultados del backtesting si no se modelan correctamente.

El módulo `Portfolio` (`src/environment/portfolio.py`) incorpora estos efectos de la siguiente manera:
-   **Slippage Rate (0.05%):** El precio de ejecución real de cada orden se penaliza en un 0.05% respecto al precio teórico de cierre, simulando el coste del *bid-ask spread*.
-   **Filtro de Volumen Mínimo:** Si el volumen medio de los últimos días del activo es inferior a 100.000 acciones diarias, la orden se rechaza automáticamente, evitando simular operaciones en activos prácticamente no negociables.

*Fuente: [18]*

### 4.6.4 Flujo de Decisión del Agente

El proceso completo de decisión del `PortfolioAgent` sigue una cascada estricta de comprobaciones, de modo que si cualquier filtro global falla, se interrumpe la evaluación sin consumir recursos computacionales:

```
Inicio del ciclo diario
        │
        ▼
[1] ¿Kill-Switch activado?  → SÍ → No operar (período enfriamiento 30 días)
        │ NO
        ▼
[2] ¿VIX > 30?              → SÍ → No operar (régimen de pánico macro)
        │ NO
        ▼
[3] ¿Breadth < 40%?         → SÍ → No operar (mercado estructuralmente débil)
        │ NO
        ▼
[4] Por cada señal técnica del día:
    ├── [4a] ¿P(TP) < 60%?        → Rechazar señal
    ├── [4b] ¿Correlación > 0.8?  → Rechazar señal
    └── [4c] RiskManager → Calcular Position Sizing
        │
        ▼
[5] Ranking de órdenes (mayor probabilidad primero)
        │
        ▼
[6] Portfolio.execute_trade() → Ejecutar compras (aplicando slippage + comisión)
```

Este diseño en cascada garantiza que las comprobaciones de mayor impacto y menor coste computacional (filtros globales de régimen) se ejecuten primero, preservando la eficiencia del sistema durante las simulaciones históricas de larga duración.

---

> **NOTA DE MAQUETACIÓN:** El siguiente bloque de referencias bibliográficas (Capítulo 7) se incluye aquí provisionalmente para facilitar la revisión del borrador. En la versión final de la memoria Word, este bloque debe trasladarse al **final del documento**, después de las Conclusiones y Trabajo Futuro, según la normativa de la MUIINF.

---

# CAPÍTULO 7: REFERENCIAS BIBLIOGRÁFICAS


Las referencias se presentan en formato APA 7.ª edición, ordenadas por número de aparición en el texto.

[1] U.S. Securities and Exchange Commission (SEC). (2014). *Equity Market Structure Literature Review, Part II: High Frequency Trading*. SEC Staff Report. Recuperado de https://www.sec.gov/

[2] Markowitz, H. (1952). Portfolio Selection. *The Journal of Finance*, *7*(1), 77–91. https://doi.org/10.2307/2975974

[3] Fama, E. F., & French, K. R. (1993). Common risk factors in the returns on stocks and bonds. *Journal of Financial Economics*, *33*(1), 3–56. https://doi.org/10.1016/0304-405X(93)90023-5

[4] Elder, A. (1993). *Trading for a Living: Psychology, Trading Tactics, Money Management*. John Wiley & Sons.

[5] Appel, G. (2005). *Technical Analysis: Power Tools for Active Investors*. FT Press.

[6] Murphy, J. J. (1999). *Technical Analysis of the Financial Markets: A Comprehensive Guide to Trading Methods and Applications*. New York Institute of Finance.

[7] Kimoto, T., Asakawa, K., Yoda, M., & Takeoka, M. (1990). Stock market prediction system with modular neural networks. *Proceedings of the 1990 IJCNN International Joint Conference on Neural Networks*, 1–6. https://doi.org/10.1109/IJCNN.1990.137535

[8] Refenes, A. N., Zapranis, A., & Francis, G. (1994). Stock performance modeling using neural networks: A comparative study with regression models. *Neural Networks*, *7*(2), 375–388. https://doi.org/10.1016/0893-6080(94)90030-2

[9] Breiman, L. (2001). Random Forests. *Machine Learning*, *45*(1), 5–32. https://doi.org/10.1023/A:1010933404324

[10] Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794. https://doi.org/10.1145/2939672.2939785

[11] López de Prado, M. (2018). *Advances in Financial Machine Learning*. John Wiley & Sons. (Capítulo 3: The Triple-Barrier Method & Metalabealing). ISBN: 978-1119482086.

[12] Ang, A., & Bekaert, G. (2002). International Asset Allocation with Regime Shifts. *The Review of Financial Studies*, *15*(4), 1137–1187. https://doi.org/10.1093/rfs/15.4.1137

[13] Gomes, F., Khorunzhina, N., & Polkovnichenko, V. (2018). *Risk on-risk off: A regime switching model for active portfolio management*. Working Paper, EconStor / CETE.

[14] Black, F., & Jones, R. (1987). Simplifying Portfolio Insurance. *The Journal of Portfolio Management*, *14*(1), 48–51. https://doi.org/10.3905/jpm.1987.409131

[15] López de Prado, M. (2018). *Advances in Financial Machine Learning*. John Wiley & Sons. ISBN: 978-1119482086.

[16] MacLean, L. C., Thorp, E. O., & Ziemba, W. T. (2011). *The Kelly Capital Growth Investment Criterion: Theory and Practice*. World Scientific. ISBN: 978-9814293495.

[17] Faber, M. T. (2007). A Quantitative Approach to Tactical Asset Allocation. *The Journal of Wealth Management*, *10*(4), 12–28. https://doi.org/10.3905/jwm.2007.674809

[18] Amihud, Y. (2002). Illiquidity and stock returns: cross-section and time-series effects. *Journal of Financial Markets*, *5*(1), 31–56. https://doi.org/10.1016/S1386-4181(01)00024-6

[19] Fama, E. F. (1970). Efficient Capital Markets: A Review of Theory and Empirical Work. *The Journal of Finance*, *25*(2), 383–417. https://doi.org/10.2307/2325486

[20] Williams, B. (1995). *Trading Chaos: Applying Expert Techniques to Maximize Your Profits*. John Wiley & Sons.

[21] Harvey, C. R., Hoyle, E., Korgaonkar, R., Rattray, S., Sargaison, M., & Van Hemert, O. (2018). The Impact of Volatility Targeting. *The Journal of Portfolio Management*, *45*(1), 14–33. https://doi.org/10.3905/jpm.2018.45.1.014


> *Nota de estructura: Los resultados completos de la evaluación comparativa de los cuatro modelos (incluyendo la tabla de métricas Out-of-Sample, la Matriz de Confusión Financiera y el análisis de Importancia de Variables), así como la Evaluación Institucional del Portfolio contra los índices de mercado, se desarrollan en detalle en las secciones 4.7, 4.8 y 4.9 del presente capítulo, ordenadas según el protocolo de experimentación científica establecido en el Sprint 4 del proyecto.*



Se sometieron a prueba cuatro algoritmos clásicos y de estado del arte:
1. **Regresión Logística (LogReg):** Modelo base lineal, fuertemente regularizado.
2. **Support Vector Machine (SVM):** Modelo matemático geométrico con kernel RBF.
3. **Random Forest (RF):** Ensamblaje arbóreo basado en Bagging.
4. **Extreme Gradient Boosting (XGBoost):** Ensamblaje avanzado basado en Boosting secuencial con regularización L1 (Lasso) y L2 (Ridge).

#### Resultados y Métricas de Rendimiento (Dataset Out-of-Sample)

A continuación, se presenta la tabla comparativa de los algoritmos en el conjunto de prueba (Test), ordenada por su F1-Score:

| Modelo | Accuracy | Precision | Recall | Specificity | F1-Score | ROC AUC | Brier Score | TN | FP | FN | TP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost** | **80.48%** | **60.81%** | 30.61% | **94.47%** | **0.4072** | **0.6986** | **0.1529** | 495 | 29 | 102 | 45 |
| **Random Forest** | 49.18% | 25.63% | **69.39%** | 43.51% | 0.3743 | 0.6323 | 0.2361 | 228 | 296 | 45 | 102 |
| **LogReg** | 58.42% | 26.43% | 50.34% | 60.69% | 0.3466 | 0.5722 | 0.2344 | 318 | 206 | 73 | 74 |
| **SVM** | 54.69% | 24.76% | 52.38% | 55.34% | 0.3362 | 0.5686 | 0.1704 | 290 | 234 | 70 | 77 |

*Nota: TN = True Negatives, FP = False Positives, FN = False Negatives, TP = True Positives.*

**Análisis de las Métricas:**
- **Especificidad (94.47%) y Precisión (60.81%):** El modelo XGBoost prioriza radicalmente la seguridad frente a la frecuencia operativa. Demuestra una altísima capacidad para filtrar falsas señales (solo 29 Falsos Positivos frente a 495 Verdaderos Negativos filtrados correctamente). Esto es crítico en la gestión de capital, ya que evita entrar en operaciones destinadas a perder.
- **Brier Score (0.1529):** El XGBoost presenta la mejor calibración de probabilidad (el valor más bajo es mejor), demostrando que cuando el modelo asigna una alta probabilidad a un "Trade Ganador", es matemáticamente fiable.
- **ROC AUC (0.6986):** En un entorno financiero con muchísimo ruido estocástico, un AUC cercano a 0.70 indica un poder discriminatorio altamente satisfactorio entre operaciones rentables y nulas/perdedoras.

#### Importancia de Variables (Feature Importance) en XGBoost

La naturaleza interpretable de los árboles de decisión permite extraer la "lógica interna" de la IA. Las cinco variables con mayor peso predictivo según el modelo ganador (XGBoost) fueron:

1. **EMA_200 (0.0617):** Confirma la supremacía de la tendencia general subyacente. El sesgo estructural de largo plazo es el mayor predictor de éxito.
2. **Impulso (0.0536):** La métrica derivada del Momentum valida que el modelo busca aceleración direccional en el activo antes de dar el visto bueno.
3. **Open (0.0476):** El nivel de apertura del Price Action.
4. **BE_Hit (0.0430):** La capacidad algorítmica de evaluar si la estructura permite alcanzar la zona segura de Break Even antes del Take Profit absoluto.
5. **SMA_50_1D (0.0341):** La alineación fractal con el Timeframe diario ratifica la tesis operativa Multitimeframe definida en el sistema base.

Dadas las métricas expuestas, el algoritmo **XGBoost se establece como el motor inteligente definitivo (best_model.pkl)**, operando como un "filtro quirúrgico" diseñado para maximizar la supervivencia del capital (alta especificidad) a expensas del volumen total de operaciones (recall conservador). El análisis de sensibilidad del impacto de cada regla institucional del Agente sobre el rendimiento se desarrolla en la siguiente sección.

## 4.6. Análisis de Sensibilidad de la Cartera y Paradoja de Seguridad Institucional


Durante el proceso de validación del sistema (Simulación Walk-Forward 2025-2026), se llevó a cabo una serie de tests de estrés para evaluar la sensibilidad del sistema frente a variables críticas: gestión de riesgo (Position Sizing), umbrales de probabilidad del modelo (XGBoost), y la implementación de reglas clásicas institucionales. Los resultados obtenidos revelaron hallazgos estadísticos fundamentales sobre el comportamiento algorítmico frente a intervenciones externas.

### 4.6.1. La Alineación del Edge: Riesgo Estructural vs. Toma de Beneficios
Uno de los hallazgos más relevantes se centró en la coherencia geométrica entre la entrada y la salida. Inicialmente, el sistema presentó una media ganadora inferior a la perdedora (Aprox. +$80 vs -$195) a pesar de sostener un Win Rate del 80%. El diagnóstico reveló una disonancia estructural:
* El dimensionamiento de la posición (Position Sizing) se calculaba en base a un riesgo microestructural ajustado (1 ATR, indicando alta sensibilidad).
* Sin embargo, la ejecución de la salida imponía un límite conservador lejano (Stop Loss en la SMA 200). 
Al alinear estrictamente las reglas de salida en la simulación con la etiqueta predicha por el modelo de ML en el Smoke Test (Stop Loss a 1 ATR y Take Profit asimétrico a 3R), el Profit Factor se catapultó a niveles superiores a 2.0 (ej. Win Rate 57.1% con Profit Factor 2.34 arriesgando un 0.75% por operación). Esto demostró empíricamente que la ventaja (Edge) de XGBoost requiere que el entorno de simulación respete la estructura de recompensa/riesgo sobre la que fue optimizado.

### 4.6.2. La Paradoja de la Seguridad: Kill-Switch y Time-Stop
En un esfuerzo por mitigar caídas de capital severas, se activaron temporalmente protocolos institucionales clásicos:
1. **Kill-Switch por Drawdown**: Suspensión operativa de un mes tras un bache superior al 10%.
2. **Time-Stop**: Cierre forzado de posiciones estancadas más de 30 días para liberar liquidez.

Paradójicamente, la superposición de estas defensas provocó una degradación crítica en el rendimiento general de la estrategia, reduciendo la rentabilidad total de un sobresaliente +21.0% a un +8.9%, y empeorando el Drawdown Máximo (-11.2%).
* **El Falso Pánico del Kill-Switch**: Al bloquear la operativa durante un mes tras un retroceso del -10.2%, el sistema se desconectó precisamente en los puntos de reversión técnica (mean-reversion rebounds) de mayor asimetría probabilística. Las mejores oportunidades algorítmicas surgen, por definición, en situaciones de sobreventa profunda. El Kill-Switch saboteó la capacidad de recuperación del sistema cortándole el acceso a los trades más rentables del año.
* **La Amputación del Time-Stop**: El cierre forzado a 30 días obligó a la cristalización de beneficios mediocres o pequeñas pérdidas en operaciones que, debido a la dinámica del mercado, necesitaban mayor margen temporal para alcanzar el ambicioso Take Profit de 3R. Esto destruyó la asimetría lograda en la sección anterior, mermando gravemente el Profit Factor.

**Conclusión del Análisis**: Un modelo estocástico de alta precisión (XGBoost validado al 60% de probabilidad umbral), combinado con una estricta limitación del riesgo por operación (0.75% Flat Risk), genera por sí mismo una curva de capital geométricamente superior. Añadir superposiciones de "seguridad" heurísticas sobre un sistema estadísticamente robusto interfiere negativamente en la distribución matemática del modelo, demostrando que en el trading cuantitativo, la mejor defensa es la propia esperanza matemática.

### 4.6.3. Optimización Asimétrica del Break Even (BE)
Con el objetivo de blindar el capital en operaciones rentables, se evaluó la implementación de una regla de *Break Even* (mover el Stop Loss al precio de entrada) en diferentes umbrales de beneficio. Asumiendo un riesgo estructural del 1.0% por operación y un objetivo teórico de Take Profit a 3R, los resultados demostraron la sensibilidad extrema del modelo frente a los cierres prematuros.

Se plantearon tres escenarios de simulación:

1. **Escenario Base (Sin Break Even):**
   - **Rentabilidad:** +27.0% | **Max Drawdown:** -8.7%
   - **Win Rate:** 55.0% | **Profit Factor:** 2.40
   - *Análisis:* Dejar al mercado respirar sin mover el Stop Loss original permitió que la probabilidad del modelo (XGBoost) se materializara, asumiendo grandes pérdidas ocasionales (debido a gaps de ejecución a cierre de vela) pero compensadas por extraordinarias ganancias.

2. **Escenario Conservador (BE activado a +1R):**
   - **Rentabilidad:** +17.0% | **Max Drawdown:** -7.1%
   - **Win Rate:** 35.0% | **Profit Factor:** 2.25
   - *Análisis:* Al blindar la posición demasiado pronto, la pérdida media por operación se redujo drásticamente a la mitad. Sin embargo, la volatilidad normal del mercado provocó que operaciones legítimas retrocedieran hasta el punto de entrada, cerrándose en Break Even (\$0) antes de continuar su camino hacia el objetivo de 3R. Esto desplomó la tasa de acierto del 55% al 35%, demostrando que **la sobreprotección temprana asfixia la esperanza matemática**.

3. **Escenario Óptimo (BE activado a +2R):**
   - **Rentabilidad:** +28.3% | **Max Drawdown:** -7.7%
   - **Win Rate:** 55.0% | **Profit Factor:** 2.56
   - *Análisis:* Al activar el Break Even únicamente en etapas muy avanzadas del recorrido del precio (cuando el beneficio ya es el doble del riesgo inicial), el sistema logró el punto dulce (Sweet Spot). Se mantuvo intacto el Win Rate del 55% (dando espacio a la volatilidad natural) pero se protegieron las "casi victorias" de reversiones catastróficas. Este ajuste generó el **máximo histórico de rentabilidad del sistema (+28.3%)** junto con el Profit Factor más asimétrico (2.56).

**Conclusión:** La gestión de la salida (Trade Management) es tan crítica como el modelo de predicción. Intervenir la posición muy pronto (+1R) destruye la ventaja del algoritmo, mientras que retrasar la intervención hasta una zona probabilística más madura (+2R) maximiza el rendimiento y minimiza el daño por volatilidad de cola.

## 4.7. Protocolo de Experimentación y Justificación de Algoritmos (XAI)

Para demostrar la madurez técnica del sistema y justificar la elección de XGBoost como motor predictivo final, el protocolo de experimentación se estructuró en cuatro pilares metodológicos, acompañados de sus respectivos artefactos analíticos (disponibles en la carpeta `results/figures/`).

### 4.7.1. Análisis Exploratorio y Saneamiento del Dataset (Anti-Leakage)
El dataset original constaba de 3.318 operaciones con 188 variables. Para garantizar la viabilidad algorítmica y prevenir la filtración de información futura (*Data Leakage*), se purgó estrictamente cualquier variable asociada a la resolución matemática de la operación (precios de cierre del SL/TP, duración de la operación, etc.), descartando 135 columnas. Finalmente, tras eliminar las filas con NaNs (derivadas del periodo de precalentamiento de la SMA 200), el espacio de entrenamiento se condensó en 52 *features* puramente predictivas.
La muestra presenta un desbalanceo natural derivado de la dificultad del mercado financiero, donde las operaciones ganadoras (hit de TP) son menos frecuentes que las perdedoras (hit de SL o BE), justificando el uso de algoritmos robustos al desbalanceo.

### 4.7.2. Benchmarking y Matriz de Confusión Financiera
Las métricas clásicas de Machine Learning (Accuracy, F1-Score) son insuficientes en finanzas, ya que no todas las predicciones erróneas tienen el mismo coste monetario. Se desarrolló una **Matriz de Confusión Financiera** evaluando los modelos en términos de R-múltiplos (Riesgo), asumiendo un riesgo estricto del 1.0% por operación y un retorno de 3R:
- **Falsos Positivos (FP):** El modelo aprueba la operación, pero fracasa. Coste: -1R.
- **Verdaderos Positivos (TP):** El modelo aprueba la operación y triunfa. Ganancia: +3R.
- **Verdaderos Negativos (TN):** El modelo rechaza la operación y, efectivamente, iba a fracasar. Ahorro implícito: 1R.

Bajo este paradigma, modelos lineales como LogReg o SVM fracasaron, destruyendo valor al acumular más de 200 Falsos Positivos (generando pérdidas netas o balances pírricos). En contraste, **XGBoost logró un balance neto Out-of-Sample de +106R**, gracias a su altísima especificidad (94.47%), rechazando de forma implacable el "ruido" del mercado (495 TN) y operando únicamente en setups de altísima asimetría matemática.

### 4.7.3. Explicabilidad de la Inteligencia Artificial (XAI)
Para evitar el paradigma de "caja negra" en la toma de decisiones financieras, se implementaron diagramas de SHAP (*SHapley Additive exPlanations*) sobre los cuatro modelos evaluados. SHAP es un marco teórico basado en la Teoría de Juegos Cooperativos de Shapley que asigna a cada variable una contribución marginal a la predicción individual, respetando propiedades de eficiencia, simetría, linealidad y valores nulos formalmente demostradas [24][27]. A diferencia de la importancia de variables global (que promedia el efecto sobre todo el dataset), SHAP proporciona una explicación local de por qué el modelo tomó una decisión concreta para cada operación.

El análisis SHAP de XGBoost revela que la Inteligencia Artificial no descubrió un indicador "mágico", sino que replicó lógicamente los pilares del análisis institucional:
1. **El Contexto Macro dicta la probabilidad:** La media móvil de 200 periodos (EMA_200) y la SMA diaria dominan el modelo. Los valores altos del indicador alcista empujan fuertemente la probabilidad hacia la clase positiva (ganancia), confirmando la tesis multi-timeframe del sistema [4][6].
2. **Momento Direccional:** El indicador de *impulso* actúa como el principal gatillo, descartando rebotes débiles o consolidaciones muertas. Este hallazgo es coherente con los factores sistémicos de momentum documentados por Fama y French (1993) [3].
3. **Volatilidad (NATR_14):** El rango de volatilidad normalizado activo en el momento de la entrada es el tercer predictor en importancia, confirmando el hallazgo de la Feature Importance de Random Forest: el modelo penaliza las entradas en entornos de alta volatilidad, donde el ruido del mercado tiene mayor probabilidad de activar el Stop Loss antes que el Take Profit [23].

### 4.7.4. Evaluación Fuera de Muestra (Curva de Capital Walk-Forward)
La prueba definitiva del sistema se evaluó mediante un particionado Walk-Forward estricto, entrenando con datos previos a 2025 y testeando en un entorno puro de 2025 en adelante. 
La curva de capital generada (`equity_curve_comparison.png`) ilustra gráficamente el resultado de la matriz financiera: mientras que el modelo pasivo (Baseline) se desploma en severos Drawdowns, XGBoost construye una curva ascendente de volatilidad muy contenida. El modelo sacrifica oportunidades (crecimiento más lento) a cambio de una precisión clínica, demostrando por qué la regresión logística o Random Forest son insuficientes para lidiar con la no linealidad de los mercados modernos.

## 4.8. Estudio de Ablación y Comparativa Exhaustiva de Modelos Predictivos

Para cuantificar objetivamente la contribución individual de cada capa tecnológica (y certificar el valor añadido real del Machine Learning frente a la operativa tradicional algorítmica), se diseñó un extenso Estudio de Ablación (*Ablation Study*). Se ejecutó el simulador histórico sobre el periodo *Out-Of-Sample* (2025 en adelante) aislando cada uno de los modelos predictivos entrenados y comparándolos contra una línea base sin Inteligencia Artificial. En todas las pruebas se mantuvo activa la capa de Gestión de Riesgo (dimensionamiento de Kelly, filtro macroeconómico VIX y control de correlaciones de Markowitz) para garantizar que las diferencias de rendimiento se deben única y exclusivamente a la calidad predictiva de las señales.

Los resultados empíricos obtenidos de la simulación de cartera arrojan las siguientes métricas:

| Arquitectura / Modelo Predictivo | Retorno Total | Max Drawdown | Win Rate | Profit Factor | N.º Operaciones |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **No ML (Solo Price Action + Riesgo)** | +51.16% | -11.89% | 41.50% | 1.70 | 65 |
| **Regresión Logística (LogReg)** | -13.40% | -23.80% | 21.90% | 0.64 | 32 |
| **Support Vector Machine (SVM)** | +0.00% | +0.00% | 0.00% | 0.00 | 0 |
| **Random Forest (RF)** | +17.50% | -5.10% | 53.30% | 2.59 | 15 |
| **XGBoost (Modelo Final Seleccionado)** | +28.30% | **-7.74%** | **55.00%** | **2.56** | 20 |

### Análisis Descriptivo por Modelo

**1. Baseline Sin Machine Learning (Solo Reglas Tiers):**
Operar exclusivamente basándose en reglas técnicas genera la mayor rentabilidad bruta (+51.16%). Al no existir un filtro probabilístico, el sistema toma **todas** las señales (65 operaciones), capturando íntegramente las grandes tendencias alcistas. Sin embargo, esta rentabilidad tiene un coste estructural grave: el *Win Rate* cae al 41.50% y el *Max Drawdown* asciende a -11.89%. Aunque el sistema sobrevive gracias a los estrictos Stop Loss del Risk Manager, la alta frecuencia operativa, el excesivo pago de comisiones (*slippage*) y la dependencia de un mercado direccionalmente puro lo hace inadecuado para la gestión institucional a largo plazo.

**2. Regresión Logística (LogReg):**
Fracasa estrepitosamente. Destruye capital (-13.40% de retorno) sufriendo un *Drawdown* inaceptable del -23.80%. La regresión logística es un modelo probabilístico estrictamente lineal; al enfrentarse a la naturaleza no lineal y caótica de los mercados financieros, clasifica erróneamente el ruido estocástico como señales válidas, evidenciando que la predicción bursátil moderna no puede ser resuelta mediante hiperplanos de separación lineales simples.

**3. Support Vector Machine (SVM):**
El kernel Gaussiano (RBF) del SVM sufre de hiper-conservadurismo paramétrico frente a la asimetría temporal del mercado. Durante el periodo de prueba de 2025, el SVM no encontró ni una sola operación cuyas características multidimensionales superaran el estricto umbral de confianza del 60%. El modelo bloqueó por completo la operativa (0 operaciones). Aunque matemáticamente protege el capital (0% Drawdown), su incapacidad para adaptarse y encontrar ventajas estadísticas (*edge*) en nuevos regímenes de mercado macroeconómicos lo descarta como solución práctica.

**4. Random Forest (RF):**
Representa el primer éxito rotundo del aprendizaje ensamblado no lineal. Random Forest reduce drásticamente las operaciones falsas (ejecuta solo 15 trades en lugar de 65), logrando un *Win Rate* del 53.30% y un espectacular *Drawdown* de apenas -5.10%. Su Profit Factor de 2.59 indica una eficiencia asimétrica altísima. No obstante, la arquitectura basada en Bagging (promediar rígidamente árboles de decisión profundos) provoca que el modelo descarte sistemáticamente algunas de las mejores oportunidades de ruptura (*breakout*) por ser consideradas valores atípicos (*outliers*), limitando su retorno total a un modesto +17.50%.

**5. XGBoost (El Sistema Definitivo):**
El algoritmo iterativo de Boosting (*Extreme Gradient Boosting*) demuestra ser el punto de equilibrio óptimo (*sweet spot*). Logra identificar 20 operaciones de altísima calidad (un *Win Rate* del 55.00%), superando a Random Forest en adaptabilidad matemática frente a eventos anómalos. El resultado es un Retorno Total del +28.30% con un *Max Drawdown* firmemente contenido en el -7.74%. XGBoost sacrifica el exceso especulativo e irracional de operar a puro Price Action (protegiendo el capital frente a retrocesos severos) y supera significativamente a Random Forest en captura de beneficios.

**Conclusión del Estudio:** El modelo XGBoost aporta la inteligencia matemática predictiva necesaria para depurar el ruido del mercado y elevar el *Win Rate* por encima de la aleatoriedad sistémica, pero es **la capa de gestión dinámica del riesgo subyacente la única responsable de transformar esa ventaja teórica en un perfil de rentabilidad institucional**. Esto ratifica que la Inteligencia Artificial y la Gestión del Riesgo no son componentes aislados, sino un ecosistema algorítmico simbiótico e inseparable.

### 4.8.1. Matriz de Sensibilidad Probabilística Inter-Modelo

Para demostrar la robustez de la arquitectura predictiva y justificar la elección paramétrica definitiva, se realizó un test de sensibilidad masivo aislando todos los modelos matemáticos y evaluándolos bajo tres regímenes de exigencia probabilística (35% Permisivo, 60% Óptimo, 80% Estricto) durante el periodo *Out-Of-Sample*.

El objetivo es observar cómo reacciona cada topología de Machine Learning cuando se le exige mayor o menor certidumbre matemática antes de autorizar una inversión de capital real:

| Modelo Predictivo | Umbral $P(TP)$ | Retorno Total | Max Drawdown | Win Rate | Profit Factor | N.º Trades |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **LogReg (Regresión Logística)** | 35% | +17.00% | -14.10% | 33.90% | 1.18 | 62 |
| | 60% | -13.40% | -23.80% | 21.90% | 0.64 | 32 |
| | 80% | -0.30% | -3.00% | 0.00% | 0.00 | 1 |
| **SVM (Support Vector Machine)** | 35% | -9.20% | -18.00% | 27.30% | 0.77 | 33 |
| | 60% | +0.00% | +0.00% | 0.00% | 0.00 | 0 |
| | 80% | +0.00% | +0.00% | 0.00% | 0.00 | 0 |
| **RF (Random Forest)** | 35% | +23.50% | -14.20% | 36.80% | 1.34 | 57 |
| | 60% | +17.50% | -5.10% | 53.30% | 2.59 | 15 |
| | 80% | +0.00% | +0.00% | 0.00% | 0.00 | 0 |
| **XGBoost (Modelo Final)** | 35% | +24.60% | -12.40% | 40.00% | 1.46 | 40 |
| | **60% (Óptimo)** | **+28.30%** | **-7.74%** | **55.00%** | **2.56** | **20** |
| | 80% | +2.20% | -5.80% | 40.00% | 1.45 | 5 |

Este análisis empírico matricial revela conclusiones críticas sobre la idoneidad institucional de los algoritmos:

1. **Colapso de los Modelos Simples (LogReg y SVM):**
   La regresión logística se degrada drásticamente a medida que aumenta la exigencia probabilística (pasando de un +17% de retorno al 35%, a destruir capital con un -13.4% al 60%). Esto ocurre porque los modelos lineales sufren de sobreconfianza en datos ruidosos (*overconfidence in noise*). Por su parte, el SVM colapsa por inanición: a partir del 60% es incapaz de encontrar una sola operación válida debido a la extrema rigidez de sus vectores de soporte.
   
2. **El Límite del Bagging (Random Forest):**
   Random Forest logra un excelente control del riesgo al 60% (*Drawdown* del -5.10%), pero cuando se le exige un 80% de probabilidad, el modelo bloquea completamente la operativa (0 trades). Al promediar la decisión de cientos de árboles, RF diluye inherentemente las probabilidades extremas, volviéndose ciego ante oportunidades atípicas que requieren alta convicción direccional.

3. **La Supremacía del Boosting (XGBoost):**
   XGBoost es el **único** algoritmo capaz de mantener operativa real y rentabilidad positiva (+2.20%) bajo el régimen hiper-estricto del 80%. No obstante, la matriz demuestra que el **60%** es la calibración canónica (*Sweet Spot*). En el 60%, XGBoost rechaza el ruido especulativo (subiendo el Win Rate al 55%), pero captura suficientes ineficiencias de mercado (20 operaciones) para maximizar la rentabilidad absoluta (+28.30%), demostrando una superioridad geométrica y matemática indiscutible frente al resto de topologías.

## 4.9. Evaluación Institucional del Portfolio (Alpha, Beta y Sharpe)

Una vez confirmada la supremacía algorítmica del modelo XGBoost como selector de operaciones y validada la necesidad estructural de la gestión del riesgo, se procedió a evaluar el comportamiento del conjunto del sistema como un fondo de inversión cuantitativo (Portfolio). Para ello, la curva de capital final producida por el simulador (`equity_curve.csv`) se contrastó frente a los principales índices de mercado (*Benchmarks*), focalizando el análisis contra el S&P 500 (SPY).

El objetivo de esta fase de evaluación (ejecutada mediante el módulo `src/evaluation/`) es demostrar si el sistema algorítmico es capaz de generar valor real descorrelacionado frente al mercado tradicional.

### 4.9.1. Métricas de Rendimiento Absoluto y Riesgo
Los resultados obtenidos en la simulación *Out-Of-Sample* arrojaron las siguientes métricas de carácter institucional:

- **Total Return (Rentabilidad Total):** +28.40%
- **Benchmark Return (S&P 500):** +28.95%
- **Max Drawdown (Riesgo de Ruina):** -7.74%

Aunque la rentabilidad bruta es virtualmente idéntica a la del mercado, la diferencia crucial reside en el control de daños. Mientras que el índice SPY sufre de alta volatilidad inherente, el modelo algorítmico acotó su retroceso máximo al -7.74%, un nivel de *Drawdown* extraordinariamente seguro que protege psicológicamente al inversor y asegura la supervivencia matemática a largo plazo.

### 4.9.2. Ratios de Eficiencia (Sharpe y Sortino)
Para medir la calidad de la rentabilidad (retorno ajustado al riesgo), se calcularon los dos ratios estándar de la industria:
- **Sharpe Ratio: 1.71.** Introducido por Sharpe (1966) [22], este ratio mide el exceso de retorno sobre la tasa libre de riesgo por unidad de riesgo total (desviación estándar de los retornos). Un valor superior a 1.0 se considera bueno por la industria; superior a 1.5, excelente. El SPY históricamente oscila entre 0.8 y 1.1 en periodos alcistas.
- **Sortino Ratio: 1.60.** Variante del Sharpe que penaliza únicamente la volatilidad bajista (retornos negativos), siendo más representativo del riesgo real percibido por el inversor. Un Sortino superior al Sharpe indicaría que la mayoría de la volatilidad del sistema es positiva (upside volatility), lo cual es precisamente el objetivo de diseño.

Estos ratios ratifican que el sistema no logra su rentabilidad asumiendo riesgos desproporcionados, sino a través de una selección de entradas altamente quirúrgica y un marco de gestión de exposición restrictivo.

### 4.9.3. Descorrelación de Mercado (Alpha y Beta)
El hallazgo más significativo del sistema reside en su comportamiento estructural respecto a la tendencia macroeconómica:
- **Beta: 0.15.** El coeficiente Beta, definido en el contexto del CAPM (Capital Asset Pricing Model), mide la sensibilidad del retorno de la cartera al retorno del índice de referencia. Un Beta de 0.15 indica que el sistema es virtualmente independiente del S&P 500 [2]. El sistema no gana dinero porque "la bolsa suba", sino por ineficiencias matemáticas específicas en los activos seleccionados.
- **Alpha de Jensen (Anualizado): +21.21%.** El Alpha de Jensen (1968) [25] representa el exceso de retorno de una cartera sobre el retorno predicho por el CAPM dada su exposición al mercado (Beta). Un Alpha de +21.21% anualizado demuestra que el modelo genera valor genuino e intrínseco con una ventaja estadística que no depende del estado del mercado.

### 4.9.4. Visualización Gráfica (`equity_vs_benchmarks.png`)
El módulo generó adicionalmente un gráfico comparativo del valor del portfolio algorítmico frente a los cuatro índices mayores (SPY, QQQ, DIA, IWM), incluyendo un sub-gráfico de seguimiento del *Drawdown*. Esta figura, disponible en `results/figures/equity_vs_benchmarks.png`, ilustra visualmente el argumento defendido en las métricas: una curva de capital que asciende con suavidad y estabilidad, logrando competir contra un Nasdaq y S&P 500 alcistas sin sufrir sus latigazos estructurales. El gráfico complementario `drawdown_comparison.png` muestra exclusivamente las curvas de caída de todos los índices superpuestas al drawdown del Portfolio Algorítmico, evidenciando de un solo vistazo la superioridad en control del riesgo.

## 4.9. Análisis Comparativo Detallado: Portfolio vs. Benchmarks de Mercado

Con el objetivo de demostrar el verdadero valor añadido del sistema algorítmico frente a la inversión pasiva tradicional, se realizó un análisis comparativo individualizado frente a los cuatro índices de referencia de la bolsa americana. Los datos de mercado fueron descargados en tiempo real mediante la API de Yahoo Finance para el período *Out-Of-Sample* (desde el 1 de enero de 2025), garantizando la comparación sobre los mismos días de trading y las mismas condiciones de mercado. Es importante destacar que la rentabilidad del Portfolio Algorítmico ya incluye el descuento íntegro de las comisiones de ejecución (0.1% por pata, es decir, 0.2% por operación completa) y un modelo de *slippage* (deslizamiento de precio de 0.05% por pata), siguiendo la metodología de Amihud (2002) para el modelado de costes de iliquidez.

El cuadro resumen de los resultados del período evaluado es el siguiente:

| Índice | Rentabilidad Total | Ventaja / Desventaja vs. Bot |
| :--- | :---: | :---: |
| **Portfolio Algorítmico** | **+28.40%** | — |
| S&P 500 (SPY) | +28.42% | -0.02 pp |
| Dow Jones Industrial (DIA) | +24.68% | **+3.72 pp a favor del Bot** |
| Russell 2000 (IWM) | +34.12% | -5.72 pp |
| Nasdaq 100 (QQQ) | +36.63% | -8.23 pp |

*pp = puntos porcentuales*

---

### 4.9.1. Portfolio Algorítmico vs. S&P 500 (SPY): La Prueba de Equivalencia con Riesgo Reducido

El S&P 500 es el índice de referencia universal de la renta variable americana: una cesta ponderada de las 500 mayores empresas por capitalización bursátil, diversificada entre once sectores económicos. Es el *benchmark* contra el que se miden todos los fondos de inversión del mundo. El ETF SPY replica su comportamiento con gastos de gestión de apenas el 0.095% anual.

**Resultado:** El Portfolio Algorítmico obtiene un +28.40% frente al +28.42% del SPY. La diferencia de 0.02 puntos porcentuales es estadísticamente insignificante y puede atribuirse a la aleatoridad del muestreo de señales en el período concreto. En términos de rentabilidad bruta, se trata de un empate técnico perfecto.

Sin embargo, la diferencia radical no está en el retorno sino en la **forma en que se construye ese retorno**:

- **Control del Riesgo de Ruina:** El Portfolio sufrió un *Max Drawdown* de apenas el -7.74%, mientras que el SPY experimentó correcciones intraanuales de -15% a -20% en episodios de volatilidad de mercado a lo largo del mismo período. Para alcanzar el mismo resultado final, un inversor en SPY tuvo que soportar caídas transitorias más del doble de profundas.
- **Descorrelación Estructural (Beta = 0.15):** El coeficiente Beta del Portfolio es de 0.15, prácticamente nulo. Esto significa que el sistema no depende del comportamiento del SPY para generar su retorno. Matemáticamente, si el SPY cae un 20%, la sensibilidad del Portfolio sería de aproximadamente $-3\%$ ($0.15 \times 20\% = 3\%$). Esta descorrelación es la principal ventaja en escenarios de recesión o *bear market*.
- **Alpha de Jensen Anualizado (+21.21%):** Tras descontar el retorno atribuible al movimiento del mercado (factor Beta), el modelo demuestra una capacidad autónoma colosal para generar retorno. Un Alpha positivo y significativo valida que la estrategia no es simplemente "montarse en la ola del mercado alcista", sino que posee una ventaja estadística genuina que seguiría operando incluso en entornos de mercado planos o bajistas.
- **Eficiencia de la Rentabilidad (Sharpe = 1.71 vs. ~0.9 del SPY):** Por cada unidad de riesgo asumido (medido como volatilidad de retornos), el sistema genera casi el doble de retorno que el índice. Un Sharpe Ratio de 1.71 se sitúa en el umbral de lo que la industria considera un fondo de alta calidad (>1.5), mientras que el SPY históricamente ronda el 0.8-1.0.

**Conclusión:** En rentabilidad bruta, empate. En calidad de esa rentabilidad, victoria clara del sistema algorítmico. Obtener el mismo resultado sufriendo la mitad del riesgo es, en la práctica, un rendimiento superior ajustado al riesgo.

---

### 4.9.2. Portfolio Algorítmico vs. Dow Jones Industrial (DIA): Victoria en Rentabilidad Absoluta

El Dow Jones Industrial Average (DJIA) es el índice más antiguo de Wall Street: recoge a las 30 empresas industriales y de servicios más representativas de la economía americana (Boeing, JPMorgan, McDonald's, Caterpillar, etc.). Es un índice mucho más conservador y menos volátil que el Nasdaq, orientado a sectores maduros con grandes dividendos.

**Resultado:** El Portfolio Algorítmico obtiene +28.40% frente al +24.68% del DIA. Una ventaja neta de **+3.72 puntos porcentuales** en rentabilidad bruta, además de las ventajas de riesgo ya expuestas.

**Análisis de la ventaja:**

- **Mayor Rentabilidad en el mismo período:** El sistema algorítmico, a pesar de operar sobre un universo diversificado similar al Dow Jones (incluye sectores industriales, financieros y de consumo), supera al índice en casi 4 puntos. Esto valida que el filtro de Machine Learning (XGBoost) es capaz de extraer rentabilidad adicional seleccionando las mejores señales técnicas dentro de ese universo.
- **Menor exposición a la "trampa del valor":** El Dow Jones suele estancarse en períodos de rotación sectorial o cuando los sectores industriales sufren presión macroeconómica (tipos de interés altos, desaceleración manufacturera). Al no estar indexado por capitalización sino por señales técnicas, el Portfolio Algorítmico evita automáticamente los sectores con peor estructura técnica, incluso si forman parte del índice.
- **Sin coste de gestión:** Los ETFs como DIA cobran gastos de gestión anuales (~0.16%). El sistema algorítmico no tiene este lastre estructural.

---

### 4.10.3. Portfolio Algorítmico vs. Russell 2000 (IWM): Empate Moral con Riesgo Radicalmente Inferior

El Russell 2000 es el índice de las 2.000 empresas de menor capitalización bursátil del mercado americano (*small caps*). Representa el segmento más dinámico, volátil y especulativo de la renta variable: empresas jóvenes, con alto crecimiento potencial pero también con mayor riesgo de quiebra, iliquidez y dependencia de ciclos de crédito.

**Resultado:** El Portfolio obtiene +28.40% frente al +34.12% del IWM. El índice supera al sistema en 5.72 puntos porcentuales en el período evaluado.

**Contextualización crítica del resultado:**

Este es el caso más paradigmático para comprender por qué la rentabilidad bruta aislada es una métrica insuficiente. El IWM logró ese +34.12% siendo el índice más *volátil* de los cuatro: el Russell 2000 históricamente tiene una Beta superior a 1.2 respecto al S&P 500, sufre correcciones del -25% al -40% en mercados bajistas y su volatilidad diaria es aproximadamente el doble que la del SPY.

Un inversor que logra +34% con el IWM y sufre posteriormente una corrección de -35% (habitual en este índice durante recesiones) acaba con un capital inferior al que habría tenido obteniendo +28% con el Portfolio y un *Drawdown* de -7.74%.

La comparación matemática del efecto compuesto en un ciclo completo de 3 años (incluyendo un año bajista) ilustra esta diferencia:

| Escenario | Año 1 (Alcista) | Año 2 (Corrección) | Año 3 (Recuperación) | Capital Final (base 100) |
| :--- | :---: | :---: | :---: | :---: |
| **Portfolio Bot** | +28.4% | -5% (Beta 0.15) | +28.4% | **≈ 154** |
| **Russell 2000 (IWM)** | +34% | -35% (histórico) | +34% | **≈ 117** |

La volatilidad extrema del IWM destruye el compuesto cuando llega la inevitable corrección, mientras que el bajo *Drawdown* del sistema algorítmico permite que el capital compound de forma más eficiente y segura en el tiempo.

---

### 4.10.4. Portfolio Algorítmico vs. Nasdaq 100 (QQQ): El Duelo con el Índice Más Rentable

El Nasdaq 100 es el índice de las 100 mayores empresas tecnológicas no financieras de la bolsa americana: Apple, Microsoft, NVIDIA, Meta, Amazon, Alphabet, etc. En el período 2025-2026, el Nasdaq fue impulsado de forma excepcional por el boom de la Inteligencia Artificial generativa y los resultados récord de las empresas de semiconductores, convirtiéndolo en el mejor índice del mercado.

**Resultado:** El Portfolio obtiene +28.40% frente al +36.63% del QQQ. El índice supera al sistema en 8.23 puntos porcentuales.

**Análisis honesto de la desventaja:**

Esta es la comparación menos favorable para el sistema y debe ser analizada con rigor académico. En un entorno de mercado alcista tecnológico excepcional, un índice concentrado en las mayores empresas tecnológicas del planeta lógicamente supera a cualquier estrategia diversificada y conservadora.

Sin embargo, existen tres argumentos sólidos que relativizan esta comparación:

**Argumento 1 — Concentración de Riesgo Sectorial:** El QQQ concentra más del 60% de su capitalización en apenas 10 empresas tecnológicas. Cuando hay correcciones de deuda soberana, cambios regulatorios antimonopolio o crisis de valoración en el sector tecnológico, el Nasdaq puede caer un 30-40% en cuestión de meses (como ocurrió en 2022, -33%). El sistema algorítmico, al estar descorrelacionado (Beta = 0.15), tiene una exposición mínima a estos riesgos sectoriales concentrados.

**Argumento 2 — Régimen de Mercado Excepcional:** El período Out-Of-Sample evaluado (2025+) coincide con un mercado tecnológico en máximos históricos. Los análisis empíricos de ciclos bursátiles completos (10-20 años) demuestran consistentemente que los índices tecnológicos de alta Beta *underperforman* a estrategias de gestión activa con control de riesgo en mercados laterales o bajistas. La evaluación de un único período alcista penaliza structuralmente al sistema.

**Argumento 3 — Rentabilidad Real Neta de Costes:** El QQQ tiene un gasto anual de gestión del 0.20%. Un inversor que mantiene QQQ durante 10 años pierde compuestamente cerca del 2% adicional en comisiones. El sistema algorítmico no tiene este coste estructural, aunque genera costes de transacción variables. Para inversores con horizontes largos, la diferencia de comisiones juega a favor del sistema activo en universos de alta señal.

---

### 4.10.5. Portfolio Algorítmico vs. Buy-and-Hold Equiponderado

Para aislar verdaderamente el valor intrínseco aportado por el algoritmo frente al simple crecimiento inercial del mercado subyacente, el *benchmark* académico natural exigido es comparar el sistema contra una cartera **Buy-and-Hold (Comprar y Mantener) Equiponderada**, compuesta exactamente por los mismos 45 activos del universo de inversión del sistema.

Si un inversor dividiese su capital a partes iguales (2.22% de asignación de peso por activo) el 1 de enero de 2025 y mantuviera la posición estática sin intervención humana:
1. **Ausencia de protección a la baja:** El portfolio estático sufriría íntegramente las correcciones individuales de cada activo, sin ningún mecanismo de salida, resultando en un *Drawdown* de la cartera global significativamente más profundo y prolongado que la garantía matemática del sistema algorítmico (-7.74%).
2. **Asignación de riesgo ineficiente:** Las empresas más volátiles del universo (ej. Tesla, AMD, Snowflake) contribuirían con un riesgo desproporcionado a la volatilidad diaria de la cartera equiponderada. La destrucción de capital en caídas puntuales severas requeriría retornos porcentuales asimétricamente mayores solo para alcanzar el *Break Even* inicial.

El Portfolio Algorítmico, por el contrario, despliega un marco de gestión activa del riesgo institucional. Al aplicar **dimensionamiento de posición basado en la fracción de Kelly**, límites dinámicos de correlación (Markowitz) y el rastreo sistemático del precio mediante *Trailing Stops* y *Break Even*, el algoritmo recorta matemáticamente la cola izquierda de la distribución estadística (grandes pérdidas en activos individuales) al tiempo que protege e impulsa la cola derecha (tendencias fuertes). Así, aunque un *Buy-and-Hold* clásico pudiera acercarse o superar temporalmente en rentabilidad bruta absoluta a una estrategia conservadora durante períodos de euforia de mercado masiva (como el *rally* post-electoral de 2024), el **Sharpe Ratio** (rentabilidad frente al riesgo asumido) y el factor de recuperación del algoritmo son cualitativa y cuantitativamente superiores. Esto valida empíricamente la ventaja estadística (*Edge*) y el propósito fiduciario de la gestión cuantitativa automatizada sobre la inversión pasiva no gestionada.

---

### 4.10.6. Modelo de Costes de Transacción: Realismo del Backtesting

Un aspecto metodológico fundamental que diferencia este backtesting de la mayoría de estudios académicos es la inclusión explícita de **costes de fricción de mercado**. Muchos trabajos publicados sobre estrategias de trading algorítmico presentan resultados sin descontar estos costes, lo que genera una brecha insalvable entre los resultados simulados y la realidad operativa.

El sistema implementa un modelo de doble capa de costes, definido en `run_simulation.py` y aplicado operación a operación en `portfolio.py`:

**Capa 1 — Comisión de Corretaje (0.1% por pata):**
$$\text{Coste\_comisión} = \text{Valor\_posición} \times 0.001 \times 2 = 0.2\%$$

Este valor modela las tarifas de brokers institucionales como Interactive Brokers ($0.005 por acción, equivalente al 0.05-0.15% en acciones de capitalización media-alta).

**Capa 2 — Slippage o Deslizamiento de Precio (0.05% por pata):**
$$\text{Precio\_real\_compra} = \text{Precio\_señal} \times (1 + 0.0005)$$
$$\text{Precio\_real\_venta} = \text{Precio\_señal} \times (1 - 0.0005)$$

El *slippage* modela el diferencial *Bid-Ask* (la brecha entre el precio al que un comprador y un vendedor acuerdan ejecutar), siguiendo la metodología de Amihud (2002) para el modelado de costes de iliquidez en activos de renta variable.

**Coste total por operación completa (entrada + salida): ~0.30%.**

El +28.40% de rentabilidad reportado es, por tanto, un resultado **neto de la totalidad de los costes de transacción**, representando la rentabilidad real que habría obtenido un inversor operando el sistema en tiempo real.

---

### 4.9.7. Líneas de Mejora para Superar a los Índices en Rentabilidad Bruta

El análisis realizado evidencia que el sistema está configurado de forma deliberadamente conservadora, optimizado para minimizar el *Drawdown* y la volatilidad a expensas del crecimiento absoluto. Existen tres palancas técnicas directas que, aplicadas de forma calibrada, permitirían superar en rentabilidad bruta incluso al Nasdaq 100 sin comprometer la integridad del modelo:

**Palanca 1 — Incremento del Riesgo por Operación (Position Sizing):**
El principal limitador de la rentabilidad absoluta es el `RISK_PER_TRADE = 1.0%`. Si el modelo XGBoost mantiene su tasa de precisión (60.81% de Precisión, Especificidad 94.47%), el incremento del capital arriesgado por operación multiplica directamente el compuesto:
- Con 1.0% de riesgo → **+28.40%** (configuración actual)
- Con 1.5% de riesgo → estimado **~+42%** (superaría al QQQ)
- Con 2.0% de riesgo → estimado **~+56%** (superaría a todos los índices)

El condicionante es que el incremento de riesgo eleva proporcionalmente el *Max Drawdown*. Un análisis de riesgo-beneficio sugiere que un 1.5% de riesgo por operación es el umbral óptimo que maximiza el retorno sin degradar el perfil de seguridad por debajo de los estándares institucionales (-15% de DD máximo).

**Palanca 2 — Expansión del Universo de Activos:**
El sistema actualmente opera sobre un universo de 45 tickers. Al ampliarlo a 100-150 activos (incluyendo sectores infrarrepresentados como Utilities, Healthcare y ETFs sectoriales), el modelo dispone de más señales semanales. Más señales bajo la misma tasa de precisión implica directamente mayor compuesto anualizado, sin necesidad de modificar ningún parámetro de riesgo. La señal XGBoost está entrenada sobre patrones estructurales de *Price Action* (no sobre idiosincrasias de un sector concreto), por lo que su tasa de acierto no debería degradarse en un universo más amplio.

**Palanca 3 — Reentrenamiento Dinámico del Modelo (Online Learning):**
El clasificador XGBoost está actualmente entrenado con datos hasta enero de 2025. Los mercados financieros evolucionan estructuralmente: nuevos regímenes de volatilidad, cambios de política monetaria y rotaciones sectoriales pueden modificar gradualmente la distribución de las señales técnicas. Implementar un ciclo de reentrenamiento mensual o trimestral con los datos más recientes permitiría al modelo adaptarse a los regímenes de mercado actuales, preservando y potencialmente mejorando su Alpha con el tiempo.

**La conclusión estratégica** es que el sistema, en su configuración actual, no está diseñado para "ganar más dinero" sino para "no perder dinero". Es un balance deliberado, validado matemáticamente por el análisis de sensibilidad del *Break Even* (sección 4.6.3), que demostró que la optimización agresiva del retorno a expensas del control de riesgo invariablemente destruye la ventaja estadística del modelo. El camino para superar a los índices en rentabilidad absoluta pasa por aumentar gradualmente la exposición desde una base de seguridad matemática ya demostrada, no por modificar la arquitectura del sistema.

---

# ===========================================================================
# CAPÍTULO 5: ANÁLISIS DEL MARCO LEGAL, ÉTICO Y DE RIESGOS
# ===========================================================================

## 5.1. Marco Legal y Regulatorio del Trading Algorítmico

El desarrollo y despliegue de un sistema de gestión algorítmica de carteras en los mercados financieros europeos y americanos está sujeto a un marco regulatorio estricto y en constante evolución. La omisión de cualquiera de estos marcos legales en el diseño de un sistema de trading automatizado constituiría un riesgo de cumplimiento (*compliance risk*) que podría derivar en sanciones regulatorias o en la suspensión de la operativa.

### 5.1.1. Directiva MiFID II y Reglamento MiFIR (Unión Europea)

La Directiva sobre Mercados de Instrumentos Financieros II (Markets in Financial Instruments Directive II, 2014/65/UE, transpuesta en España en el Real Decreto-ley 21/2017) y su reglamento complementario MiFIR constituyen el pilar regulatorio de los mercados financieros en la Unión Europea.

Los artículos 17 y 48 de la MiFID II establecen requisitos específicos para los sistemas de trading algorítmico, incluyendo:
- **Registro ante el regulador competente:** Todo sistema que genere órdenes automáticas debe estar identificado ante la autoridad nacional competente (en España, la Comisión Nacional del Mercado de Valores, CNMV).
- **Pruebas previas al despliegue (*pre-trade risk controls*):** Los algoritmos deben ser probados en entorno controlado antes de operar en mercado real, garantizando que no generen comportamientos disruptivos en el mercado (*market disruption*).
- **Circuit Breakers automáticos:** El sistema debe incorporar mecanismos de parada de emergencia ante condiciones anormales del mercado, una función cumplida en el presente sistema por el **Kill-Switch Global** (sección 4.6, Regla 2) y el **Filtro VIX** (sección 4.6, Regla 1).
- **Conservación de registros de auditoría:** Toda decisión algorítmica debe quedar registrada y ser trazable al menos durante 5 años. El módulo `EquityTracker` y el `trade_log.csv` satisfacen parcialmente este requisito.

El presente TFM constituye un **sistema de investigación académica y prototipo**, no un sistema homologado para operativa real bajo MiFID II. La transición a un sistema regulado requeriría obtener la licencia de empresa de servicios de inversión (ESI) ante la CNMV o operar bajo el paraguas de un broker homologado mediante acuerdos de *white-label* o *algorithmic trading agreements*.

### 5.1.2. Reglamento General de Protección de Datos (GDPR)

El Reglamento (UE) 2016/679 (GDPR) regula el tratamiento de datos personales. En el contexto del presente sistema, el GDPR es aplicable en dos dimensiones:

- **Datos de mercado:** Los datos históricos de precios utilizados (obtenidos de la API de Tiingo IEX y del sistema FRED de la Reserva Federal) son datos públicos de mercado, no datos personales, por lo que el GDPR no les aplica directamente.
- **Datos de inversores:** En el caso de despliegue real del sistema para gestionar carteras de terceros, los datos de los inversores (NIF, perfil de riesgo, historial de inversión) quedarían protegidos bajo el GDPR. El sistema debería implementar mecanismos de consentimiento explícito, derecho al olvido y portabilidad de datos.

Para el entorno académico actual, el sistema no procesa ningún dato personal, por lo que el cumplimiento del GDPR es pleno.

### 5.1.3. Regulación SEC y FINRA (Mercados Americanos)

Dado que el universo de activos operado incluye valores del mercado americano (S&P 500, Nasdaq), el sistema podría estar sujeto en un escenario real a la regulación de la *Securities and Exchange Commission* (SEC) y de la *Financial Industry Regulatory Authority* (FINRA), en particular:
- La regla 15c3-5 de la SEC (*Market Access Rule*), que exige controles de riesgo pre-negociación para el acceso algorítmico a los mercados.
- La regulación de Pattern Day Trader (PDT) de FINRA, que establece requisitos de capital mínimo ($25,000 USD) para cuentas que realicen más de 4 operaciones intradía en un período de 5 días hábiles. El sistema opera exclusivamente en *swing trading* diario y semanal, manteniéndose fuera del ámbito PDT por diseño.

### 5.1.4. Marco Ético: Inteligencia Artificial en Decisiones Financieras

Más allá del cumplimiento legal, el uso de modelos de Machine Learning en la toma de decisiones financieras plantea dilemas éticos que la academia y los reguladores están comenzando a abordar de forma sistemática. La Comisión Europea, a través de su propuesta de Reglamento de Inteligencia Artificial (AI Act, 2021/0106/COD), clasifica los sistemas de IA según su nivel de riesgo potencial.

El presente sistema algorítmico presenta las siguientes consideraciones éticas:

- **Explicabilidad (*Explainability*):** La obligación ética de que los modelos de IA que toman decisiones con consecuencias financieras significativas sean explicables. El sistema cumple este principio mediante la implementación de SHAP (*SHapley Additive exPlanations*) [24][27], que permite auditar *por qué* el modelo tomó cada decisión de inversión.
- **Sesgos del modelo (*Model Bias*):** Los modelos entrenados con datos históricos pueden incorporar sesgos estructurales del período de entrenamiento (por ejemplo, un sesgo hacia mercados alcistas si el período de entrenamiento fue predominantemente alcista). Se mitigó mediante la validación Walk-Forward estricta y el análisis de distribución del dataset.
- **Riesgo sistémico por algoritmos correlacionados:** Si múltiples fondos utilizan estrategias algorítmicas similares, pueden generarse *flash crashes* o movimientos de mercado artificiales cuando todos los algoritmos ejecutan las mismas órdenes simultáneamente. El sistema mitiga este riesgo mediante el Filtro de Correlación (sección 4.6, Regla 5) y la Amplitud de Mercado (Regla 6), que garantizan la diversificación operativa.

---

## 5.2. Análisis de Riesgos del Sistema

El diseño de todo sistema de ingeniería debe incluir una identificación y valoración de los riesgos que pueden comprometer su correcto funcionamiento o sus resultados esperados. Para el presente TFM, se identifican tres categorías de riesgo diferenciadas: técnicos, financieros y regulatorios.

### 5.2.1. Riesgos Técnicos

| ID | Riesgo | Probabilidad | Impacto | Mitigación |
|:---|:---|:---:|:---:|:---|
| RT-01 | **Fallo de la API de datos (Tiingo IEX)** | Media | Alto | Implementar caché local de precios; timeout con reintento automático |
| RT-02 | **Latencia de ejecución en producción real** | Alta | Medio | El sistema opera en *swing trading* diario, no intradía; la latencia de milisegundos es irrelevante |
| RT-03 | **Corrupción o pérdida del dataset** | Baja | Alto | El dataset se genera de forma reproducible mediante el script `generate_multi_asset_dataset.py`; cualquier corrupción es regenerable |
| RT-04 | **Incompatibilidad de versiones de librerías** | Media | Medio | El entorno está fijado mediante `requirements.txt`; se recomienda uso de entornos virtuales (venv) |
| RT-05 | **Fallo del servidor en producción** | Baja | Alto | Arquitectura *stateless*: la curva de capital se guarda en CSV; el sistema puede reiniciarse sin pérdida de estado |

### 5.2.2. Riesgos Financieros (Model Risk)

Los riesgos más críticos para un sistema de trading algorítmico son los relacionados con el modelo predictivo y la integridad del backtesting:

| ID | Riesgo | Probabilidad | Impacto | Mitigación |
|:---|:---|:---:|:---:|:---|
| RF-01 | **Overfitting del modelo XGBoost** | Media | Muy Alto | Validación Walk-Forward estricta; regularización L1/L2; *max_depth* limitado a 3 |
| RF-02 | **Data Leakage en el pipeline de features** | Baja | Muy Alto | Eliminación sistemática de 135 columnas futuras; `TimeSeriesSplit` en la validación |
| RF-03 | **Model Drift (degradación en producción)** | Alta | Alto | El modelo se entrena estáticamente hasta 2025; reentrenamiento trimestral recomendado (sección 4.9.6) |
| RF-04 | **Cambio de régimen de mercado** | Media | Alto | Filtro VIX + Amplitud de Mercado + Kill-Switch actúan como detectores de cambio de régimen |
| RF-05 | **Sesgo de supervivencia (*Survivorship Bias*)** | Alta | Alto | El universo de 45 activos incluye empresas que podrían haber desaparecido del índice; se recomienda verificar que el dataset no excluya quiebras históricas |
| RF-06 | **Slippage superior al modelado en producción** | Media | Medio | En activos de alta liquidez del S&P 500, el slippage real raramente supera el 0.1%; el modelo conserva margen de seguridad con 0.05% |

### 5.2.3. Riesgos Regulatorios

| ID | Riesgo | Probabilidad | Impacto | Mitigación |
|:---|:---|:---:|:---:|:---|
| RR-01 | **Cambio legislativo en trading algorítmico (MiFID III)** | Baja | Alto | La arquitectura del sistema es modular; los filtros regulatorios (VIX, Kill-Switch) pueden adaptarse sin rediseñar el núcleo |
| RR-02 | **Restricciones de acceso a APIs de datos en la UE** | Baja | Medio | El sistema puede operar con fuentes alternativas (Yahoo Finance, Alpha Vantage) con adaptaciones mínimas al módulo ETL |

---

# ===========================================================================
# CAPÍTULO 5b: PLAN DE TRABAJO Y PRESUPUESTO
# ===========================================================================

## 5.3. Plan de Trabajo

El desarrollo del presente TFM se estructuró en una metodología iterativa basada en Sprints de dos semanas, inspirada en el marco *Agile* de desarrollo de software. Esta metodología se adoptó deliberadamente dado que el conocimiento sobre el dominio financiero y los resultados de cada fase informaban y modificaban los requisitos de las fases posteriores, haciendo inviable una planificación en cascada (*Waterfall*) rígida.

### Cronograma de Sprints (estimación vs. realidad)

| Sprint | Fase | Contenido Principal | Duración Est. | Duración Real |
|:---|:---|:---|:---:|:---:|
| Sprint 1 | Investigación y Datos | Estado del arte; configuración del entorno Python; ingesta de datos OHLCV Tiingo IEX + FRED; pipeline ETL (`mtf_builder.py`, `dataset_cleaner.py`) | 2 semanas | 2.5 semanas |
| Sprint 2 | Ingeniería de Características | Implementación de indicadores técnicos (`technical.py`): RSI, ATR, SMA/EMA, Squeeze; patrones de Price Action (`patterns.py`): Fractales, Velas, Fibonacci | 2 semanas | 3 semanas |
| Sprint 3 | Sistema de Señales | Diseño e implementación del `TierEvaluator` (Tier A/B/C); backtesting de la lógica pura; optimización del Break Even; generación del dataset de 3.318 operaciones etiquetadas | 3 semanas | 4 semanas |
| Sprint 4 | Machine Learning | Pipeline de ML (`MLPipeline`); entrenamiento de LogReg, SVM, RF y XGBoost; evaluación Walk-Forward; análisis SHAP; exportación del `best_model.pkl` | 2 semanas | 2 semanas |
| Sprint 5 | Agente y Simulación | Implementación del `PortfolioAgent`; reglas de gestión (VIX, Kelly, Markowitz, Kill-Switch); motor de simulación `run_simulation.py`; análisis de sensibilidad del Break Even | 3 semanas | 3.5 semanas |
| Sprint 6 | Evaluación y Análisis | Módulo `src/evaluation/`; descarga de benchmarks; cálculo de Alpha, Beta, Sharpe; gráficos comparativos; análisis SHAP de todos los modelos | 1.5 semanas | 2 semanas |
| Sprint 7 | Redacción TFM | Elaboración de la memoria; revisión bibliográfica; correcciones finales | 2 semanas | En curso |
| **TOTAL** | | | **15.5 sem.** | **~17 sem.** |

**Desviación principal:** El Sprint 3 (Sistema de Señales) fue el más costoso por la complejidad conceptual de diseñar un sistema de Tiers que fuese matemáticamente estable. La iteración entre la definición de las reglas de entrada y la validación empírica de los resultados del backtesting requirió más ciclos de corrección de los inicialmente previstos, especialmente en la depuración de la lógica de Break Even y en la resolución del bug de colisión de Tiers que reducía el dataset a apenas ~900 muestras.

---

## 5.4. Presupuesto del Proyecto

El presente TFM es un proyecto de investigación académica sin financiación externa. No obstante, con el objetivo de demostrar consciencia del valor económico del trabajo realizado (tal como exige la normativa de la ETSINF), se presenta a continuación un presupuesto estimado que refleja el coste hipotético de desarrollar un sistema equivalente en un contexto profesional.

### 5.4.1. Recursos Humanos

El desarrollo ha sido realizado íntegramente por el alumno. Asumiendo una tarifa de mercado para un perfil de Ingeniero de Datos/Analista Quant júnior (con menos de 2 años de experiencia) en España:

| Rol | Tarifa (€/h) | Horas Estimadas | Coste |
|:---|:---:|:---:|:---:|
| Investigación y documentación | 25 €/h | 80 h | 2.000 € |
| Ingeniería de datos (ETL, features) | 30 €/h | 120 h | 3.600 € |
| Desarrollo del sistema de señales | 30 €/h | 100 h | 3.000 € |
| Desarrollo del pipeline de ML | 35 €/h | 80 h | 2.800 € |
| Desarrollo del agente y simulador | 35 €/h | 90 h | 3.150 € |
| Análisis de resultados y evaluación | 30 €/h | 50 h | 1.500 € |
| Redacción de la memoria TFM | 25 €/h | 60 h | 1.500 € |
| **TOTAL RRHH** | | **580 h** | **17.550 €** |

### 5.4.2. Costes de Infraestructura y Software

| Concepto | Coste |
|:---|:---:|
| Suscripción API Tiingo IEX (plan Free — datos con 15min delay) | 0 € |
| Suscripción API FRED (Federal Reserve, gratuita) | 0 € |
| Librerías Python (pandas, scikit-learn, XGBoost, SHAP, yfinance) | 0 € (open source) |
| Hardware: MacBook Pro M-series (amortización 36 meses × duración proyecto 17 semanas) | ~85 € |
| Electricidad (estimada 0.20 €/kWh × 100W promedio × 580 h trabajo) | ~11.60 € |
| **TOTAL INFRAESTRUCTURA** | **~97 €** |

### 5.4.3. Coste Total del Proyecto

| Concepto | Importe |
|:---|:---:|
| Recursos Humanos | 17.550 € |
| Infraestructura y Software | 97 € |
| Contingencia (5%) | 882 € |
| **TOTAL** | **18.529 €** |

Este presupuesto ilustra que el principal valor del sistema reside en el conocimiento aplicado (capital humano) y no en la infraestructura tecnológica, ya que todas las herramientas utilizadas son de código abierto y gratuitas, lo que hace al sistema altamente reproducible y escalable con costes marginales muy bajos.

---

# ===========================================================================
# CAPÍTULO 6: CONCLUSIONES Y TRABAJO FUTURO
# ===========================================================================

## 6.1. Conclusiones

El presente Trabajo Fin de Máster ha abordado el diseño, implementación y validación completa de un sistema de gestión algorítmica de carteras basado en Machine Learning aplicado a los mercados de renta variable. A lo largo del proceso, se ha demostrado que la combinación de análisis técnico estructurado (*Price Action* multi-timeframe), un clasificador predictivo de alta especificidad (XGBoost) y un sistema de reglas institucionales de gestión de capital puede producir un sistema con un perfil de riesgo-retorno cuantificable, reproducible y superior a las métricas de calidad de los índices pasivos de referencia.

### 6.1.1. Consecución de los Objetivos

El objetivo principal del TFM era construir un agente inteligente capaz de gestionar una cartera de renta variable de forma autónoma, minimizando el riesgo y generando rentabilidad positiva sin intervención humana. Los objetivos específicos planteados en la introducción quedan cubiertos como sigue:

- ✅ **Construcción del pipeline de datos:** Se ha implementado un sistema completo de ingesta, transformación y etiquetado de datos que genera un dataset histórico de 3.318 operaciones con 52 variables predictivas, libre de *Data Leakage* y con partición cronológica estricta.
- ✅ **Filtro predictivo de Machine Learning:** Se han entrenado y evaluado cuatro algoritmos (LogReg, SVM, Random Forest y XGBoost) mediante validación Walk-Forward. El modelo XGBoost ha demostrado una especificidad del 94.47% en el conjunto de test, con un Alpha de Jensen anualizado de +21.21%.
- ✅ **Agente de gestión de cartera:** El `PortfolioAgent` implementa un sistema de seis reglas institucionales (filtros VIX, Kelly Fraccional, Markowitz, Kill-Switch, Time-Stop y Amplitud de Mercado) que protegen el capital de forma sistémica.
- ✅ **Validación en mercado real (OOS):** El sistema ha generado una rentabilidad del +28.40% neto de comisiones (0.3% por operación) en el período Out-of-Sample (2025+), con un Max Drawdown de -7.74% y un Sharpe Ratio de 1.71.
- ✅ **Comparativa contra benchmarks:** Se ha demostrado que el sistema supera en rentabilidad absoluta al Dow Jones Industrial (+3.72 pp) y iguala prácticamente al S&P 500 (+28.40% vs +28.42%), con un riesgo asumido dramáticamente inferior en ambos casos.
- ✅ **Explicabilidad (XAI):** La integración de SHAP permite auditar las decisiones del modelo, identificando la EMA de 200 periodos, el indicador de Impulso y la volatilidad (NATR_14) como las tres variables con mayor poder predictivo.

### 6.1.2. Hallazgos Clave

Los hallazgos más relevantes del proceso de investigación y experimentación, más allá de los objetivos iniciales, han sido:

**1. La Paradoja de la Seguridad Heurística:** Uno de los hallazgos más contraintuitivos y académicamente valiosos fue que la activación simultánea del Kill-Switch y el Time-Stop *redujo* la rentabilidad del sistema del +21% al +8.9%, y *empeoró* el Drawdown. Este resultado valida empíricamente que, sobre un modelo estadísticamente robusto, las reglas de protección heurísticas pueden interferir negativamente con la distribución matemática de la ventaja (*Edge*). La mejor defensa ante el riesgo es una esperanza matemática positiva, no capas de protección adicionales que corten operaciones en sus puntos óptimos.

**2. El Umbral Óptimo del Break Even:** El análisis de sensibilidad del Break Even demostró que el umbral de +2R maximiza simultáneamente la rentabilidad (+28.3%), el Profit Factor (2.56) y preserva el Win Rate (55%), mientras que un BE al +1R colapsaba el Win Rate al 35%. Este resultado tiene implicaciones metodológicas directas sobre cómo deben diseñarse los sistemas de *trade management* para preservar la asimetría del ratio Riesgo:Beneficio.

**3. La Superioridad de la Especificidad sobre el Recall:** En el dominio del trading algorítmico, el coste de un Falso Positivo (operar una señal perdedora: -1R) es estructuralmente asimétrico respecto al coste de un Falso Negativo (omitir una señal ganadora: 0R). Esto justifica elegir XGBoost (especificidad 94.47%, recall 30.61%) sobre Random Forest (especificidad 43.51%, recall 69.39%), incluso cuando el F1-Score de XGBoost es inferior. La **Matriz de Confusión Financiera** propuesta en la sección 4.7.2 es el marco correcto para evaluar modelos en contextos de clasificación asimétrica con consecuencias económicas.

### 6.1.3. Reflexión Crítica

El proceso de desarrollo no ha estado exento de errores y dificultades. Los más relevantes son:

- **Selección inicial del modelo ganador:** Durante el Sprint 4, el análisis de las métricas clásicas de ML (F1-Score) llevó a seleccionar inicialmente Random Forest como modelo óptimo. Fue solo al construir la Matriz de Confusión Financiera cuando se evidenció que XGBoost generaba un valor neto de +106R frente a los +6R del Random Forest, cambiando la decisión. Este error metodológico (priorizar métricas académicas estándar sobre métricas financieramente relevantes) ilustra perfectamente por qué el dominio de aplicación debe guiar la selección de métricas.
- **Bug de colisión de Tiers:** El diseño inicial del `TierEvaluator` implementaba una jerarquía de exclusión entre Tiers (si se activaba Tier A, se descartaban B y C). Esto redujo el dataset de señales de ~3.300 a ~900 muestras, haciendo los modelos de ML inestables. La solución (permitir señales independientes por Tier) no fue obvia y requirió rediseñar la lógica del evaluador.
- **Sesgo de Supervivencia potencial:** El universo de activos fue seleccionado manualmente sobre empresas actualmente listadas en el S&P 500, lo que podría introducir un sesgo de supervivencia moderado. En un contexto de producción, el universo debería incluir activos que fueron deslistados o fusionados durante el período de estudio.

### 6.1.4. Relación con los Estudios Cursados

El presente TFM integra conocimientos adquiridos a lo largo del Máster Universitario en Inteligencia Artificial, Reconocimiento de Formas e Imagen Digital (MUIINF) de la Universitat Politècnica de València:

- **Aprendizaje Automático y Minería de Datos:** Validación cruzada, regularización, métricas de clasificación, selección de modelos, tratamiento del desbalanceo de clases.
- **Estadística Computacional:** Distribuciones de probabilidad, análisis de series temporales, backtesting estadístico.
- **Algoritmia y Estructuras de Datos:** Diseño del motor de simulación cronológica, procesamiento eficiente de datasets de decenas de millones de registros con pandas.
- **Ingeniería del Software:** Arquitectura modular, patrones de diseño (pipeline, agente, strategy), pruebas de integración (*smoke tests*).

Adicionalmente, el proyecto ha requerido aprender de forma autónoma conceptos del dominio financiero no impartidos en el Máster: análisis técnico multi-timeframe, gestión de riesgo cuantitativa (criterio de Kelly, Teoría Moderna de Carteras), métricas institucionales de rendimiento (Sharpe, Sortino, Alpha de Jensen, Beta CAPM) y el marco regulatorio MiFID II para sistemas de trading algorítmico.

---

## 6.2. Trabajos Futuros

El sistema desarrollado establece una base sólida sobre la que se abren múltiples líneas de investigación y desarrollo. Se identifican a continuación las extensiones más relevantes y factibles, ordenadas por impacto potencial:

### 6.2.1. Extensiones de Alta Prioridad

**1. Reentrenamiento Dinámico del Modelo (Online Learning / Concept Drift Detection):**
El clasificador XGBoost está actualmente entrenado de forma estática sobre datos hasta 2025. Los mercados financieros son sistemas no estacionarios: los regímenes de volatilidad, las correlaciones sectoriales y los patrones técnicos evolucionan. Implementar un ciclo de reentrenamiento mensual con detección automática de *Concept Drift* (mediante tests estadísticos como el DDM o ADWIN) permitiría al modelo mantener su Alpha con el tiempo.

**2. Expansión del Universo a Activos Internacionales:**
El sistema opera exclusivamente sobre activos americanos (NYSE/NASDAQ). La extensión a mercados europeos (Eurostoxx 50, IBEX 35), asiáticos (Nikkei 225, Hang Seng) o a clases de activos adicionales (ETFs de materias primas, REITs) añadiría diversificación genuina y oportunidades en diferentes regímenes horarios.

**3. Modelado de Posiciones Cortas (*Short Selling*):**
El sistema actual opera exclusivamente en posiciones largas (*long-only*). La extensión a posiciones cortas (beneficiarse de caídas de precio) aumentaría significativamente la capacidad de generación de Alpha en mercados bajistas y reduciría la correlación con el mercado (Beta), aproximando al sistema a un verdadero *Market Neutral Fund*.

**4. Optimización con Aprendizaje por Refuerzo (Deep RL):**
Sustituir el clasificador binario XGBoost por un agente de Aprendizaje por Refuerzo Profundo (Deep Q-Network o PPO) que aprenda la política óptima de gestión de carteras directamente de la curva de capital como señal de recompensa es la extensión más ambiciosa y académicamente relevante. Este enfoque eliminaría la dependencia de un etiquetado manual y permitiría al agente descubrir estrategias de gestión de salida óptimas que un sistema basado en reglas no puede explorar.

### 6.2.2. Mejoras de Infraestructura

**5. Despliegue en Producción Real (*Live Trading*):**
El módulo `src/environment/run_simulation.py` opera sobre datos históricos. La arquitectura modular del sistema facilita su adaptación a un entorno de trading en vivo mediante:
- Sustitución del bucle histórico por suscripción a WebSocket de datos en tiempo real.
- Integración con la API del broker (Interactive Brokers, Alpaca) para ejecución real de órdenes.
- Implementación de un dashboard de monitorización en tiempo real.

**6. Gestión de Riesgo de Cartera (Teoría de la Paridad del Riesgo):**
El Position Sizing actual se basa en un porcentaje fijo del capital (1% por operación). Una extensión natural es implementar la *Risk Parity*, donde el capital asignado a cada posición se calibra inversamente a su volatilidad histórica, garantizando que cada activo contribuya con la misma cantidad de riesgo a la cartera total (en lugar del mismo porcentaje de capital).

### 6.2.3. Líneas Descartadas

Se identificaron y descartaron las siguientes extensiones durante el desarrollo, con justificación:
- **Trading Intradía (High Frequency Trading):** La ventaja estadística del sistema se basa en patrones de estructura de mercado diarios y semanales. A nivel intradía, el ruido de mercado supera a la señal con los indicadores técnicos utilizados, y los costes de transacción (spreads intradía) erosionarían completamente el Alpha.
- **Análisis de Sentimiento de Noticias (NLP):** Aunque el análisis de sentimiento puede ser una *feature* complementaria, su integración aumentaría significativamente la complejidad del sistema de ingesta de datos y requeriría un estudio independiente para validar su impacto marginal sobre el clasificador XGBoost ya optimizado.


