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

La motivación principal de este proyecto radica en la necesidad de cerrar esta brecha tecnológica. Se pretende demostrar que, mediante la integración de Análisis Multi-Timeframe (MTF) riguroso, algoritmos de detección de *Price Action* y modelos de Machine Learning (como XGBoost o Random Forest), es posible construir un sistema de gestión cuantitativa de carteras autónomo, robusto y estadísticamente rentable, accesible sin requerir la infraestructura de un fondo de inversión.

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

El sistema propuesto en este TFM adopta un enfoque radicalmente distinto: el clasificador
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

Tabla 3.1: Universo de activos del sistema
| Sector | Activos representativos |
|---|---|
| Índices y ETFs de referencia | SPY, QQQ, DIA, IWM |
| Tecnología y Semiconductores | AAPL, MSFT, NVDA, GOOGL, META, AMD |
| Ciberseguridad y Nube | CRWD, PANW, SNOW, PLTR |
| Servicios Financieros | JPM, V, GS |
| Salud y Biotecnología | LLY, NVO, ABBV |
| Energía e Industria | XOM, CVX, CAT |
| Activos de cobertura | TLT, GLD, SLV |

La selección de renta variable americana como universo principal responde a tres criterios:
(i) alta liquidez, que garantiza la ejecutabilidad de las órdenes sin impacto de mercado
significativo; (ii) disponibilidad de datos históricos de calidad desde 2010; y (iii)
eficiencia del mercado norteamericano, que facilita la comparación con el benchmark de Buy
and Hold sobre el S&P 500.


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

Esta feature es condición necesaria para el Tier A* (máxima probabilidad) del sistema.


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

Este patrón es fundamental en el Tier A* (junto con la divergencia RSI), ya que representa
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


## 4.4 Sistema de Evaluación de Señales: TierEvaluator

El `TierEvaluator` clasifica cada vela del dataset en uno de los cuatro Tiers de probabilidad
(A*, A, B, C) o en la categoría nula (sin señal), aplicando un conjunto de condiciones lógicas
en cascada de mayor a menor exigencia.

### 4.4.1 Filtro Base Multi-Timeframe (Obligatorio para todos los Tiers)

Antes de evaluar cualquier patrón de Price Action, el sistema verifica la confluencia de las
tres temporalidades:

  Condición_4H:  close_4H > SMA_200_4H              (tendencia operativa alcista)
  Condición_1D:  close_1D > SMA_200_1D              (tendencia diaria alcista)
  Condición_1W:  slope_SMA_200_1W > 0               (tendencia macro positiva)
  filtro_base   = Condición_4H AND Condición_1D AND Condición_1W

Si el filtro base falla, la señal NO se genera. Esta es la primera línea de defensa del
sistema contra operar en contra de la tendencia institucional.

### 4.4.2 Tier A* — Probabilidad Extrema

Representa el setup de máxima confluencia: barrido de liquidez + retroceso Fibonacci + divergencia RSI.

  Tier_A_star = filtro_base
                AND is_bullish_wick_reclaim = 1
                AND |dist_fib_retr_618| < 1.5%
                AND |dist_SMA_200| < 5%
                AND is_bullish_divergence = 1

### 4.4.3 Tier A — Probabilidad Alta

Retroceso profundo sin barrido de liquidez explícito:
  Tier_A = filtro_base AND |dist_fib_retr_618| < 1.5% AND |dist_SMA_200| < 5% AND NOT Tier_A_star

### 4.4.4 Tier B — Probabilidad Media

Doble suelo sobre soporte algorítmico:
  Tier_B = filtro_base AND is_support_fractal = 1 AND is_bullish_wick_reclaim = 1
           AND |dist_SMA_200| < 5% AND NOT Tier_A_star AND NOT Tier_A

### 4.4.5 Tier C — Probabilidad Baja

Confirmación tardía vía ruptura de resistencia:
  Tier_C = filtro_base AND close > último_fractal_de_resistencia
           AND close_{t-1} <= resistencia_{t-1} AND NOT Tier_A_star AND NOT Tier_A AND NOT Tier_B

### 4.4.6 Asignación de Capital por Tier

El capital en riesgo máximo se determina en función del Tier asignado:

Tabla 4.1: Asignación de capital por Tier
| Tier | Capital en riesgo | Justificación |
|---|---|---|
| A* | 2.0% del capital total | Máxima confluencia de señales independientes |
| A  | 1.5% del capital total | Alta probabilidad sin confirmación de divergencia |
| B  | 1.0% del capital total | Probabilidad media, doble suelo confirmado |
| C  | 0.5% del capital total | Confirmación tardía, mayor riesgo de falsa ruptura |

Esta asignación asimétrica implementa el principio de Kelly Criterion generalizado: se
arriesga más capital cuando la probabilidad de éxito es mayor y menos cuando es menor,
maximizando el crecimiento esperado del capital a largo plazo.


## 4.5 El Filtro Predictivo (Machine Learning)

A pesar de la rigurosidad matemática del sistema de Tiers basado en *Price Action* descrito en la sección anterior, los mercados financieros presentan un alto grado de estocasticidad que produce, inevitablemente, un porcentaje significativo de señales falsas (operaciones perdedoras que tocan el Stop Loss). Para mitigar este problema, se ha integrado una capa adicional de inteligencia artificial: un filtro predictivo de Machine Learning.

El objetivo de este modelo no es predecir genéricamente la dirección del mercado, sino actuar como un "segundo juez". Toma como entrada exclusivamente las operaciones que el `TierEvaluator` ya ha validado, analiza sus características técnicas (features) en el momento exacto de la señal, y predice la probabilidad matemática de que dicha operación alcance el Take Profit antes que el Stop Loss.

### 4.5.1 Prevención de Data Leakage y Pipeline de Datos

El desafío técnico más crítico en el modelado financiero predictivo es la prevención de la filtración de información futura (*data leakage*). Dado que el dataset fue etiquetado matemáticamente por el backtester utilizando precios futuros reales (hit de TP o SL), estas columnas revelan directamente el resultado de la operación.

Para evitar esto, se ha implementado la clase `MLPipeline` (`src/models/ml_pipeline.py`), que realiza las siguientes operaciones de seguridad de forma automatizada:
1.  **Purga de variables del futuro:** Elimina programáticamente 81 columnas del dataset, incluyendo cualquier variable terminada en `_precio`, `_vela` o `_hit`, conservando estrictamente los indicadores técnicos e índices de *Price Action* calculados hasta el momento de la entrada.
2.  **Partición Cronológica (TimeSeriesSplit):** A diferencia de un problema de clasificación tradicional donde los datos pueden particionarse aleatoriamente (K-Fold tradicional), los datos financieros poseen dependencia temporal. El pipeline utiliza `TimeSeriesSplit` para evaluar los modelos: entrena con el pasado y testea en el futuro.
3.  **Escalado:** Las variables numéricas son estandarizadas mediante `StandardScaler` (ajustado exclusivamente sobre los datos de entrenamiento) para garantizar un aprendizaje estable en algoritmos sensibles a la magnitud, como Support Vector Machines.

### 4.5.2 Modelos Predictivos Base y Control del Sobreajuste

Al trabajar con series temporales financieras y, especialmente en las fases iniciales del desarrollo con un dataset limitado, el riesgo de sobreajuste (*overfitting*) es severo. Si se permite que el modelo memorice el "ruido" del mercado, su capacidad de generalización en operaciones futuras reales se desploma.

Por ello, se establecieron tres modelos base (líneas base o *baselines*) fuertemente regularizados:
-   **Regresión Logística (LogReg):** Configurada con una regularización L2 agresiva (`C=0.05`). Actúa como el baseline lineal del sistema.
-   **Support Vector Machine (SVM):** Utilizando un kernel Gaussiano (RBF) con un margen de regularización suave (`C=0.5`).
-   **Random Forest (RF):** Ensamblaje de 50 árboles de decisión, severamente limitados en profundidad (`max_depth=3`) y con exigencia de al menos 3 muestras por hoja (`min_samples_leaf=3`) para forzar la abstracción.

### 4.5.3 Clasificador Avanzado: XGBoost

Como modelo final, se implementó `TradeSelectorXGB` basado en XGBoost (*eXtreme Gradient Boosting*). Este algoritmo construye árboles de decisión secuencialmente para minimizar los errores de sus predecesores. 

Para su configuración financiera, se aplicó una regularización combinada: `reg_alpha=0.5` (Lasso) para forzar dispersión reduciendo a cero los pesos de características irrelevantes, y `reg_lambda=1.0` (Ridge) para penalizar ponderaciones excesivas. Adicionalmente, el ratio de aprendizaje se redujo a `learning_rate=0.05` y la profundidad máxima a 3, obligando al modelo a aprender patrones sutiles y robustos en lugar de particularidades del dataset.

### 4.5.4 Evaluación de Modelos y Análisis de Variables (Feature Importance)

Los cuatro modelos compitieron directamente sobre el conjunto de Test. Se priorizó el uso del **F1-Score** como métrica principal, dada su capacidad matemática para penalizar tanto los Falsos Positivos (señales aceptadas que resultan en pérdidas) como los Falsos Negativos (señales ganadoras descartadas).

**Resultados Comparativos en el Test Set (Operaciones Futuras):**
-   **Random Forest:** Logró el mejor desempeño global, con un F1-Score de **0.778**, un Accuracy del 87.9% y tan solo 3 Falsos Positivos.
-   **SVM:** Obtuvo un F1-Score de **0.737** (Accuracy 84.8%).
-   **XGBoost:** Mostró un Recall perfecto del 100% (no descartó ninguna operación ganadora), pero con mayor número de Falsos Positivos, resultando en un F1-Score de **0.640**.
-   **Regresión Logística:** Como era de esperar dada la no-linealidad del mercado, quedó rezagada con un F1-Score de **0.522**.

**Análisis de Importancia de Variables (Feature Importance):**
El análisis paramétrico de los modelos basados en árboles reveló qué características técnicas del mercado tienen mayor poder predictivo. Destacan significativamente:
1.  **Volatilidad (`ATR_14`):** El parámetro dominante absoluto. Entornos de alta volatilidad aumentan drásticamente la probabilidad matemática de que el ruido del mercado alcance el Stop Loss antes que el Take Profit.
2.  **Momentum (`impulso`):** El tamaño relativo de la vela de entrada indica la convicción institucional detrás del movimiento.
3.  **Sobreextensión del Precio (`dist_SMA_50` y `dist_SMA_200`):** La distancia porcentual del precio respecto a las medias móviles principales resultó crítica, confirmando empíricamente el principio de "reversión a la media" del mercado.

Concluido el análisis, el modelo ganador (Random Forest) fue exportado estáticamente mediante la librería *Joblib* junto con su escalador métrico, listo para ser desplegado como el núcleo de decisión probabilística del Agente Gestor de Cartera en la fase de simulación en vivo.


## 4.6 El Agente Gestor de Cartera: `PortfolioAgent`

Una vez que el filtro predictivo de Machine Learning aprueba una señal técnica, la decisión de inversión entra en su fase más crítica: ¿cuánto capital comprometer, cuándo salir y qué hacer si el mercado entra en un régimen adverso? Estas responsabilidades recaen sobre el `PortfolioAgent`, implementado en `src/models/agent_logic.py`.

Este módulo constituye la **capa de orquestación superior** del sistema. Su función es actuar como el "director de inversiones" que coordina la inteligencia del modelo predictivo con las reglas matemáticas de gestión de capital, aplicando un conjunto de filtros defensivos basados en la literatura académica de gestión cuantitativa de carteras.

El diseño del Agente responde directamente a los tres desafíos identificados en el Capítulo 3:
1.  **Selección de señales de calidad:** El Agente aplica un umbral probabilístico estricto para filtrar las predicciones del Random Forest.
2.  **Gestión dinámica del riesgo:** Se conecta con el `RiskManager` para calcular matemáticamente el número exacto de acciones a comprar.
3.  **Control de exposición global:** Implementa múltiples reglas de rebalanceo para controlar la concentración sectorial, la correlación entre activos y el régimen macro del mercado.

### 4.6.1 Filtro Probabilístico de la Inteligencia Artificial

El primer filtro que aplica el `PortfolioAgent` es la comprobación de la probabilidad de éxito predicha por el modelo. Cada señal técnica validada por el `TierEvaluator` es escalada mediante el `scaler.pkl` y pasada al modelo `best_model.pkl` para obtener su probabilidad de alcanzar el Take Profit (`P(TP)`).

El umbral de aceptación está configurado por defecto en **P(TP) ≥ 0.75** (75%). Este valor fue determinado empíricamente durante la evaluación del Sprint 4: por debajo de este umbral, el número de Falsos Positivos en el conjunto de test aumenta significativamente, erosionando la rentabilidad esperada de la cartera.

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

La asignación fija de capital por Tier (Tier A* = 2%, Tier A = 1.5%, etc.) presenta una limitación: no se adapta a los periodos en los que el modelo de Machine Learning está en una racha de errores. El Criterio de Kelly, formalizado matemáticamente por Kelly (1956) y ampliado por MacLean, Thorp y Ziemba (2011), proporciona la fracción óptima del capital a invertir en función de la tasa de aciertos y el ratio ganancia/pérdida esperado:

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
    ├── [4a] ¿P(TP) < 75%?        → Rechazar señal
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


# ===========================================================================
# CAPÍTULO 7: REFERENCIAS BIBLIOGRÁFICAS
# ===========================================================================

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
