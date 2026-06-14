# Estrategia de Trading Algorítmico — Sistema Multi-Timeframe Long-Only

**Proyecto:** TFM — Gestión Dinámica del Riesgo y Rebalanceo de Carteras  
**Autor:** Sergio Rebollo  
**Versión:** 1.0  
**Última actualización:** Junio 2026

---

## 1. Filosofía y Descripción General

La estrategia desarrollada es un **sistema algorítmico Long-Only (solo compras)** basado en la convergencia de múltiples temporalidades (*Multi-Timeframe Analysis*). El sistema busca **reincorporarse a tendencias macroeconómicas alcistas** validadas en temporalidades altas (1 Semana y 1 Día) mediante **entradas de alta precisión** en la temporalidad operativa (4 Horas), utilizando principios de Price Action y reversión a la media.

> [!IMPORTANT]
> La estrategia es Long-Only. No abre posiciones en corto. Toda la lógica parte de la premisa de que la tendencia macro es alcista y que se busca el mejor punto de incorporación al movimiento.

---

## 2. Universo de Activos

El universo de trading cubre 50+ activos de renta variable americana agrupados por sectores:

| Sector | Ejemplos |
|---|---|
| Índices y Benchmarks | SPY, QQQ, DIA, IWM |
| Big Tech y Semiconductores | AAPL, MSFT, NVDA, GOOGL, META, AMD |
| Ciberseguridad y Nube | CRWD, PANW, SNOW, PLTR |
| Finanzas y Cripto-Proxies | JPM, V, COIN, MSTR |
| Salud y Biotecnología | LLY, NVO, ABBV, ISRG |
| Energía e Industria | XOM, CVX, CAT |
| Cobertura y Refugio | TLT, GLD, SLV |

Datos descargados desde **Tiingo IEX** (precios OHLCV intradiarios) y **FRED** (indicadores macroeconómicos).

---

## 3. Estructura Multi-Timeframe (MTF)

El sistema opera con una jerarquía de tres temporalidades con roles distintos:

```
1 SEMANA  (1W)  ─── Filtro de Tendencia Macro (largo plazo)
                     ¿El precio está por encima de SMA 200?
                     ¿La pendiente de SMA 200 es positiva?
        │
        ▼
1 DÍA    (1D)  ─── Filtro de Tendencia de Medio Plazo
                     ¿El precio está por encima de SMA 200?
                     ¿Hay divergencia alcista en RSI?
        │
        ▼
4 HORAS  (4H)  ─── Temporalidad Operativa (entradas)
                     Detección de patrones de Price Action
                     Cálculo del Tier de probabilidad
                     Cálculo de Stop Loss y Position Sizing
```

> [!NOTE]
> Las temporalidades superiores (1D, 1W) actúan como **filtros de permiso**. Si el contexto macro no es alcista, el sistema NO genera señales de compra aunque en 4H aparezcan patrones técnicos válidos.

---

## 4. Universo de Variables (Feature Engineering)

El precio bruto se transforma en características estacionarias y relativas que eliminan la dependencia de la escala del precio.

### 4.1 Filtros de Tendencia y Distancias

| Variable | Fórmula | Uso en el sistema |
|---|---|---|
| `SMA_50`, `SMA_200` | Media simple de N cierres | Filtro direccional |
| `EMA_9`, `EMA_21` | Media exponencial de N cierres | Momentum rápido |
| `dist_SMA_50` | `(close - SMA_50) / SMA_50 × 100` | Detección de sobreextensión |
| `dist_SMA_200` | `(close - SMA_200) / SMA_200 × 100` | Filtro de cercanía al soporte macro |
| `slope_SMA_50` | `(SMA_50[t] - SMA_50[t-5]) / SMA_50[t-5] × 100` | Momentum de tendencia corto plazo |
| `slope_SMA_200` | `(SMA_200[t] - SMA_200[t-5]) / SMA_200[t-5] × 100` | Momentum de tendencia macro |

### 4.2 Osciladores y Divergencias

| Variable | Descripción |
|---|---|
| `RSI_14` | Índice de Fuerza Relativa, período 14. Sobrecompra > 70, sobreventa < 30 |
| `is_bullish_divergence` | **Binaria (0/1)**. Detecta mínimos más bajos en precio con mínimos más altos en RSI (pérdida de momentum bajista). Ventana: 20 períodos |

### 4.3 Volatilidad

| Variable | Descripción |
|---|---|
| `ATR_14` | Average True Range: mide el rango medio de movimiento (incluye gaps entre sesiones) |
| `NATR_14` | `ATR_14 / close × 100`. Versión porcentual normalizada para comparar activos de distintos precios |
| `atr_squeeze` | `ATR_14 / SMA(ATR_14, 20)`. Ratio de compresión. Valores < 1.0 indican volatilidad comprimida (posible breakout) |

### 4.4 Price Action y Geometría del Precio

| Variable | Descripción |
|---|---|
| `is_resistance_fractal` | **Binaria**. Máximo local rodeado de 2 máximos más bajos a cada lado (Fractales de Bill Williams) |
| `is_support_fractal` | **Binaria**. Mínimo local rodeado de 2 mínimos más altos a cada lado |
| `is_hammer` | **Binaria**. Vela con mecha inferior ≥ 2× el cuerpo y mecha superior < 10% del rango |
| `is_inverted_hammer` | **Binaria**. Vela con mecha superior ≥ 2× el cuerpo y mecha inferior < 10% del rango |
| `is_bullish_wick_reclaim` | **Binaria**. La mecha inferior representa más del 60% del rango total de la vela (rechazo agresivo del precio) |
| `is_bearish_wick_reclaim` | **Binaria**. La mecha superior representa más del 60% del rango total de la vela |
| `dist_fib_retr_382` | Distancia % del precio al nivel de retroceso Fibonacci 38.2% (calculado sobre la ventana rolling de 50 períodos) |
| `dist_fib_retr_618` | Distancia % del precio al nivel de retroceso Fibonacci 61.8% |
| `dist_fib_ext_up_382` | Distancia % del precio al objetivo de extensión alcista Fibonacci 38.2% |
| `dist_fib_ext_up_618` | Distancia % del precio al objetivo de extensión alcista Fibonacci 61.8% |

### 4.5 Contexto Macroeconómico (FRED)

| Variable | Serie FRED | Interpretación |
|---|---|---|
| `FEDFUNDS` | Tipo de interés Fed | Coste del dinero |
| `CPIAUCSL` | IPC (inflación) | Poder adquisitivo |
| `UNRATE` | Desempleo | Salud del mercado laboral |
| `T10Y2Y` | Spread 10Y-2Y | Indicador de recesión (negativo = inversión de curva) |
| `VIXCLS` | VIX | Índice del miedo. Valores > 30 = pánico de mercado |

---

## 5. Sistema de Scoring de Entradas (Tiers)

El agente clasifica cada oportunidad en cuatro niveles jerárquicos según la **confluencia matemática de señales**.

### Condición Base (Obligatoria para todos los Tiers)
```
close_4H > SMA_200_4H    Y    close_1D > SMA_200_1D    Y    slope_SMA_200_1W > 0
```
Si no se cumple la condición base, no se genera ninguna señal independientemente de los patrones de Price Action.

---

### Tier A* — Probabilidad Extrema (Setup Óptimo)

**Contexto:** El precio corrige agresivamente rompiendo temporalmente soportes clave para capturar la liquidez acumulada bajo ellos (falsa ruptura / barrido de stops).

**Condiciones (TODAS simultáneas):**
```
filtro_base                              = True
is_bullish_wick_reclaim                  = 1    (barrido de liquidez con rechazo)
dist_fib_retr_618.abs()                  < 1.5% (rechaza el nivel Fib 61.8%)
dist_SMA_200.abs()                       < 5%   (cerca del soporte macro)
is_bullish_divergence                    = 1    (divergencia RSI confirmada)
```

**Lógica:** El precio barre los stops por debajo de un soporte (wick largo), toca el retroceso del 61.8% de Fibonacci, y el RSI confirma que los bajistas están perdiendo fuerza. Es el patrón de máxima probabilidad.

---

### Tier A — Probabilidad Alta

**Contexto:** Retroceso ordenado y profundo a favor de la tendencia principal, sin barrido de liquidez agresivo.

**Condiciones (TODAS simultáneas):**
```
filtro_base                              = True
dist_fib_retr_618.abs()                  < 1.5% (visita el nivel Fib 61.8%)
dist_SMA_200.abs()                       < 5%   (cerca del soporte macro)
NOT Tier A*                              (no es una falsa ruptura)
```

**Diferencia con A*:** No exige la divergencia RSI ni el wick de rechazo agresivo. El precio simplemente llega al Fib 61.8% cerca de la media y repunta.

---

### Tier B — Probabilidad Media (Doble Suelo)

**Contexto:** Re-testeo de una zona de valor previamente validada. El mercado regresa al nivel de soporte del último mínimo fractal.

**Condiciones:**
```
filtro_base                              = True
is_support_fractal                       = 1    (respeta un suelo algorítmico previo)
is_bullish_wick_reclaim                  = 1    (rechazo confirmado)
dist_SMA_200.abs()                       < 5%   (no muy alejado de la media)
NOT Tier A* y NOT Tier A
```

**Lógica:** El precio forma un segundo suelo en la misma zona que el primero (Doble Suelo), confirmando que ese nivel es un soporte relevante. La mecha de rechazo valida que los compradores están activos en esa zona.

---

### Tier C — Probabilidad Baja (Breakout)

**Contexto:** Operativa de confirmación tardía. El precio supera la resistencia que limitaba el retroceso.

**Condiciones:**
```
filtro_base                              = True
close > último fractal de resistencia    (breakout confirmado)
close[t-1] <= resistencia[t-1]           (cruce en esta vela, no era un breakout previo)
NOT Tier A*, A o B
```

**Riesgo:** Las entradas en breakout tienen mayor riesgo de falsa ruptura y peor ratio riesgo/beneficio. Por eso se le asigna el menor capital en riesgo.

---

## 6. Gestión Dinámica del Riesgo (Position Sizing)

El capital **nunca** se distribuye de forma equitativa. Se pondera en función de la calidad matemática de la señal.

### 6.1 Asignación de Capital Máximo en Riesgo

| Tier | % Riesgo del Capital Total | Ejemplo en $100,000 |
|---|---|---|
| A* | 2.0% | $2,000 en riesgo máximo |
| A | 1.5% | $1,500 en riesgo máximo |
| B | 1.0% | $1,000 en riesgo máximo |
| C | 0.5% | $500 en riesgo máximo |

### 6.2 Cálculo del Stop Loss

El Stop Loss se coloca por debajo del mínimo local con un margen de volatilidad para absorber el ruido de mercado y evitar *whipsaws*:

$$SL = \text{Low}_{\text{vela}} - (0.5 \times \text{ATR}_{14})$$

> Para el **Tier C (Breakout)**, el SL se sitúa por debajo del nivel de resistencia recién quebrado o de la SMA de referencia.

### 6.3 Cálculo del Tamaño de Posición

$$\text{Riesgo por Acción} = \text{Precio de Entrada} - SL$$

$$\text{Nº Acciones} = \frac{\text{Capital Total} \times \%_{\text{Tier}}}{\text{Riesgo por Acción}}$$

**Filtro anti-apalancamiento:** Si el capital requerido supera el total disponible (operación de contado), el número de acciones se limita automáticamente al máximo financiable sin margen:

$$\text{Nº Acciones}_{\text{máx}} = \left\lfloor \frac{\text{Capital Total}}{\text{Precio de Entrada}} \right\rfloor$$

### 6.4 Ejemplo Práctico

```
Capital: $100,000  |  Señal: Tier A (1.5%)  |  Activo: SPY
─────────────────────────────────────────────────────────
Precio de Entrada : $678.91
Low de la vela    : $674.50
ATR_14            : $4.50

Stop Loss = 674.50 - (0.5 × 4.50) = $672.25
Riesgo por Acción = 678.91 - 672.25 = $6.66
Riesgo Monetario  = 100,000 × 1.5% = $1,500
Nº Acciones       = 1,500 / 6.66   = 225 (ajustado a 147 por filtro anti-apalancamiento)
Capital Requerido = 147 × 678.91   = $99,800 (99.8% de la cartera)
```

---

## 7. Implementación en Código

```
src/
├── data/
│   ├── tiingo_loader.py        → Descarga OHLCV (Tiingo IEX)
│   ├── fred_loader.py          → Descarga macro (FRED)
│   └── data_merger.py          → Fusiona market + macro con ffill
│
├── features/
│   ├── technical.py            → RSI, ATR, NATR, SMAs, EMAs,
│   │                             dist_SMA, slope_SMA, atr_squeeze,
│   │                             is_bullish_divergence
│   └── patterns.py             → Fractales, Hammers, Wick Reclaims, Fibonacci
│
├── models/
│   ├── tier_evaluator.py       → Scoring de Tiers (A*, A, B, C)
│   └── xgboost_filter.py       → Filtro XGBoost (pendiente)
│
└── environment/
    ├── risk_manager.py         → Stop Loss + Position Sizing
    ├── portfolio.py            → Gestión de cartera (pendiente)
    └── backtester.py           → Motor de backtest (pendiente)
```

---

## 8. Parámetros Configurables (`config/settings.yaml`)

| Parámetro | Valor por defecto | Descripción |
|---|---|---|
| `initial_capital` | 100,000$ | Capital inicial de la simulación |
| `timeframe_primary` | `4Hour` | Temporalidad operativa principal |
| `start_date` | 2018-01-01 | Inicio del histórico |
| `end_date` | 2026-05-01 | Fin del histórico |
| `fib_tolerance` | 1.5% | Proximidad mínima al nivel Fib 61.8% para activar señal |
| `max_sma_dist` | 5.0% | Distancia máxima a SMA 200 para permitir entrada |
| `atr_sl_multiplier` | 0.5 | Multiplicador ATR para el Stop Loss |

---

## 9. Flujo Completo del Sistema

```
1. Descargar datos (tiingo_loader + fred_loader)
        ↓
2. Fusionar y alinear temporalmente (data_merger)
        ↓
3. Calcular features técnicas + macro (technical.py + patterns.py)
        ↓
4. TierEvaluator → clasifica cada vela en A*, A, B, C o None
        ↓
5. [Pendiente] Filtro XGBoost → veta o confirma cada señal del Tier
        ↓
6. RiskManager → calcula SL, Nº Acciones y Capital en Riesgo
        ↓
7. [Pendiente] Backtester → simula resultado real de cada operación
        ↓
8. [Pendiente] Métricas → Win Rate, Profit Factor, Max Drawdown, Sharpe Ratio
```

### ¿Por qué XGBoost va DESPUÉS del TierEvaluator?

El XGBoost **no es un pre-filtro genérico** de activos. Es una **segunda capa de confirmación estadística** que actúa exclusivamente sobre las señales ya generadas por el sistema de Tiers.

| Rol | XGBoost ANTES del Tier | XGBoost DESPUÉS del Tier ✅ |
|---|---|---|
| Función | Pre-filtro de activos | Confirmación de señales |
| Entrada del modelo | Todas las velas de todos los activos | Solo las velas donde Tier ≠ None |
| Pregunta que responde | "¿Subirá este activo?" | "¿Esta señal concreta será rentable?" |
| Entrenamiento | Sobre movimiento de precio genérico | Sobre los resultados reales del Backtester |
| Precisión | Baja (no sabe del patrón de entrada) | Alta (entrenado con el contexto de la señal) |

**Flujo de decisión combinado:**
```
Tier A detectado en NVDA (4H)
        ↓
XGBoost evalúa: dado este Tier A con estos valores de RSI,
dist_SMA, atr_squeeze, divergencia... ¿cuál es P(ganancia)?
        ↓
Si P(ganancia) > umbral (ej. 0.60) → RiskManager ejecuta
Si P(ganancia) < umbral            → Señal descartada
```

Este diseño es más riguroso académicamente porque el dataset de entrenamiento
del XGBoost tiene etiquetas reales: cada señal del Backtester queda marcada
como `resultado = ganancia | pérdida`, permitiendo un aprendizaje supervisado
directamente sobre la calidad del sistema de Tiers.
