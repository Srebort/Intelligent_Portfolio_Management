"""
Módulo de Gestión Dinámica de Riesgo (Risk Manager).

Implementa la lógica de Position Sizing y cálculo algorítmico de Stop Loss
basándose en el Tier asignado por el TierEvaluator y la volatilidad actual
del activo medida por el ATR.

Filosofía del módulo:
    El capital NO se distribuye de forma equitativa entre todas las operaciones.
    Se pondera en función de la calidad matemática del patrón (Tier), de modo
    que se arriesga más en confluencias de alta probabilidad y menos en
    confirmaciones tardías o de menor calidad.

Asignación de Capital Máximo en Riesgo por Tier:
    - Tier A* : 2.0% del capital total (Probabilidad Extrema)
    - Tier A  : 1.5% del capital total (Probabilidad Alta)
    - Tier B  : 1.0% del capital total (Probabilidad Media)
    - Tier C  : 0.5% del capital total (Probabilidad Baja)

Fórmula de Stop Loss:
    SL = Mínimo Local - (0.5 × ATR_14)

Fórmula de Position Sizing:
    Riesgo por Acción = Precio de Entrada - SL
    Nº Acciones = (Capital × %Tier) / Riesgo por Acción
"""

import logging
import pandas as pd
import numpy as np

logger = logging.getLogger("RiskManager")


class RiskManager:
    """
    Calcula los parámetros exactos de cada operación: Stop Loss, tamaño de
    posición y capital en riesgo, garantizando que una pérdida nunca supere
    el porcentaje máximo predefinido para el Tier de la señal.
    """

    def __init__(self, capital_inicial: float = 100000.0):
        """
        Inicializa el Risk Manager con el capital disponible de la cartera.

        Args:
            capital_inicial (float): Capital total disponible en la cartera ($).
                                     Por defecto 100,000$.
        """
        self.capital = capital_inicial

        # Mapa de asignación de riesgo máximo por Tier
        # El porcentaje representa la fracción máxima del capital total que
        # se puede perder si la operación alcanza el Stop Loss.
        self.tier_risk_allocation = {
            "A*": 0.020,  # 2.0% del capital total
            "A":  0.015,  # 1.5% del capital total
            "B":  0.010,  # 1.0% del capital total
            "C":  0.005,  # 0.5% del capital total
        }

    def calculate_trade_parameters(self, row: pd.Series) -> dict:
        """
        Calcula los parámetros completos de la operación para una fila del DataFrame.

        Pasos internos:
            1. Valida que la fila tiene un Tier asignado y datos suficientes.
            2. Calcula el Stop Loss: SL = low - 0.5 * ATR_14
            3. Calcula el riesgo monetario máximo según el Tier.
            4. Divide para obtener el número de acciones a comprar.
            5. Aplica el filtro anti-apalancamiento: si el capital requerido
               supera el total disponible, ajusta las acciones al máximo posible.

        Args:
            row (pd.Series): Una fila del DataFrame que incluya al menos las
                             columnas 'Tier', 'close', 'low' y 'ATR_14'.

        Returns:
            dict: Diccionario con los parámetros de la operación:
                  - tier, precio_entrada, stop_loss, riesgo_por_accion,
                    riesgo_monetario_max, num_acciones, capital_requerido,
                    porcentaje_cartera_usado.
                  Devuelve un dict vacío si el Tier no es válido o faltan datos.
        """
        # Validar Tier
        tier = row.get("Tier")
        if pd.isna(tier) or tier not in self.tier_risk_allocation:
            return {}

        # Extraer datos necesarios
        precio_entrada = row.get("close")
        minimo_local   = row.get("low")
        atr            = row.get("ATR_14")

        if pd.isna(precio_entrada) or pd.isna(minimo_local) or pd.isna(atr):
            logger.warning("Faltan datos de precio o ATR para calcular el riesgo.")
            return {}

        # 1. Stop Loss: mínimo de la vela - margen de volatilidad dinámica (0.5 * ATR)
        stop_loss = minimo_local - (0.5 * atr)

        # Validar coherencia matemática del SL
        if stop_loss >= precio_entrada:
            logger.warning(
                "SL calculado (%.2f) >= precio entrada (%.2f). Operacion descartada.",
                stop_loss, precio_entrada,
            )
            return {}

        # 2. Riesgo monetario máximo permitido para este Tier
        porcentaje_riesgo = self.tier_risk_allocation[tier]
        riesgo_monetario  = self.capital * porcentaje_riesgo

        # 3. Riesgo por acción (distancia de invalidación de la tesis)
        riesgo_por_accion = precio_entrada - stop_loss

        # 4. Número de acciones (fracción fija dinámica)
        num_acciones = int(riesgo_monetario // riesgo_por_accion)

        # 5. Filtro anti-apalancamiento: no comprar más de lo que permite el capital
        #    sin necesitar margen. Tope máximo = Capital / Precio de Entrada.
        max_acciones_posibles = int(self.capital // precio_entrada)
        if num_acciones > max_acciones_posibles:
            logger.info(
                "Ajustando acciones de %d a %d para evitar apalancamiento.",
                num_acciones, max_acciones_posibles,
            )
            num_acciones = max_acciones_posibles

        capital_requerido = num_acciones * precio_entrada

        return {
            "tier":                   tier,
            "precio_entrada":         round(precio_entrada, 2),
            "stop_loss":              round(stop_loss, 2),
            "riesgo_por_accion":      round(riesgo_por_accion, 2),
            "riesgo_monetario_max":   round(riesgo_monetario, 2),
            "num_acciones":           num_acciones,
            "capital_requerido":      round(capital_requerido, 2),
            "porcentaje_cartera_usado": round((capital_requerido / self.capital) * 100, 2),
        }

    def get_stop_losses(
        self,
        precio_entrada: float,
        low_vela: float,
        atr: float,
        fractal_low: float,
        sma_50: float,
        sma_200: float,
        impulso: float,
    ) -> dict:
        """
        Calcula todos los niveles de Stop Loss válidos para una señal.

        Un SL solo se incluye si está estrictamente por debajo del precio de entrada.
        Los SL basados en SMA se descartan si la media está por encima del precio.

        Stop Loss disponibles:
            SL1 - ATR Vela      : min_vela_entrada - 0.5 * ATR  [Siempre válido]
            SL2 - Fractal       : min_fractal_previo - 0.5 * ATR
            SL3 - SMA 50        : SMA_50 - 0.5 * ATR             [Si SMA50 < entrada]
            SL4 - SMA 200       : SMA_200 - 0.5 * ATR            [Si SMA200 < entrada]
            SL5 - Fib 38.2%     : entrada - 0.382 * impulso
            SL6 - Fib 50.0%     : entrada - 0.500 * impulso
            SL7 - Fib 61.8%     : entrada - 0.618 * impulso

        Args:
            precio_entrada (float): Precio de cierre de la vela de entrada.
            low_vela       (float): Precio mínimo de la vela de entrada.
            atr            (float): ATR_14 en el momento de la señal.
            fractal_low    (float): Precio mínimo del último fractal de soporte.
            sma_50         (float): Valor de la SMA 50 en el momento de la señal.
            sma_200        (float): Valor de la SMA 200 en el momento de la señal.
            impulso        (float): Distancia desde el fractal hasta el precio de entrada.

        Returns:
            dict: {nombre_sl: precio_sl} solo con los SL geométricamente válidos.
        """
        sl = {}

        # SL1: ATR de la vela (siempre válido)
        sl1 = low_vela - 0.5 * atr
        if sl1 < precio_entrada:
            sl["SL1_ATR"] = round(sl1, 4)

        # SL2: Fractal de soporte previo
        if not np.isnan(fractal_low):
            sl2 = fractal_low - 0.5 * atr
            if sl2 < precio_entrada:
                sl["SL2_Fractal"] = round(sl2, 4)

        # SL3: SMA 50 (solo si está por debajo del precio)
        if not np.isnan(sma_50) and sma_50 < precio_entrada:
            sl3 = sma_50 - 0.5 * atr
            if sl3 < precio_entrada:
                sl["SL3_SMA50"] = round(sl3, 4)

        # SL4: SMA 200 (solo si está por debajo del precio)
        if not np.isnan(sma_200) and sma_200 < precio_entrada:
            sl4 = sma_200 - 0.5 * atr
            if sl4 < precio_entrada:
                sl["SL4_SMA200"] = round(sl4, 4)

        # SL5, SL6, SL7: Retrocesos Fibonacci
        if not np.isnan(impulso) and impulso > 0:
            for ratio, nombre in [
                (0.382, "SL5_Fib382"),
                (0.500, "SL6_Fib500"),
                (0.618, "SL7_Fib618"),
            ]:
                sl_fib = precio_entrada - ratio * impulso
                if sl_fib < precio_entrada:
                    sl[nombre] = round(sl_fib, 4)

        return sl

    def get_take_profits(
        self,
        precio_entrada: float,
        riesgo_por_accion: float,
        impulso: float,
    ) -> dict:
        """
        Calcula todos los niveles de Take Profit para una señal.

        Take Profits disponibles:
            TP1 - 2R            : entrada + 2.0 * riesgo_por_accion
            TP2 - 2.5R          : entrada + 2.5 * riesgo_por_accion
            TP3 - 3R            : entrada + 3.0 * riesgo_por_accion
            TP4 - Fib Ext 100%  : entrada + 1.000 * impulso
            TP5 - Fib Ext 161.8%: entrada + 1.618 * impulso
            TP6 - Fib Ext 261.8%: entrada + 2.618 * impulso

        Args:
            precio_entrada     (float): Precio de cierre de la vela de entrada.
            riesgo_por_accion  (float): Distancia en $ entre entrada y SL de referencia.
            impulso            (float): Distancia desde el fractal hasta el precio de entrada.

        Returns:
            dict: {nombre_tp: precio_tp}
        """
        tp = {}

        # TPs aritméticos (siempre válidos)
        for multiplier, nombre in [
            (2.0, "TP1_2R"),
            (2.5, "TP2_25R"),
            (3.0, "TP3_3R"),
        ]:
            tp[nombre] = round(precio_entrada + multiplier * riesgo_por_accion, 4)

        # TPs Fibonacci (requieren impulso válido)
        if not np.isnan(impulso) and impulso > 0:
            for ratio, nombre in [
                (1.000, "TP4_Fib100"),
                (1.618, "TP5_Fib1618"),
                (2.618, "TP6_Fib2618"),
            ]:
                tp[nombre] = round(precio_entrada + ratio * impulso, 4)

        return tp

