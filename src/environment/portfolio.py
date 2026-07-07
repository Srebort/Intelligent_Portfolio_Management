"""
Módulo de gestión de la Billetera Virtual.

Registra el estado del capital, las posiciones abiertas y aplica comisiones reales.
Incorpora un modelo de Slippage (deslizamiento de precio) e Illiquidity Penalty
para simular condiciones realistas de ejecución de órdenes.

Referencia (Slippage/Liquidez):
    Amihud, Y. (2002). Illiquidity and stock returns: cross-section and time-series
    effects. Journal of Financial Markets, 5(1), 31-56.
"""

class Portfolio:
    """
    Maintains the state of capital, available cash, open positions, and applies trading commissions.
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission_rate: float = 0.001,
        slippage_rate: float = 0.0005,
        min_volume_threshold: int = 100000,
    ):
        """
        Initializes the portfolio.

        Args:
            initial_capital (float): Starting capital for the simulation.
            commission_rate (float): Commission rate applied per trade (e.g., 0.1%).
            slippage_rate (float): Estimated price slippage per trade due to bid-ask spread
                                   (e.g., 0.05% = 0.0005). Applied to the execution price.
            min_volume_threshold (int): Minimum average daily volume required to execute a
                                        trade. Orders for illiquid assets are rejected.
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.min_volume_threshold = min_volume_threshold

    def execute_trade(
        self,
        ticker: str,
        quantity: float,
        price: float,
        side: str,
        avg_volume: int = None,
    ):
        """
        Executes a trade and updates cash, positions, accounting for commissions
        and market slippage (Amihud, 2002).

        El slippage penaliza el precio de ejecución real:
            - En compras  (BUY):  precio_real = precio * (1 + slippage_rate)
            - En ventas  (SELL): precio_real = precio * (1 - slippage_rate)

        Args:
            ticker (str): The asset ticker.
            quantity (float): Number of shares/units.
            price (float): Theoretical execution price (e.g., close price).
            side (str): 'BUY' or 'SELL'.
            avg_volume (int): Average daily volume of the asset. If provided and below
                              min_volume_threshold, the trade is rejected (illiquidity filter).
        """
        if quantity <= 0:
            return

        # --- Filtro de Liquidez (Illiquidity Penalty) ---
        if avg_volume is not None and avg_volume < self.min_volume_threshold:
            print(
                f"[Portfolio] ❌ Orden {side} {ticker} rechazada: volumen medio "
                f"{avg_volume:,} < mínimo {self.min_volume_threshold:,} (activo ilíquido)."
            )
            return

        # --- Aplicar Slippage al precio de ejecución real ---
        if side.upper() == 'BUY':
            execution_price = price * (1 + self.slippage_rate)
        else:
            execution_price = price * (1 - self.slippage_rate)

        trade_value = quantity * execution_price
        commission = trade_value * self.commission_rate

        if side.upper() == 'BUY':
            total_cost = trade_value + commission
            if self.cash >= total_cost:
                self.cash -= total_cost
                self.positions[ticker] = self.positions.get(ticker, 0.0) + quantity
            else:
                print(f"[Portfolio] ⚠️  Sin liquidez para comprar {quantity} acciones de {ticker}.")

        elif side.upper() == 'SELL':
            # Verificamos si tenemos suficientes acciones (con una pequeña tolerancia por float)
            current_qty = self.positions.get(ticker, 0.0)
            if current_qty > 0:
                # Si vendemos más de lo que tenemos, lo ajustamos al máximo que tenemos
                if quantity > current_qty:
                    quantity = current_qty
                    trade_value = quantity * price
                    commission = trade_value * self.commission_rate
                
                total_proceeds = trade_value - commission
                self.cash += total_proceeds
                self.positions[ticker] -= quantity
                
                # Si vendemos todo (o queda un residuo float muy pequeño), borramos la posición
                if self.positions[ticker] < 1e-6:
                    del self.positions[ticker]
        
    def get_portfolio_value(self, current_prices: dict) -> float:
        """
        Calculates the total value of the portfolio (cash + positions value).
        
        Args:
            current_prices (dict): Latest prices for the held assets.
            
        Returns:
            float: Total portfolio value.
        """
        positions_value = 0.0
        for ticker, quantity in self.positions.items():
            # Si no hay precio actualizado, asumimos 0 o podríamos ignorarlo
            price = current_prices.get(ticker, 0.0)
            positions_value += quantity * price
            
        return self.cash + positions_value
