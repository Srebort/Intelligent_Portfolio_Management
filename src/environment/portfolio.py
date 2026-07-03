"""
Module for tracking portfolio state.
"""

class Portfolio:
    """
    Maintains the state of capital, available cash, open positions, and applies trading commissions.
    """
    
    def __init__(self, initial_capital: float = 100000.0, commission_rate: float = 0.001):
        """
        Initializes the portfolio.
        
        Args:
            initial_capital (float): Starting capital for the simulation.
            commission_rate (float): Commission rate applied per trade (e.g., 0.1%).
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}
        self.commission_rate = commission_rate

    def execute_trade(self, ticker: str, quantity: float, price: float, side: str):
        """
        Executes a trade and updates cash, positions, and accounts for commissions.
        
        Args:
            ticker (str): The asset ticker.
            quantity (float): Number of shares/units.
            price (float): Execution price.
            side (str): 'BUY' or 'SELL'.
        """
        if quantity <= 0:
            return

        trade_value = quantity * price
        commission = trade_value * self.commission_rate

        if side.upper() == 'BUY':
            total_cost = trade_value + commission
            if self.cash >= total_cost:
                self.cash -= total_cost
                self.positions[ticker] = self.positions.get(ticker, 0.0) + quantity
            else:
                # Log o manejar falta de liquidez
                pass

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
