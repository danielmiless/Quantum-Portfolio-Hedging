# src/integration/broker_interface.py
"""
Broker Interface for Alpaca Trading
Updated for latest Alpaca API (v0.26+)
"""

import os
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from enum import Enum

try:
    import alpaca_trade_api as tradeapi
    from alpaca_trade_api.entity import Order, Position
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    logging.warning("Alpaca API not available")


class OrderType(Enum):
    """Order types for trading."""
    BUY = "buy"
    SELL = "sell"


class OrderSide(Enum):
    """Order sides."""
    LONG = "long"
    SHORT = "short"


class BrokerInterface:
    """Base broker interface."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.connected = False
    
    def connect(self) -> bool:
        """Connect to broker."""
        raise NotImplementedError
    
    def disconnect(self) -> bool:
        """Disconnect from broker."""
        raise NotImplementedError
    
    def get_market_data(self, ticker: str) -> Dict:
        """Get market data for ticker."""
        raise NotImplementedError
    
    def submit_order(self, ticker: str, quantity: float, order_type: OrderType) -> Optional[Dict]:
        """Submit an order."""
        raise NotImplementedError
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        raise NotImplementedError
    
    def get_positions(self) -> List[Dict]:
        """Get current positions."""
        raise NotImplementedError
    
    def get_account(self) -> Dict:
        """Get account information."""
        raise NotImplementedError


class AlpacaInterface(BrokerInterface):
    """Alpaca broker interface - UPDATED FOR CURRENT API."""
    
    def __init__(self, api_key: Optional[str] = None, secret_key: Optional[str] = None, 
                 base_url: Optional[str] = None, paper: bool = True):
        """Initialize Alpaca interface."""
        super().__init__()
        
        self.api_key = api_key or os.getenv('ALPACA_API_KEY')
        self.secret_key = secret_key or os.getenv('ALPACA_SECRET_KEY')
        self.base_url = base_url or os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
        self.paper = paper
        self.api = None
        
        if not self.api_key or not self.secret_key:
            raise ValueError("Alpaca API credentials not provided")
    
    def connect(self) -> bool:
        """Connect to Alpaca."""
        try:
            if not ALPACA_AVAILABLE:
                self.logger.error("Alpaca API not available")
                return False
            
            self.api = tradeapi.REST(
                self.api_key,
                self.secret_key,
                self.base_url,
                api_version='v2'
            )
            
            # Test connection by getting account
            account = self.api.get_account()
            self.connected = True
            self.logger.info(f"Connected to Alpaca: {self.base_url}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Alpaca: {e}")
            self.connected = False
            return False
    
    def disconnect(self) -> bool:
        """Disconnect from Alpaca."""
        try:
            self.connected = False
            self.logger.info("Disconnected from Alpaca")
            return True
        except Exception as e:
            self.logger.error(f"Error disconnecting from Alpaca: {e}")
            return False
    
    def get_market_data(self, ticker: str) -> Dict:
        """Get latest market data for a ticker (FIXED for current API)."""
        try:
            if not self.connected:
                return {'last_price': 0, 'error': 'Not connected'}
            
            # Use get_latest_bar() which works with current Alpaca API
            bar = self.api.get_latest_bar(ticker)
            
            if bar:
                return {
                    'last_price': float(bar.c),  # close price
                    'bid': float(bar.o),  # open price (as bid reference)
                    'ask': float(bar.c),  # close price (as ask reference)
                    'open': float(bar.o),
                    'high': float(bar.h),
                    'low': float(bar.l),
                    'volume': int(bar.v),
                    'timestamp': bar.t
                }
            else:
                return {'last_price': 0, 'error': f'No data for {ticker}'}
                
        except Exception as e:
            self.logger.error(f"Error getting market data for {ticker}: {e}")
            return {'last_price': 0, 'error': str(e)}
    
    def submit_order(self, ticker: str, quantity: float, order_type: OrderType, 
                     order_class: str = "simple", time_in_force: str = "day") -> Optional[Dict]:
        """Submit an order to Alpaca."""
        try:
            if not self.connected:
                self.logger.error("Not connected to Alpaca")
                return None
            
            # Determine side
            side = "buy" if order_type == OrderType.BUY else "sell"
            
            # Submit order
            order = self.api.submit_order(
                symbol=ticker,
                qty=quantity,
                side=side,
                type="market",  # Market order
                time_in_force=time_in_force,
                order_class=order_class
            )
            
            self.logger.info(f"Order submitted: {order.id} - {side.upper()} {quantity} {ticker}")
            
            return {
                'order_id': order.id,
                'symbol': order.symbol,
                'qty': order.qty,
                'side': order.side,
                'status': order.status,
                'created_at': order.created_at,
                'filled_avg_price': order.filled_avg_price
            }
            
        except Exception as e:
            self.logger.error(f"Failed to submit order: {e}")
            return None
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        try:
            if not self.connected:
                self.logger.error("Not connected to Alpaca")
                return False
            
            self.api.cancel_order(order_id)
            self.logger.info(f"Order cancelled: {order_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    def get_positions(self) -> List[Dict]:
        """Get current positions."""
        try:
            if not self.connected:
                self.logger.error("Not connected to Alpaca")
                return []
            
            positions = self.api.list_positions()
            result = []
            
            for position in positions:
                result.append({
                    'symbol': position.symbol,
                    'qty': float(position.qty),
                    'side': position.side,
                    'avg_fill_price': float(position.avg_fill_price),
                    'current_price': float(position.current_price),
                    'market_value': float(position.market_value),
                    'unrealized_pl': float(position.unrealized_pl),
                    'unrealized_plpc': float(position.unrealized_plpc),
                    'lastday_price': float(position.lastday_price)
                })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to get positions: {e}")
            return []
    
    def get_account(self) -> Dict:
        """Get account information."""
        try:
            if not self.connected:
                self.logger.error("Not connected to Alpaca")
                return {}
            
            account = self.api.get_account()
            
            return {
                'account_number': account.account_number,
                'status': account.status,
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value),
                'pattern_day_trader': account.pattern_day_trader,
                'trading_blocked': account.trading_blocked,
                'account_blocked': account.account_blocked,
                'multiplier': account.multiplier,
                'buying_power': float(account.buying_power),
                'long_market_value': float(account.long_market_value),
                'short_market_value': float(account.short_market_value),
                'equity': float(account.equity),
                'last_equity': float(account.last_equity),
                'daytrade_buying_power': float(account.daytrade_buying_power)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get account: {e}")
            return {}
    
    def get_orders(self, status: str = "open") -> List[Dict]:
        """Get orders with specific status."""
        try:
            if not self.connected:
                self.logger.error("Not connected to Alpaca")
                return []
            
            orders = self.api.list_orders(status=status)
            result = []
            
            for order in orders:
                result.append({
                    'order_id': order.id,
                    'symbol': order.symbol,
                    'qty': float(order.qty),
                    'side': order.side,
                    'status': order.status,
                    'type': order.order_type,
                    'time_in_force': order.time_in_force,
                    'created_at': order.created_at,
                    'filled_qty': float(order.filled_qty),
                    'filled_avg_price': order.filled_avg_price
                })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to get orders: {e}")
            return []
    
    def get_order(self, order_id: str) -> Optional[Dict]:
        """Get specific order details."""
        try:
            if not self.connected:
                self.logger.error("Not connected to Alpaca")
                return None
            
            order = self.api.get_order(order_id)
            
            return {
                'order_id': order.id,
                'symbol': order.symbol,
                'qty': float(order.qty),
                'side': order.side,
                'status': order.status,
                'type': order.order_type,
                'time_in_force': order.time_in_force,
                'created_at': order.created_at,
                'filled_qty': float(order.filled_qty),
                'filled_avg_price': order.filled_avg_price
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get order {order_id}: {e}")
            return None
    
    def get_bars(self, ticker: str, timeframe: str = "1Day", limit: int = 100) -> List[Dict]:
        """Get historical bars."""
        try:
            if not self.connected:
                self.logger.error("Not connected to Alpaca")
                return []
            
            bars = self.api.get_bars(ticker, timeframe, limit=limit)
            result = []
            
            for bar in bars:
                result.append({
                    'timestamp': bar.t,
                    'open': float(bar.o),
                    'high': float(bar.h),
                    'low': float(bar.l),
                    'close': float(bar.c),
                    'volume': int(bar.v)
                })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to get bars for {ticker}: {e}")
            return []


class BrokerFactory:
    """Factory for creating broker instances."""
    
    @staticmethod
    def create_broker(broker_name: str, **kwargs) -> Optional[BrokerInterface]:
        """Create broker instance."""
        if broker_name.lower() == 'alpaca':
            try:
                return AlpacaInterface(**kwargs)
            except Exception as e:
                logging.error(f"Failed to create Alpaca broker: {e}")
                return None
        else:
            logging.error(f"Unknown broker: {broker_name}")
            return None


if __name__ == "__main__":
    # Load .env file first
    from dotenv import load_dotenv
    from pathlib import Path
    
    # Find and load .env
    env_path = Path(__file__).parent.parent.parent / '.env'
    load_dotenv(env_path)
    
    # Test the broker interface
    logging.basicConfig(level=logging.INFO)
    
    broker = BrokerFactory.create_broker('alpaca')
