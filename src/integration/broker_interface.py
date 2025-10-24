# src/integration/broker_interface.py
"""
Broker API integration with environment variable support and fixed imports
"""

import numpy as np
import pandas as pd
import sys
import os
from typing import Dict, List, Tuple, Optional, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import requests
import threading
import time
import warnings

# Fix imports by adding project paths
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
src_dir = current_dir.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Import centralized config
try:
    from config import Config
except ImportError:
    # Fallback if config not available
    class Config:
        ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
        ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
        ALPACA_BASE_URL = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
        IB_HOST = os.getenv('IB_HOST', '127.0.0.1')
        IB_PORT = int(os.getenv('IB_PORT', '7497'))
        IB_CLIENT_ID = int(os.getenv('IB_CLIENT_ID', '1'))

# Optional broker-specific imports
try:
    import ib_insync
    IB_AVAILABLE = True
except ImportError:
    IB_AVAILABLE = False
    warnings.warn("Interactive Brokers API not available. Install with: pip install ib_insync")

try:
    import alpaca_trade_api as tradeapi
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    warnings.warn("Alpaca API not available. Install with: pip install alpaca-trade-api")


class OrderType(Enum):
    """Order types supported across brokers."""
    MARKET = "MKT"
    LIMIT = "LMT" 
    STOP = "STP"
    STOP_LIMIT = "STP LMT"
    MARKET_ON_CLOSE = "MOC"
    LIMIT_ON_CLOSE = "LOC"


class OrderStatus(Enum):
    """Order status lifecycle."""
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIAL"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    ERROR = "ERROR"


@dataclass
class Order:
    """Universal order representation."""
    symbol: str
    quantity: float
    order_type: OrderType
    side: str  # 'BUY' or 'SELL'
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = 'DAY'
    order_id: Optional[str] = None
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = None
    filled_at: Optional[datetime] = None
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    commission: float = 0.0
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass 
class Position:
    """Position representation."""
    symbol: str
    quantity: float
    avg_cost: float
    market_price: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float = 0.0
    
    @property
    def total_pnl(self) -> float:
        return self.realized_pnl + self.unrealized_pnl


class BrokerInterface(ABC):
    """Abstract base class for broker integrations."""
    
    def __init__(self, credentials: Dict[str, str]):
        self.credentials = credentials
        self.connected = False
        self.account_info = {}
        
    @abstractmethod
    def connect(self) -> bool:
        """Connect to broker API."""
        pass
        
    @abstractmethod
    def disconnect(self):
        """Disconnect from broker API."""
        pass
        
    @abstractmethod
    def submit_order(self, order: Order) -> str:
        """Submit order and return order ID."""
        pass
        
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel order by ID."""
        pass
        
    @abstractmethod
    def get_order_status(self, order_id: str) -> OrderStatus:
        """Get order status."""
        pass
        
    @abstractmethod
    def get_positions(self) -> List[Position]:
        """Get current positions."""
        pass
        
    @abstractmethod
    def get_account_info(self) -> Dict:
        """Get account information."""
        pass
        
    @abstractmethod
    def get_market_data(self, symbol: str) -> Dict:
        """Get current market data for symbol."""
        pass


class InteractiveBrokersInterface(BrokerInterface):
    """Interactive Brokers API integration using environment variables."""
    
    def __init__(self, host: Optional[str] = None, 
                 port: Optional[int] = None,
                 client_id: Optional[int] = None):
        """
        Initialize IB interface.
        
        Args:
            host: IB Gateway/TWS host (uses Config.IB_HOST if None)
            port: IB Gateway/TWS port (uses Config.IB_PORT if None)
            client_id: Client ID (uses Config.IB_CLIENT_ID if None)
        """
        # Use environment variables as defaults
        self.host = host or Config.IB_HOST
        self.port = port or Config.IB_PORT
        self.client_id = client_id or Config.IB_CLIENT_ID
        
        super().__init__({})
        self.ib = None
        
        if not IB_AVAILABLE:
            raise ImportError("ib_insync not available. Install with: pip install ib_insync")
            
    def connect(self) -> bool:
        """Connect to Interactive Brokers TWS/Gateway."""
        try:
            self.ib = ib_insync.IB()
            self.ib.connect(self.host, self.port, clientId=self.client_id)
            self.connected = True
            self.account_info = self._get_account_summary()
            print(f"✅ Connected to Interactive Brokers at {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to IBKR: {e}")
            return False
            
    def disconnect(self):
        """Disconnect from IBKR."""
        if self.ib and self.connected:
            self.ib.disconnect()
            self.connected = False
            
    def submit_order(self, order: Order) -> str:
        """Submit order to IBKR."""
        if not self.connected:
            raise RuntimeError("Not connected to broker")
            
        # Create IBKR contract
        contract = ib_insync.Stock(order.symbol, 'SMART', 'USD')
        
        # Create IBKR order
        ib_order = ib_insync.Order()
        ib_order.action = order.side
        ib_order.totalQuantity = abs(order.quantity)
        ib_order.orderType = order.order_type.value
        ib_order.tif = order.time_in_force
        
        if order.price:
            ib_order.lmtPrice = order.price
        if order.stop_price:
            ib_order.auxPrice = order.stop_price
            
        # Submit order
        trade = self.ib.placeOrder(contract, ib_order)
        order_id = str(trade.order.orderId)
        
        # Update order object
        order.order_id = order_id
        order.status = OrderStatus.SUBMITTED
        
        return order_id
        
    def cancel_order(self, order_id: str) -> bool:
        """Cancel order."""
        try:
            self.ib.cancelOrder(int(order_id))
            return True
        except Exception as e:
            print(f"Failed to cancel order {order_id}: {e}")
            return False
            
    def get_order_status(self, order_id: str) -> OrderStatus:
        """Get order status."""
        try:
            trades = self.ib.trades()
            for trade in trades:
                if str(trade.order.orderId) == order_id:
                    ib_status = trade.orderStatus.status
                    return self._map_ib_status(ib_status)
            return OrderStatus.ERROR
        except Exception:
            return OrderStatus.ERROR
            
    def _map_ib_status(self, ib_status: str) -> OrderStatus:
        """Map IB status to our OrderStatus."""
        mapping = {
            'Submitted': OrderStatus.SUBMITTED,
            'Filled': OrderStatus.FILLED,
            'PartiallyFilled': OrderStatus.PARTIALLY_FILLED,
            'Cancelled': OrderStatus.CANCELLED,
            'Inactive': OrderStatus.REJECTED,
            'PendingSubmit': OrderStatus.PENDING,
            'PendingCancel': OrderStatus.PENDING
        }
        return mapping.get(ib_status, OrderStatus.ERROR)
        
    def get_positions(self) -> List[Position]:
        """Get current positions."""
        positions = []
        for position in self.ib.positions():
            if position.position != 0:
                # Get market data for unrealized P&L
                market_data = self.get_market_data(position.contract.symbol)
                market_price = market_data.get('last_price', position.avgCost)
                
                pos = Position(
                    symbol=position.contract.symbol,
                    quantity=position.position,
                    avg_cost=position.avgCost,
                    market_price=market_price,
                    market_value=position.position * market_price,
                    unrealized_pnl=position.unrealizedPNL or 0.0,
                    realized_pnl=0.0
                )
                positions.append(pos)
        return positions
        
    def get_account_info(self) -> Dict:
        """Get account information."""
        return self.account_info
        
    def _get_account_summary(self) -> Dict:
        """Get account summary from IBKR."""
        try:
            account_values = self.ib.accountSummary()
            summary = {}
            for item in account_values:
                summary[item.tag] = item.value
            return summary
        except Exception:
            return {}
            
    def get_market_data(self, symbol: str) -> Dict:
        """Get market data."""
        try:
            contract = ib_insync.Stock(symbol, 'SMART', 'USD')
            ticker = self.ib.reqMktData(contract)
            self.ib.sleep(1)  # Wait for data
            
            return {
                'symbol': symbol,
                'last_price': ticker.last,
                'bid': ticker.bid,
                'ask': ticker.ask,
                'bid_size': ticker.bidSize,
                'ask_size': ticker.askSize,
                'volume': ticker.volume
            }
        except Exception as e:
            print(f"Failed to get market data for {symbol}: {e}")
            return {'symbol': symbol, 'last_price': 0.0}


class AlpacaInterface(BrokerInterface):
    """Alpaca API integration using environment variables."""
    
    def __init__(self, api_key: Optional[str] = None, 
                 secret_key: Optional[str] = None,
                 base_url: Optional[str] = None):
        """
        Initialize Alpaca interface.
        
        Args:
            api_key: API key (uses Config.ALPACA_API_KEY if None)
            secret_key: Secret key (uses Config.ALPACA_SECRET_KEY if None)
            base_url: Base URL (uses Config.ALPACA_BASE_URL if None)
        """
        # Use environment variables as defaults
        self.api_key = api_key or Config.ALPACA_API_KEY
        self.secret_key = secret_key or Config.ALPACA_SECRET_KEY
        self.base_url = base_url or Config.ALPACA_BASE_URL
        
        if not self.api_key or not self.secret_key:
            raise ValueError(
                "Alpaca API credentials not found. "
                "Set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env file"
            )
        
        credentials = {
            'api_key': self.api_key,
            'secret_key': self.secret_key
        }
        super().__init__(credentials)
        self.api = None
        
        if not ALPACA_AVAILABLE:
            raise ImportError("alpaca-trade-api not available. Install with: pip install alpaca-trade-api")
    
    def connect(self) -> bool:
        """Connect to Alpaca API."""
        try:
            self.api = tradeapi.REST(
                self.api_key,
                self.secret_key,
                base_url=self.base_url,
                api_version='v2'
            )
            
            # Test connection
            account = self.api.get_account()
            self.connected = True
            self.account_info = self._format_account_info(account)
            print(f"✅ Connected to Alpaca ({self.base_url})")
            return True
            
        except Exception as e:
            print(f"❌ Failed to connect to Alpaca: {e}")
            return False
            
    def disconnect(self):
        """Disconnect from Alpaca."""
        self.connected = False
        self.api = None
        
    def submit_order(self, order: Order) -> str:
        """Submit order to Alpaca."""
        if not self.connected:
            raise RuntimeError("Not connected to broker")
            
        # Map order type
        alpaca_order_type = self._map_order_type(order.order_type)
        
        # Prepare order parameters
        order_params = {
            'symbol': order.symbol,
            'qty': abs(order.quantity),
            'side': order.side.lower(),
            'type': alpaca_order_type,
            'time_in_force': order.time_in_force.lower()
        }
        
        if order.price:
            order_params['limit_price'] = order.price
        if order.stop_price:
            order_params['stop_price'] = order.stop_price
            
        try:
            response = self.api.submit_order(**order_params)
            order_id = response.id
            
            # Update order object
            order.order_id = order_id
            order.status = OrderStatus.SUBMITTED
            
            return order_id
            
        except Exception as e:
            order.status = OrderStatus.ERROR
            raise RuntimeError(f"Failed to submit order: {e}")
            
    def _map_order_type(self, order_type: OrderType) -> str:
        """Map our OrderType to Alpaca order type."""
        mapping = {
            OrderType.MARKET: 'market',
            OrderType.LIMIT: 'limit',
            OrderType.STOP: 'stop',
            OrderType.STOP_LIMIT: 'stop_limit',
            OrderType.MARKET_ON_CLOSE: 'market',
            OrderType.LIMIT_ON_CLOSE: 'limit'
        }
        return mapping.get(order_type, 'market')
        
    def cancel_order(self, order_id: str) -> bool:
        """Cancel order."""
        try:
            self.api.cancel_order(order_id)
            return True
        except Exception as e:
            print(f"Failed to cancel order {order_id}: {e}")
            return False
            
    def get_order_status(self, order_id: str) -> OrderStatus:
        """Get order status."""
        try:
            order = self.api.get_order(order_id)
            return self._map_alpaca_status(order.status)
        except Exception:
            return OrderStatus.ERROR
            
    def _map_alpaca_status(self, alpaca_status: str) -> OrderStatus:
        """Map Alpaca status to our OrderStatus."""
        mapping = {
            'new': OrderStatus.PENDING,
            'partially_filled': OrderStatus.PARTIALLY_FILLED,
            'filled': OrderStatus.FILLED,
            'done_for_day': OrderStatus.CANCELLED,
            'canceled': OrderStatus.CANCELLED,
            'expired': OrderStatus.CANCELLED,
            'replaced': OrderStatus.PENDING,
            'pending_cancel': OrderStatus.PENDING,
            'pending_replace': OrderStatus.PENDING,
            'accepted': OrderStatus.SUBMITTED,
            'pending_new': OrderStatus.PENDING,
            'accepted_for_bidding': OrderStatus.SUBMITTED,
            'stopped': OrderStatus.CANCELLED,
            'rejected': OrderStatus.REJECTED,
            'suspended': OrderStatus.ERROR,
            'calculated': OrderStatus.PENDING
        }
        return mapping.get(alpaca_status, OrderStatus.ERROR)
        
    def get_positions(self) -> List[Position]:
        """Get current positions."""
        positions = []
        try:
            alpaca_positions = self.api.list_positions()
            for pos in alpaca_positions:
                position = Position(
                    symbol=pos.symbol,
                    quantity=float(pos.qty),
                    avg_cost=float(pos.avg_cost),
                    market_price=float(pos.market_value) / float(pos.qty) if float(pos.qty) != 0 else 0,
                    market_value=float(pos.market_value),
                    unrealized_pnl=float(pos.unrealized_pl),
                    realized_pnl=0.0
                )
                positions.append(position)
        except Exception as e:
            print(f"Failed to get positions: {e}")
            
        return positions
        
    def get_account_info(self) -> Dict:
        """Get account information."""
        return self.account_info
        
    def _format_account_info(self, account) -> Dict:
        """Format Alpaca account info."""
        return {
            'account_id': account.id,
            'buying_power': float(account.buying_power),
            'cash': float(account.cash),
            'portfolio_value': float(account.portfolio_value),
            'equity': float(account.equity),
            'last_equity': float(account.last_equity),
            'multiplier': account.multiplier,
            'day_trading_buying_power': float(account.daytrading_buying_power),
            'regt_buying_power': float(account.regt_buying_power)
        }
        
    def get_market_data(self, symbol: str) -> Dict:
        """Get market data."""
        try:
            # Get latest quote
            quote = self.api.get_latest_quote(symbol)
            
            return {
                'symbol': symbol,
                'last_price': (float(quote.bid_price) + float(quote.ask_price)) / 2,
                'bid': float(quote.bid_price),
                'ask': float(quote.ask_price),
                'bid_size': int(quote.bid_size),
                'ask_size': int(quote.ask_size),
                'timestamp': quote.timestamp
            }
        except Exception as e:
            print(f"Failed to get market data for {symbol}: {e}")
            return {'symbol': symbol, 'last_price': 0.0}


class BrokerFactory:
    """Factory for creating broker interfaces with environment variables."""
    
    @staticmethod
    def create_broker(broker_name: str, 
                     api_key: Optional[str] = None,
                     secret_key: Optional[str] = None,
                     **kwargs) -> BrokerInterface:
        """
        Create broker interface using environment variables.
        
        Args:
            broker_name: Broker name ('alpaca', 'interactive_brokers')
            api_key: Optional API key override
            secret_key: Optional secret key override
            **kwargs: Additional broker-specific parameters
            
        Returns:
            Broker interface instance
        """
        broker_name = broker_name.lower()
        
        if broker_name in ['alpaca']:
            return AlpacaInterface(
                api_key=api_key,
                secret_key=secret_key,
                base_url=kwargs.get('base_url')
            )
        elif broker_name in ['interactive_brokers', 'ib', 'ibkr']:
            return InteractiveBrokersInterface(
                host=kwargs.get('host'),
                port=kwargs.get('port'),
                client_id=kwargs.get('client_id')
            )
        else:
            raise ValueError(f"Unsupported broker: {broker_name}")


# Example usage
if __name__ == "__main__":
    print("Broker Integration Test")
    print("=" * 40)
    
    try:
        # Create broker using environment variables
        broker = BrokerFactory.create_broker('alpaca')
        
        if broker.connect():
            print("\n✅ Successfully connected!")
            print("\nAccount Information:")
            account = broker.get_account_info()
            print(f"  Portfolio Value: ${account.get('portfolio_value', 0):,.2f}")
            print(f"  Buying Power: ${account.get('buying_power', 0):,.2f}")
            print(f"  Cash: ${account.get('cash', 0):,.2f}")
            
            # Get positions
            positions = broker.get_positions()
            print(f"\nCurrent Positions: {len(positions)}")
            for pos in positions[:5]:  # Show first 5
                print(f"  {pos.symbol}: {pos.quantity:+,.0f} shares @ ${pos.avg_cost:.2f}")
                print(f"    Market Value: ${pos.market_value:,.2f}, P&L: ${pos.unrealized_pnl:+,.2f}")
            
            # Get market data
            market_data = broker.get_market_data('AAPL')
            print(f"\nAAPL Market Data:")
            print(f"  Last: ${market_data.get('last_price', 0):.2f}")
            print(f"  Bid/Ask: ${market_data.get('bid', 0):.2f} / ${market_data.get('ask', 0):.2f}")
            
            broker.disconnect()
            print("\n✅ Test completed successfully")
            
        else:
            print("❌ Failed to connect to broker")
            
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        print("\nMake sure to:")
        print("  1. Create .env file in project root")
        print("  2. Add your ALPACA_API_KEY and ALPACA_SECRET_KEY")
        print("  3. Run from project root directory")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
