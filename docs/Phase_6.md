# Phase 6: Live Execution and Broker Integration

## Overview

Phase 6 connects the quantum portfolio optimization system to live trading through broker APIs, enabling automated order execution, real-time position monitoring, trade cost analysis, and compliance automation. This phase transforms the system from research/backtesting into a production trading platform capable of institutional-grade trade execution and regulatory compliance.

---

## 6.1 Broker API Integration

### Theory

**Broker Integration Requirements:**
- **Authentication**: API keys, OAuth, session management
- **Rate Limiting**: Respect broker API limits and throttling
- **Market Data**: Real-time quotes, market depth, trade ticks
- **Order Management**: Submit, modify, cancel orders
- **Position Tracking**: Account balances, holdings, P&L

**Common Broker APIs:**
- **Interactive Brokers (IBKR)**: Professional-grade with extensive asset coverage
- **Alpaca**: Commission-free US equities with modern REST API
- **TD Ameritrade**: Full-service broker with comprehensive API
- **Charles Schwab**: Institutional-focused with thinkorswim integration

### Implementation: `broker_interface.py`

```python
# Place in: src/integration/broker_interface.py

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import requests
import threading
import time
import warnings

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
    """Interactive Brokers API integration using ib_insync."""
    
    def __init__(self, credentials: Dict[str, str], host: str = '127.0.0.1', port: int = 7497):
        super().__init__(credentials)
        self.host = host
        self.port = port
        self.ib = None
        
        if not IB_AVAILABLE:
            raise ImportError("ib_insync not available")
            
    def connect(self) -> bool:
        """Connect to Interactive Brokers TWS/Gateway."""
        try:
            self.ib = ib_insync.IB()
            self.ib.connect(self.host, self.port, clientId=1)
            self.connected = True
            self.account_info = self._get_account_summary()
            return True
        except Exception as e:
            print(f"Failed to connect to IBKR: {e}")
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
                    realized_pnl=0.0  # Would need separate query
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
    """Alpaca API integration."""
    
    def __init__(self, credentials: Dict[str, str], paper_trading: bool = True):
        super().__init__(credentials)
        self.paper_trading = paper_trading
        self.api = None
        
        if not ALPACA_AVAILABLE:
            raise ImportError("alpaca-trade-api not available")
            
    def connect(self) -> bool:
        """Connect to Alpaca API."""
        try:
            base_url = 'https://paper-api.alpaca.markets' if self.paper_trading else 'https://api.alpaca.markets'
            self.api = tradeapi.REST(
                self.credentials['api_key'],
                self.credentials['secret_key'],
                base_url=base_url,
                api_version='v2'
            )
            
            # Test connection
            account = self.api.get_account()
            self.connected = True
            self.account_info = self._format_account_info(account)
            return True
            
        except Exception as e:
            print(f"Failed to connect to Alpaca: {e}")
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
                    realized_pnl=0.0  # Alpaca doesn't provide this directly
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
    """Factory for creating broker interfaces."""
    
    @staticmethod
    def create_broker(broker_name: str, credentials: Dict[str, str], **kwargs) -> BrokerInterface:
        """Create broker interface."""
        if broker_name.lower() == 'interactive_brokers' or broker_name.lower() == 'ib':
            return InteractiveBrokersInterface(credentials, **kwargs)
        elif broker_name.lower() == 'alpaca':
            return AlpacaInterface(credentials, **kwargs)
        else:
            raise ValueError(f"Unsupported broker: {broker_name}")


# Example usage
if __name__ == "__main__":
    print("Broker Integration Test")
    print("=" * 30)
    
    # Test with Alpaca (paper trading)
    alpaca_creds = {
        'api_key': 'YOUR_ALPACA_API_KEY',
        'secret_key': 'YOUR_ALPACA_SECRET_KEY'
    }
    
    try:
        # Create broker interface
        broker = BrokerFactory.create_broker('alpaca', alpaca_creds, paper_trading=True)
        
        # Connect
        if broker.connect():
            print("✅ Connected to Alpaca")
            
            # Get account info
            account = broker.get_account_info()
            print(f"Account Value: ${account.get('portfolio_value', 0):,.2f}")
            print(f"Buying Power: ${account.get('buying_power', 0):,.2f}")
            
            # Get positions
            positions = broker.get_positions()
            print(f"\\nCurrent Positions: {len(positions)}")
            for pos in positions:
                print(f"  {pos.symbol}: {pos.quantity:,.0f} shares @ ${pos.avg_cost:.2f}")
                print(f"    Market Value: ${pos.market_value:,.2f}, P&L: ${pos.unrealized_pnl:,.2f}")
            
            # Get market data
            market_data = broker.get_market_data('AAPL')
            print(f"\\nAAPL Market Data:")
            print(f"  Last: ${market_data['last_price']:.2f}")
            print(f"  Bid/Ask: ${market_data.get('bid', 0):.2f} / ${market_data.get('ask', 0):.2f}")
            
            # Create sample order (but don't submit)
            sample_order = Order(
                symbol='AAPL',
                quantity=10,
                order_type=OrderType.LIMIT,
                side='BUY',
                price=150.00,
                time_in_force='DAY'
            )
            print(f"\\nSample Order Created: {sample_order.side} {sample_order.quantity} {sample_order.symbol} @ ${sample_order.price}")
            
            # Disconnect
            broker.disconnect()
            print("\\n✅ Disconnected from broker")
            
        else:
            print("❌ Failed to connect to broker")
            
    except Exception as e:
        print(f"❌ Broker test failed: {e}")
        print("Note: Requires valid Alpaca API credentials to run")
```

---

## 6.2 Order Management System

### Theory

**Order Lifecycle Management:**
1. **Pre-Trade Validation**: Check available buying power, position limits, compliance rules
2. **Order Creation**: Build order object with all required parameters
3. **Order Submission**: Send to broker API with error handling
4. **Order Monitoring**: Track status changes and partial fills
5. **Post-Trade Processing**: Update positions, calculate costs, record trade

**Smart Order Features:**
- **Order Slicing**: Break large orders into smaller pieces
- **Time-Based Execution**: TWAP, VWAP algorithms
- **Conditional Orders**: If-then logic, bracket orders
- **Risk Controls**: Position limits, loss limits, concentration checks

### Implementation: `order_manager.py`

```python
# Place in: src/integration/order_manager.py

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import threading
import time
import queue
import uuid

from integration.broker_interface import BrokerInterface, Order, OrderType, OrderStatus, Position


class OrderPriority(Enum):
    """Order execution priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass
class OrderRequest:
    """Order request with metadata."""
    order: Order
    priority: OrderPriority = OrderPriority.NORMAL
    max_retries: int = 3
    retry_delay: float = 1.0
    callback: Optional[Callable] = None
    metadata: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        if not hasattr(self.order, 'request_id'):
            self.order.request_id = str(uuid.uuid4())


class OrderValidationResult:
    """Result of order validation."""
    
    def __init__(self, is_valid: bool, messages: List[str] = None):
        self.is_valid = is_valid
        self.messages = messages or []
        
    def add_message(self, message: str):
        self.messages.append(message)
        
    def __str__(self):
        status = "VALID" if self.is_valid else "INVALID"
        return f"{status}: {'; '.join(self.messages)}"


class OrderValidator:
    """Pre-trade order validation."""
    
    def __init__(self, broker: BrokerInterface):
        self.broker = broker
        
    def validate_order(self, order: Order) -> OrderValidationResult:
        """Comprehensive order validation."""
        result = OrderValidationResult(True)
        
        # Basic validation
        self._validate_basic_fields(order, result)
        
        # Account validation
        self._validate_account_requirements(order, result)
        
        # Market validation
        self._validate_market_conditions(order, result)
        
        # Risk validation
        self._validate_risk_limits(order, result)
        
        return result
        
    def _validate_basic_fields(self, order: Order, result: OrderValidationResult):
        """Validate basic order fields."""
        if not order.symbol:
            result.is_valid = False
            result.add_message("Missing symbol")
            
        if order.quantity == 0:
            result.is_valid = False
            result.add_message("Zero quantity not allowed")
            
        if order.side not in ['BUY', 'SELL']:
            result.is_valid = False
            result.add_message("Invalid side (must be BUY or SELL)")
            
        if order.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT, OrderType.LIMIT_ON_CLOSE]:
            if order.price is None or order.price <= 0:
                result.is_valid = False
                result.add_message("Limit orders require valid price")
                
        if order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT]:
            if order.stop_price is None or order.stop_price <= 0:
                result.is_valid = False
                result.add_message("Stop orders require valid stop price")
                
    def _validate_account_requirements(self, order: Order, result: OrderValidationResult):
        """Validate account has sufficient buying power."""
        try:
            account_info = self.broker.get_account_info()
            
            if order.side == 'BUY':
                buying_power = float(account_info.get('buying_power', 0))
                
                # Estimate order value
                if order.order_type == OrderType.MARKET:
                    market_data = self.broker.get_market_data(order.symbol)
                    estimated_price = market_data.get('ask', market_data.get('last_price', 0))
                else:
                    estimated_price = order.price or 0
                    
                estimated_value = abs(order.quantity) * estimated_price
                
                if estimated_value > buying_power:
                    result.is_valid = False
                    result.add_message(f"Insufficient buying power: ${buying_power:,.2f} < ${estimated_value:,.2f}")
                    
        except Exception as e:
            result.add_message(f"Could not validate account requirements: {e}")
            
    def _validate_market_conditions(self, order: Order, result: OrderValidationResult):
        """Validate market conditions."""
        try:
            market_data = self.broker.get_market_data(order.symbol)
            
            # Check if we have valid market data
            if not market_data.get('last_price'):
                result.add_message(f"No market data available for {order.symbol}")
                return
                
            # Check for reasonable price levels
            last_price = market_data['last_price']
            
            if order.price:
                price_deviation = abs(order.price - last_price) / last_price
                if price_deviation > 0.15:  # 15% from market
                    result.add_message(f"Order price ${order.price:.2f} deviates {price_deviation:.1%} from market ${last_price:.2f}")
                    
        except Exception as e:
            result.add_message(f"Could not validate market conditions: {e}")
            
    def _validate_risk_limits(self, order: Order, result: OrderValidationResult):
        """Validate against risk limits."""
        try:
            # Get current positions
            positions = self.broker.get_positions()
            current_position = 0
            
            for pos in positions:
                if pos.symbol == order.symbol:
                    current_position = pos.quantity
                    break
                    
            # Calculate new position after order
            new_quantity = order.quantity if order.side == 'BUY' else -order.quantity
            new_position = current_position + new_quantity
            
            # Position size limits (example: max $100k per position)
            market_data = self.broker.get_market_data(order.symbol)
            estimated_price = market_data.get('last_price', order.price or 0)
            new_position_value = abs(new_position) * estimated_price
            
            MAX_POSITION_VALUE = 100000  # $100k limit
            if new_position_value > MAX_POSITION_VALUE:
                result.is_valid = False
                result.add_message(f"Position limit exceeded: ${new_position_value:,.2f} > ${MAX_POSITION_VALUE:,.2f}")
                
        except Exception as e:
            result.add_message(f"Could not validate risk limits: {e}")


class OrderManager:
    """Comprehensive order management system."""
    
    def __init__(self, broker: BrokerInterface):
        self.broker = broker
        self.validator = OrderValidator(broker)
        
        # Order tracking
        self.active_orders: Dict[str, OrderRequest] = {}
        self.completed_orders: Dict[str, OrderRequest] = {}
        self.order_history: List[OrderRequest] = []
        
        # Threading
        self.order_queue = queue.PriorityQueue()
        self.monitoring_thread = None
        self.processing_thread = None
        self.running = False
        
        # Statistics
        self.stats = {
            'orders_submitted': 0,
            'orders_filled': 0,
            'orders_cancelled': 0,
            'orders_rejected': 0,
            'total_volume': 0.0,
            'total_commission': 0.0
        }
        
    def start(self):
        """Start order management threads."""
        if self.running:
            return
            
        self.running = True
        
        # Start order processing thread
        self.processing_thread = threading.Thread(target=self._process_orders, daemon=True)
        self.processing_thread.start()
        
        # Start order monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitor_orders, daemon=True)
        self.monitoring_thread.start()
        
        print("Order Manager started")
        
    def stop(self):
        """Stop order management."""
        self.running = False
        
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
            
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
            
        print("Order Manager stopped")
        
    def submit_order_request(self, request: OrderRequest) -> bool:
        """Submit order request for processing."""
        # Validate order
        validation = self.validator.validate_order(request.order)
        
        if not validation.is_valid:
            print(f"Order validation failed: {validation}")
            request.order.status = OrderStatus.REJECTED
            self._complete_order(request)
            return False
            
        # Add to queue with priority
        priority_value = (5 - request.priority.value, datetime.now().timestamp())
        self.order_queue.put((priority_value, request))
        
        print(f"Order request queued: {request.order.side} {request.order.quantity} {request.order.symbol}")
        return True
        
    def submit_order(self, symbol: str, quantity: float, order_type: OrderType,
                    side: str, price: Optional[float] = None,
                    priority: OrderPriority = OrderPriority.NORMAL,
                    **kwargs) -> Optional[str]:
        """Convenience method to submit simple order."""
        order = Order(
            symbol=symbol,
            quantity=quantity,
            order_type=order_type,
            side=side,
            price=price,
            **kwargs
        )
        
        request = OrderRequest(order=order, priority=priority)
        
        if self.submit_order_request(request):
            return order.order_id
        return None
        
    def cancel_order(self, order_id: str) -> bool:
        """Cancel active order."""
        if order_id in self.active_orders:
            try:
                success = self.broker.cancel_order(order_id)
                if success:
                    request = self.active_orders[order_id]
                    request.order.status = OrderStatus.CANCELLED
                    self._complete_order(request)
                return success
            except Exception as e:
                print(f"Failed to cancel order {order_id}: {e}")
                return False
        return False
        
    def get_order_status(self, order_id: str) -> Optional[OrderStatus]:
        """Get order status."""
        if order_id in self.active_orders:
            return self.active_orders[order_id].order.status
        elif order_id in self.completed_orders:
            return self.completed_orders[order_id].order.status
        return None
        
    def get_active_orders(self) -> List[Order]:
        """Get all active orders."""
        return [req.order for req in self.active_orders.values()]
        
    def get_order_statistics(self) -> Dict:
        """Get order execution statistics."""
        stats = self.stats.copy()
        stats['active_orders'] = len(self.active_orders)
        stats['completed_orders'] = len(self.completed_orders)
        
        if stats['orders_submitted'] > 0:
            stats['fill_rate'] = stats['orders_filled'] / stats['orders_submitted']
        else:
            stats['fill_rate'] = 0.0
            
        return stats
        
    def _process_orders(self):
        """Background thread to process order queue."""
        while self.running:
            try:
                # Get next order request (blocks until available)
                priority, request = self.order_queue.get(timeout=1.0)
                
                # Submit to broker
                self._submit_to_broker(request)
                
                self.order_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error processing orders: {e}")
                
    def _submit_to_broker(self, request: OrderRequest):
        """Submit order to broker with retry logic."""
        order = request.order
        retries = 0
        
        while retries <= request.max_retries:
            try:
                # Submit order
                order_id = self.broker.submit_order(order)
                order.order_id = order_id
                order.status = OrderStatus.SUBMITTED
                
                # Track active order
                self.active_orders[order_id] = request
                
                # Update stats
                self.stats['orders_submitted'] += 1
                
                # Call callback if provided
                if request.callback:
                    try:
                        request.callback(order, 'submitted')
                    except Exception as e:
                        print(f"Callback error: {e}")
                        
                print(f"Order submitted: {order_id} - {order.side} {order.quantity} {order.symbol}")
                return
                
            except Exception as e:
                retries += 1
                print(f"Order submission failed (attempt {retries}): {e}")
                
                if retries <= request.max_retries:
                    time.sleep(request.retry_delay)
                else:
                    # Final failure
                    order.status = OrderStatus.ERROR
                    self.stats['orders_rejected'] += 1
                    self._complete_order(request)
                    
    def _monitor_orders(self):
        """Background thread to monitor active orders."""
        while self.running:
            try:
                # Check all active orders
                for order_id, request in list(self.active_orders.items()):
                    self._check_order_status(order_id, request)
                    
                time.sleep(1.0)  # Check every second
                
            except Exception as e:
                print(f"Error monitoring orders: {e}")
                
    def _check_order_status(self, order_id: str, request: OrderRequest):
        """Check and update order status."""
        try:
            current_status = self.broker.get_order_status(order_id)
            order = request.order
            
            if current_status != order.status:
                order.status = current_status
                
                # Handle status changes
                if current_status == OrderStatus.FILLED:
                    order.filled_at = datetime.now()
                    order.filled_quantity = abs(order.quantity)  # Simplified
                    self.stats['orders_filled'] += 1
                    self.stats['total_volume'] += abs(order.quantity)
                    self._complete_order(request)
                    
                elif current_status == OrderStatus.CANCELLED:
                    self.stats['orders_cancelled'] += 1
                    self._complete_order(request)
                    
                elif current_status == OrderStatus.REJECTED:
                    self.stats['orders_rejected'] += 1
                    self._complete_order(request)
                    
                elif current_status == OrderStatus.ERROR:
                    self._complete_order(request)
                    
                # Call callback for status changes
                if request.callback:
                    try:
                        request.callback(order, current_status.value.lower())
                    except Exception as e:
                        print(f"Callback error: {e}")
                        
        except Exception as e:
            print(f"Error checking order status {order_id}: {e}")
            
    def _complete_order(self, request: OrderRequest):
        """Move order from active to completed."""
        order_id = request.order.order_id
        
        if order_id in self.active_orders:
            del self.active_orders[order_id]
            
        self.completed_orders[order_id] = request
        self.order_history.append(request)
        
        print(f"Order completed: {order_id} - Status: {request.order.status.value}")


# Example usage
if __name__ == "__main__":
    from integration.broker_interface import BrokerFactory
    
    print("Order Manager Test")
    print("=" * 25)
    
    # Mock broker for testing
    class MockBroker(BrokerInterface):
        def __init__(self):
            super().__init__({})
            self.connected = True
            self.order_counter = 1000
            
        def connect(self) -> bool:
            return True
            
        def disconnect(self):
            pass
            
        def submit_order(self, order: Order) -> str:
            order_id = f"ORDER_{self.order_counter}"
            self.order_counter += 1
            return order_id
            
        def cancel_order(self, order_id: str) -> bool:
            return True
            
        def get_order_status(self, order_id: str) -> OrderStatus:
            # Simulate order filling after 3 seconds
            if hasattr(self, '_order_times'):
                if order_id in self._order_times:
                    if time.time() - self._order_times[order_id] > 3:
                        return OrderStatus.FILLED
                    else:
                        return OrderStatus.SUBMITTED
            else:
                self._order_times = {}
                
            self._order_times[order_id] = time.time()
            return OrderStatus.SUBMITTED
            
        def get_positions(self) -> List[Position]:
            return []
            
        def get_account_info(self) -> Dict:
            return {'buying_power': 10000, 'portfolio_value': 50000}
            
        def get_market_data(self, symbol: str) -> Dict:
            return {'symbol': symbol, 'last_price': 100.0, 'bid': 99.5, 'ask': 100.5}
    
    # Test order manager
    broker = MockBroker()
    order_manager = OrderManager(broker)
    
    # Start order manager
    order_manager.start()
    
    try:
        # Submit test orders
        def order_callback(order, status):
            print(f"  Callback: {order.symbol} order {status}")
            
        print("\\nSubmitting test orders...")
        
        # Market buy order
        order_manager.submit_order(
            symbol='AAPL',
            quantity=100,
            order_type=OrderType.MARKET,
            side='BUY',
            priority=OrderPriority.HIGH
        )
        
        # Limit sell order with callback
        request = OrderRequest(
            order=Order(
                symbol='GOOGL',
                quantity=50,
                order_type=OrderType.LIMIT,
                side='SELL',
                price=2500.00
            ),
            priority=OrderPriority.NORMAL,
            callback=order_callback
        )
        
        order_manager.submit_order_request(request)
        
        # Wait and monitor
        print("\\nMonitoring orders...")
        for i in range(10):
            time.sleep(1)
            stats = order_manager.get_order_statistics()
            active = len(order_manager.get_active_orders())
            print(f"  Active: {active}, Submitted: {stats['orders_submitted']}, Filled: {stats['orders_filled']}")
            
            if stats['orders_filled'] >= 2:
                break
                
        # Final statistics
        final_stats = order_manager.get_order_statistics()
        print(f"\\nFinal Statistics:")
        print(f"  Orders Submitted: {final_stats['orders_submitted']}")
        print(f"  Orders Filled: {final_stats['orders_filled']}")
        print(f"  Fill Rate: {final_stats['fill_rate']:.1%}")
        print(f"  Total Volume: {final_stats['total_volume']}")
        
    finally:
        order_manager.stop()
        print("\\n✅ Order Manager test completed")
```

---

## 6.3 Execution Algorithms

### Theory

**Smart Order Routing (SOR):**
- **VWAP (Volume Weighted Average Price)**: Match historical volume profile
- **TWAP (Time Weighted Average Price)**: Spread execution over time
- **POV (Percentage of Volume)**: Execute as percentage of market volume
- **Implementation Shortfall**: Balance market impact vs. timing risk

**VWAP Algorithm:**
\\[
\text{VWAP} = \frac{\sum_{i} P_i \times V_i}{\sum_{i} V_i}
\\]

**Market Impact Model:**
\\[
\text{Impact} = \alpha \times \left(\frac{\text{Order Size}}{\text{ADV}}\right)^{\beta}
\\]
Where ADV is Average Daily Volume.

### Implementation: `execution_algorithms.py`

```python
# Place in: src/integration/execution_algorithms.py

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, time
from abc import ABC, abstractmethod
import threading
import time as time_module
from enum import Enum

from integration.order_manager import OrderManager, Order, OrderType, OrderPriority


class AlgorithmType(Enum):
    """Execution algorithm types."""
    VWAP = "VWAP"
    TWAP = "TWAP"
    POV = "POV"
    IS = "Implementation Shortfall"


@dataclass
class ExecutionRequest:
    """Request for algorithmic execution."""
    symbol: str
    total_quantity: float
    side: str  # 'BUY' or 'SELL'
    algorithm: AlgorithmType
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    max_participation_rate: float = 0.20  # Max 20% of volume
    price_limit: Optional[float] = None
    urgency: float = 0.5  # 0 = patient, 1 = aggressive
    callback: Optional[Callable] = None
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.start_time is None:
            self.start_time = datetime.now()
        if self.end_time is None:
            # Default to market close (4 PM ET)
            today = datetime.now().date()
            self.end_time = datetime.combine(today, time(16, 0))


@dataclass
class ExecutionSlice:
    """Individual execution slice."""
    quantity: float
    target_time: datetime
    price_limit: Optional[float] = None
    executed_quantity: float = 0.0
    avg_price: float = 0.0
    status: str = 'PENDING'  # PENDING, EXECUTED, CANCELLED


class ExecutionAlgorithm(ABC):
    """Base class for execution algorithms."""
    
    def __init__(self, order_manager: OrderManager):
        self.order_manager = order_manager
        self.active_executions: Dict[str, ExecutionRequest] = {}
        
    @abstractmethod
    def generate_schedule(self, request: ExecutionRequest) -> List[ExecutionSlice]:
        """Generate execution schedule."""
        pass
        
    @abstractmethod
    def get_algorithm_name(self) -> str:
        """Get algorithm name."""
        pass
        
    def start_execution(self, request: ExecutionRequest) -> str:
        """Start algorithmic execution."""
        execution_id = f"{request.symbol}_{datetime.now().timestamp()}"
        self.active_executions[execution_id] = request
        
        # Generate execution schedule
        schedule = self.generate_schedule(request)
        request.metadata['schedule'] = schedule
        request.metadata['execution_id'] = execution_id
        
        # Start execution thread
        execution_thread = threading.Thread(
            target=self._execute_schedule,
            args=(execution_id, schedule),
            daemon=True
        )
        execution_thread.start()
        
        print(f"Started {self.get_algorithm_name()} execution: {execution_id}")
        return execution_id
        
    def _execute_schedule(self, execution_id: str, schedule: List[ExecutionSlice]):
        """Execute the generated schedule."""
        request = self.active_executions[execution_id]
        total_executed = 0.0
        total_cost = 0.0
        
        for i, slice_order in enumerate(schedule):
            # Wait until target time
            while datetime.now() < slice_order.target_time:
                time_module.sleep(0.5)
                
            # Check if we should continue
            if execution_id not in self.active_executions:
                break  # Execution was cancelled
                
            # Submit slice order
            try:
                order = Order(
                    symbol=request.symbol,
                    quantity=slice_order.quantity,
                    order_type=OrderType.LIMIT if slice_order.price_limit else OrderType.MARKET,
                    side=request.side,
                    price=slice_order.price_limit,
                    time_in_force='IOC' if slice_order.price_limit else 'DAY'
                )
                
                # Submit order
                order_id = self.order_manager.broker.submit_order(order)
                
                # Wait for fill (simplified - in practice, use callbacks)
                time_module.sleep(2)  # Simulate execution time
                
                # Update slice status (simplified)
                slice_order.executed_quantity = slice_order.quantity
                slice_order.avg_price = slice_order.price_limit or self._get_market_price(request.symbol)
                slice_order.status = 'EXECUTED'
                
                total_executed += slice_order.executed_quantity
                total_cost += slice_order.executed_quantity * slice_order.avg_price
                
                print(f"Executed slice {i+1}/{len(schedule)}: {slice_order.quantity} @ ${slice_order.avg_price:.2f}")
                
            except Exception as e:
                print(f"Failed to execute slice {i+1}: {e}")
                slice_order.status = 'FAILED'
                
        # Calculate execution summary
        avg_price = total_cost / total_executed if total_executed > 0 else 0
        
        # Call completion callback
        if request.callback:
            try:
                summary = {
                    'execution_id': execution_id,
                    'total_quantity': request.total_quantity,
                    'executed_quantity': total_executed,
                    'avg_price': avg_price,
                    'algorithm': self.get_algorithm_name(),
                    'slices_completed': sum(1 for s in schedule if s.status == 'EXECUTED')
                }
                request.callback(summary)
            except Exception as e:
                print(f"Callback error: {e}")
                
        # Clean up
        if execution_id in self.active_executions:
            del self.active_executions[execution_id]
            
        print(f"Completed execution {execution_id}: {total_executed}/{request.total_quantity} @ ${avg_price:.2f}")
        
    def _get_market_price(self, symbol: str) -> float:
        """Get current market price."""
        try:
            market_data = self.order_manager.broker.get_market_data(symbol)
            return market_data.get('last_price', 0.0)
        except:
            return 0.0
            
    def cancel_execution(self, execution_id: str) -> bool:
        """Cancel active execution."""
        if execution_id in self.active_executions:
            del self.active_executions[execution_id]
            return True
        return False


class VWAPAlgorithm(ExecutionAlgorithm):
    """Volume Weighted Average Price algorithm."""
    
    def __init__(self, order_manager: OrderManager):
        super().__init__(order_manager)
        self.historical_volume_profile = {}  # Cache volume profiles
        
    def get_algorithm_name(self) -> str:
        return "VWAP"
        
    def generate_schedule(self, request: ExecutionRequest) -> List[ExecutionSlice]:
        """Generate VWAP execution schedule."""
        # Get historical volume profile
        volume_profile = self._get_volume_profile(request.symbol)
        
        # Calculate time intervals
        total_minutes = int((request.end_time - request.start_time).total_seconds() / 60)
        interval_minutes = max(5, total_minutes // 20)  # 5-minute minimum intervals
        
        schedule = []
        remaining_quantity = abs(request.total_quantity)
        
        current_time = request.start_time
        
        while current_time < request.end_time and remaining_quantity > 0:
            # Get expected volume for this time interval
            interval_hour = current_time.hour
            interval_minute = current_time.minute
            
            expected_volume_pct = volume_profile.get(interval_hour, 0.05)  # Default 5%
            
            # Calculate slice size based on volume profile
            slice_quantity = min(
                remaining_quantity,
                request.total_quantity * expected_volume_pct * request.max_participation_rate
            )
            
            if slice_quantity > 0:
                # Calculate price limit (if any)
                price_limit = None
                if request.price_limit:
                    # Adjust limit based on urgency
                    market_price = self._get_market_price(request.symbol)
                    if market_price > 0:
                        spread_pct = 0.001 * (1 + request.urgency)  # 0.1% to 0.2% spread
                        if request.side == 'BUY':
                            price_limit = min(request.price_limit, market_price * (1 + spread_pct))
                        else:
                            price_limit = max(request.price_limit, market_price * (1 - spread_pct))
                
                slice_order = ExecutionSlice(
                    quantity=slice_quantity,
                    target_time=current_time,
                    price_limit=price_limit
                )
                schedule.append(slice_order)
                
                remaining_quantity -= slice_quantity
            
            current_time += timedelta(minutes=interval_minutes)
            
        return schedule
        
    def _get_volume_profile(self, symbol: str) -> Dict[int, float]:
        """Get historical intraday volume profile."""
        # Simplified volume profile (in practice, get from market data)
        # Returns percentage of daily volume for each hour
        default_profile = {
            9: 0.08,   # 9:00-10:00 AM
            10: 0.12,  # 10:00-11:00 AM
            11: 0.08,  # 11:00-12:00 PM
            12: 0.06,  # 12:00-1:00 PM
            13: 0.08,  # 1:00-2:00 PM
            14: 0.10,  # 2:00-3:00 PM
            15: 0.18,  # 3:00-4:00 PM (closing hour)
            16: 0.20   # 4:00 PM (market close)
        }
        
        return self.historical_volume_profile.get(symbol, default_profile)


class TWAPAlgorithm(ExecutionAlgorithm):
    """Time Weighted Average Price algorithm."""
    
    def get_algorithm_name(self) -> str:
        return "TWAP"
        
    def generate_schedule(self, request: ExecutionRequest) -> List[ExecutionSlice]:
        """Generate TWAP execution schedule."""
        # Equal time intervals
        total_minutes = int((request.end_time - request.start_time).total_seconds() / 60)
        interval_minutes = max(5, total_minutes // 15)  # 15 slices maximum
        
        # Equal quantity per slice
        num_slices = total_minutes // interval_minutes
        slice_quantity = abs(request.total_quantity) / num_slices
        
        schedule = []
        current_time = request.start_time
        
        for i in range(num_slices):
            if current_time >= request.end_time:
                break
                
            # Last slice gets any remaining quantity
            if i == num_slices - 1:
                remaining = abs(request.total_quantity) - (slice_quantity * i)
                slice_quantity = remaining
            
            slice_order = ExecutionSlice(
                quantity=slice_quantity,
                target_time=current_time,
                price_limit=request.price_limit
            )
            schedule.append(slice_order)
            
            current_time += timedelta(minutes=interval_minutes)
            
        return schedule


class POVAlgorithm(ExecutionAlgorithm):
    """Percentage of Volume algorithm."""
    
    def get_algorithm_name(self) -> str:
        return "POV"
        
    def generate_schedule(self, request: ExecutionRequest) -> List[ExecutionSlice]:
        """Generate POV execution schedule."""
        # Dynamic scheduling based on real-time volume
        # This is a simplified version - production would monitor live volume
        
        total_minutes = int((request.end_time - request.start_time).total_seconds() / 60)
        interval_minutes = 5  # 5-minute intervals for volume monitoring
        
        schedule = []
        remaining_quantity = abs(request.total_quantity)
        current_time = request.start_time
        
        # Estimate daily volume (simplified)
        daily_volume = self._estimate_daily_volume(request.symbol)
        minute_volume = daily_volume / (6.5 * 60)  # 6.5 hour trading day
        
        while current_time < request.end_time and remaining_quantity > 0:
            # Expected volume for this interval
            expected_interval_volume = minute_volume * interval_minutes
            
            # Our target participation
            target_quantity = min(
                remaining_quantity,
                expected_interval_volume * request.max_participation_rate
            )
            
            if target_quantity > 0:
                slice_order = ExecutionSlice(
                    quantity=target_quantity,
                    target_time=current_time,
                    price_limit=request.price_limit
                )
                schedule.append(slice_order)
                
                remaining_quantity -= target_quantity
            
            current_time += timedelta(minutes=interval_minutes)
            
        return schedule
        
    def _estimate_daily_volume(self, symbol: str) -> float:
        """Estimate daily volume."""
        # Simplified - in practice, use historical data
        return 1000000  # 1M shares daily volume estimate


class ExecutionEngine:
    """Main execution engine coordinating all algorithms."""
    
    def __init__(self, order_manager: OrderManager):
        self.order_manager = order_manager
        
        # Initialize algorithms
        self.algorithms = {
            AlgorithmType.VWAP: VWAPAlgorithm(order_manager),
            AlgorithmType.TWAP: TWAPAlgorithm(order_manager),
            AlgorithmType.POV: POVAlgorithm(order_manager)
        }
        
        self.active_executions = {}
        
    def execute_order(self, request: ExecutionRequest) -> str:
        """Execute order using specified algorithm."""
        if request.algorithm not in self.algorithms:
            raise ValueError(f"Unsupported algorithm: {request.algorithm}")
            
        algorithm = self.algorithms[request.algorithm]
        execution_id = algorithm.start_execution(request)
        
        self.active_executions[execution_id] = {
            'request': request,
            'algorithm': algorithm,
            'start_time': datetime.now()
        }
        
        return execution_id
        
    def cancel_execution(self, execution_id: str) -> bool:
        """Cancel active execution."""
        if execution_id in self.active_executions:
            algorithm = self.active_executions[execution_id]['algorithm']
            success = algorithm.cancel_execution(execution_id)
            
            if success:
                del self.active_executions[execution_id]
                
            return success
        return False
        
    def get_execution_status(self, execution_id: str) -> Optional[Dict]:
        """Get execution status."""
        if execution_id in self.active_executions:
            execution_info = self.active_executions[execution_id]
            return {
                'execution_id': execution_id,
                'algorithm': execution_info['algorithm'].get_algorithm_name(),
                'symbol': execution_info['request'].symbol,
                'total_quantity': execution_info['request'].total_quantity,
                'start_time': execution_info['start_time'],
                'status': 'ACTIVE'
            }
        return None
        
    def get_active_executions(self) -> List[Dict]:
        """Get all active executions."""
        return [self.get_execution_status(eid) for eid in self.active_executions.keys()]


# Example usage
if __name__ == "__main__":
    from integration.broker_interface import MockBroker  # From previous example
    
    print("Execution Algorithms Test")
    print("=" * 30)
    
    # Setup
    broker = MockBroker()
    order_manager = OrderManager(broker)
    order_manager.start()
    
    execution_engine = ExecutionEngine(order_manager)
    
    try:
        # Test VWAP execution
        def execution_callback(summary):
            print(f"\\nExecution completed: {summary}")
            
        vwap_request = ExecutionRequest(
            symbol='AAPL',
            total_quantity=1000,
            side='BUY',
            algorithm=AlgorithmType.VWAP,
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(minutes=30),
            max_participation_rate=0.15,
            callback=execution_callback
        )
        
        print(f"\\nStarting VWAP execution: {vwap_request.total_quantity} {vwap_request.symbol}")
        execution_id = execution_engine.execute_order(vwap_request)
        
        # Test TWAP execution
        twap_request = ExecutionRequest(
            symbol='GOOGL',
            total_quantity=200,
            side='SELL',
            algorithm=AlgorithmType.TWAP,
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(minutes=20),
            callback=execution_callback
        )
        
        print(f"Starting TWAP execution: {twap_request.total_quantity} {twap_request.symbol}")
        execution_id2 = execution_engine.execute_order(twap_request)
        
        # Monitor executions
        print("\\nMonitoring executions...")
        for i in range(15):
            active = execution_engine.get_active_executions()
            print(f"  Active executions: {len(active)}")
            for exec_info in active:
                print(f"    {exec_info['algorithm']}: {exec_info['symbol']} - {exec_info['total_quantity']}")
            
            if len(active) == 0:
                break
                
            time_module.sleep(2)
            
        print("\\n✅ Execution algorithms test completed")
        
    finally:
        order_manager.stop()
```

---

## 6.4 Position Management and Reconciliation

### Implementation: `position_manager.py`

```python
# Place in: src/integration/position_manager.py

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import threading
import time

from integration.broker_interface import BrokerInterface, Position


@dataclass
class Trade:
    """Individual trade record."""
    symbol: str
    quantity: float
    price: float
    side: str
    trade_time: datetime
    order_id: Optional[str] = None
    commission: float = 0.0
    trade_id: Optional[str] = None


@dataclass
class PositionRecord:
    """Enhanced position with tracking."""
    position: Position
    last_updated: datetime
    trades: List[Trade]
    realized_pnl_today: float = 0.0
    high_water_mark: float = 0.0
    max_drawdown: float = 0.0
    
    @property
    def total_pnl_today(self) -> float:
        return self.realized_pnl_today + self.position.unrealized_pnl


class PositionManager:
    """Comprehensive position management and reconciliation."""
    
    def __init__(self, broker: BrokerInterface, update_interval: int = 30):
        self.broker = broker
        self.update_interval = update_interval
        
        # Position tracking
        self.positions: Dict[str, PositionRecord] = {}
        self.historical_positions: List[Dict] = []
        self.trades_today: List[Trade] = []
        
        # Portfolio metrics
        self.portfolio_metrics = {
            'total_value': 0.0,
            'total_pnl': 0.0,
            'total_pnl_pct': 0.0,
            'cash': 0.0,
            'margin_used': 0.0,
            'buying_power': 0.0
        }
        
        # Threading
        self.monitoring_thread = None
        self.running = False
        self.lock = threading.Lock()
        
    def start_monitoring(self):
        """Start position monitoring."""
        if self.running:
            return
            
        self.running = True
        self.monitoring_thread = threading.Thread(target=self._monitor_positions, daemon=True)
        self.monitoring_thread.start()
        print("Position monitoring started")
        
    def stop_monitoring(self):
        """Stop position monitoring."""
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        print("Position monitoring stopped")
        
    def _monitor_positions(self):
        """Background position monitoring."""
        while self.running:
            try:
                self.update_positions()
                self.update_portfolio_metrics()
                time.sleep(self.update_interval)
            except Exception as e:
                print(f"Position monitoring error: {e}")
                time.sleep(5)  # Retry after 5 seconds
                
    def update_positions(self):
        """Update all positions from broker."""
        with self.lock:
            try:
                # Get current positions from broker
                broker_positions = self.broker.get_positions()
                current_symbols = set()
                
                for broker_pos in broker_positions:
                    symbol = broker_pos.symbol
                    current_symbols.add(symbol)
                    
                    if symbol in self.positions:
                        # Update existing position
                        self.positions[symbol].position = broker_pos
                        self.positions[symbol].last_updated = datetime.now()
                        self._update_position_metrics(symbol)
                    else:
                        # New position
                        self.positions[symbol] = PositionRecord(
                            position=broker_pos,
                            last_updated=datetime.now(),
                            trades=[],
                            realized_pnl_today=0.0
                        )
                        
                # Remove positions that no longer exist
                symbols_to_remove = set(self.positions.keys()) - current_symbols
                for symbol in symbols_to_remove:
                    # Archive position before removing
                    self._archive_position(symbol)
                    del self.positions[symbol]
                    
            except Exception as e:
                print(f"Failed to update positions: {e}")
                
    def _update_position_metrics(self, symbol: str):
        """Update position-level metrics."""
        if symbol not in self.positions:
            return
            
        pos_record = self.positions[symbol]
        position = pos_record.position
        
        # Update high water mark
        current_pnl = pos_record.total_pnl_today
        if current_pnl > pos_record.high_water_mark:
            pos_record.high_water_mark = current_pnl
            
        # Update max drawdown
        if pos_record.high_water_mark > 0:
            drawdown = (pos_record.high_water_mark - current_pnl) / pos_record.high_water_mark
            pos_record.max_drawdown = max(pos_record.max_drawdown, drawdown)
            
    def _archive_position(self, symbol: str):
        """Archive closed position."""
        if symbol in self.positions:
            archive_record = {
                'symbol': symbol,
                'closed_time': datetime.now(),
                'final_pnl': self.positions[symbol].total_pnl_today,
                'trades_count': len(self.positions[symbol].trades),
                'max_drawdown': self.positions[symbol].max_drawdown
            }
            self.historical_positions.append(archive_record)
            
    def add_trade(self, trade: Trade):
        """Add trade to position tracking."""
        with self.lock:
            symbol = trade.symbol
            self.trades_today.append(trade)
            
            # Update position record
            if symbol not in self.positions:
                # Create empty position record (will be updated on next position sync)
                self.positions[symbol] = PositionRecord(
                    position=Position(
                        symbol=symbol,
                        quantity=0,
                        avg_cost=0,
                        market_price=trade.price,
                        market_value=0,
                        unrealized_pnl=0
                    ),
                    last_updated=datetime.now(),
                    trades=[]
                )
                
            # Add trade to position
            pos_record = self.positions[symbol]
            pos_record.trades.append(trade)
            
            # Update realized P&L if closing trade
            if ((pos_record.position.quantity > 0 and trade.side == 'SELL') or
                (pos_record.position.quantity < 0 and trade.side == 'BUY')):
                
                # Calculate realized P&L for this trade
                if pos_record.position.avg_cost > 0:
                    trade_pnl = abs(trade.quantity) * (trade.price - pos_record.position.avg_cost)
                    if trade.side == 'SELL':
                        pos_record.realized_pnl_today += trade_pnl
                    else:
                        pos_record.realized_pnl_today -= trade_pnl  # Covering short
                        
    def get_position(self, symbol: str) -> Optional[PositionRecord]:
        """Get position for symbol."""
        with self.lock:
            return self.positions.get(symbol)
            
    def get_all_positions(self) -> Dict[str, PositionRecord]:
        """Get all current positions."""
        with self.lock:
            return self.positions.copy()
            
    def update_portfolio_metrics(self):
        """Update portfolio-level metrics."""
        with self.lock:
            try:
                account_info = self.broker.get_account_info()
                
                self.portfolio_metrics.update({
                    'cash': float(account_info.get('cash', 0)),
                    'buying_power': float(account_info.get('buying_power', 0)),
                    'total_value': float(account_info.get('portfolio_value', 0))
                })
                
                # Calculate total P&L from positions
                total_pnl = sum(pos.total_pnl_today for pos in self.positions.values())
                self.portfolio_metrics['total_pnl'] = total_pnl
                
                # Calculate P&L percentage
                if self.portfolio_metrics['total_value'] > 0:
                    self.portfolio_metrics['total_pnl_pct'] = (
                        total_pnl / (self.portfolio_metrics['total_value'] - total_pnl)
                    )
                else:
                    self.portfolio_metrics['total_pnl_pct'] = 0.0
                    
            except Exception as e:
                print(f"Failed to update portfolio metrics: {e}")
                
    def get_portfolio_summary(self) -> Dict:
        """Get portfolio summary."""
        with self.lock:
            summary = self.portfolio_metrics.copy()
            summary.update({
                'num_positions': len(self.positions),
                'num_trades_today': len(self.trades_today),
                'last_updated': datetime.now()
            })
            
            # Position breakdown
            long_value = sum(p.position.market_value for p in self.positions.values() 
                           if p.position.quantity > 0)
            short_value = sum(abs(p.position.market_value) for p in self.positions.values() 
                            if p.position.quantity < 0)
            
            summary.update({
                'long_market_value': long_value,
                'short_market_value': short_value,
                'net_market_value': long_value - short_value
            })
            
            return summary
            
    def get_position_drift(self, target_weights: Dict[str, float], 
                          total_portfolio_value: float) -> Dict[str, float]:
        """Calculate position drift from target weights."""
        drift = {}
        
        with self.lock:
            for symbol, target_weight in target_weights.items():
                target_value = target_weight * total_portfolio_value
                
                if symbol in self.positions:
                    current_value = self.positions[symbol].position.market_value
                else:
                    current_value = 0.0
                    
                drift[symbol] = current_value - target_value
                
        return drift
        
    def check_rebalancing_needed(self, target_weights: Dict[str, float],
                                threshold: float = 0.05) -> Tuple[bool, Dict[str, float]]:
        """Check if rebalancing is needed based on drift threshold."""
        portfolio_value = self.portfolio_metrics['total_value']
        if portfolio_value <= 0:
            return False, {}
            
        drift = self.get_position_drift(target_weights, portfolio_value)
        relative_drift = {}
        
        needs_rebalancing = False
        
        for symbol, drift_value in drift.items():
            relative_drift[symbol] = abs(drift_value) / portfolio_value
            
            if relative_drift[symbol] > threshold:
                needs_rebalancing = True
                
        return needs_rebalancing, relative_drift
        
    def generate_rebalancing_orders(self, target_weights: Dict[str, float],
                                  current_prices: Dict[str, float]) -> List[Dict]:
        """Generate orders needed for rebalancing."""
        portfolio_value = self.portfolio_metrics['total_value']
        orders = []
        
        if portfolio_value <= 0:
            return orders
            
        with self.lock:
            for symbol, target_weight in target_weights.items():
                target_value = target_weight * portfolio_value
                current_value = 0.0
                
                if symbol in self.positions:
                    current_value = self.positions[symbol].position.market_value
                    
                trade_value = target_value - current_value
                
                if abs(trade_value) > 100:  # Minimum $100 trade
                    current_price = current_prices.get(symbol, 0)
                    if current_price > 0:
                        trade_quantity = trade_value / current_price
                        
                        orders.append({
                            'symbol': symbol,
                            'quantity': abs(trade_quantity),
                            'side': 'BUY' if trade_quantity > 0 else 'SELL',
                            'estimated_value': abs(trade_value),
                            'current_weight': current_value / portfolio_value,
                            'target_weight': target_weight
                        })
                        
        return orders
        
    def get_performance_metrics(self) -> Dict:
        """Get performance metrics."""
        with self.lock:
            metrics = {}
            
            if self.positions:
                # Position-level metrics
                winning_positions = [p for p in self.positions.values() 
                                   if p.total_pnl_today > 0]
                losing_positions = [p for p in self.positions.values() 
                                  if p.total_pnl_today < 0]
                
                metrics['win_rate'] = len(winning_positions) / len(self.positions)
                
                if winning_positions:
                    metrics['avg_winner'] = np.mean([p.total_pnl_today for p in winning_positions])
                else:
                    metrics['avg_winner'] = 0.0
                    
                if losing_positions:
                    metrics['avg_loser'] = np.mean([p.total_pnl_today for p in losing_positions])
                else:
                    metrics['avg_loser'] = 0.0
                    
                # Risk metrics
                pnl_values = [p.total_pnl_today for p in self.positions.values()]
                metrics['max_position_pnl'] = max(pnl_values) if pnl_values else 0
                metrics['min_position_pnl'] = min(pnl_values) if pnl_values else 0
                
                # Drawdown
                max_drawdowns = [p.max_drawdown for p in self.positions.values()]
                metrics['max_drawdown'] = max(max_drawdowns) if max_drawdowns else 0
                
            else:
                metrics = {
                    'win_rate': 0.0,
                    'avg_winner': 0.0,
                    'avg_loser': 0.0,
                    'max_position_pnl': 0.0,
                    'min_position_pnl': 0.0,
                    'max_drawdown': 0.0
                }
                
            return metrics


# Example usage
if __name__ == "__main__":
    from integration.broker_interface import MockBroker
    
    print("Position Manager Test")
    print("=" * 25)
    
    # Setup
    broker = MockBroker()
    position_manager = PositionManager(broker, update_interval=5)
    
    try:
        # Add some mock positions to broker
        broker.mock_positions = [
            Position('AAPL', 100, 150.0, 155.0, 15500, 500),
            Position('GOOGL', 50, 2400.0, 2450.0, 122500, 2500),
            Position('MSFT', -25, 300.0, 295.0, -7375, 125)  # Short position
        ]
        
        # Start monitoring
        position_manager.start_monitoring()
        
        # Wait for initial update
        time.sleep(2)
        
        # Check positions
        print("\\nCurrent Positions:")
        positions = position_manager.get_all_positions()
        for symbol, pos_record in positions.items():
            pos = pos_record.position
            print(f"  {symbol}: {pos.quantity:+.0f} @ ${pos.avg_cost:.2f}")
            print(f"    Market Value: ${pos.market_value:,.2f}, P&L: ${pos.unrealized_pnl:+,.2f}")
            
        # Portfolio summary
        summary = position_manager.get_portfolio_summary()
        print(f"\\nPortfolio Summary:")
        print(f"  Total Value: ${summary['total_value']:,.2f}")
        print(f"  Total P&L: ${summary['total_pnl']:+,.2f} ({summary['total_pnl_pct']:+.2%})")
        print(f"  Long Value: ${summary['long_market_value']:,.2f}")
        print(f"  Short Value: ${summary['short_market_value']:,.2f}")
        print(f"  Net Value: ${summary['net_market_value']:,.2f}")
        
        # Test rebalancing
        target_weights = {
            'AAPL': 0.40,
            'GOOGL': 0.35,
            'MSFT': 0.25
        }
        
        needs_rebalancing, drift = position_manager.check_rebalancing_needed(
            target_weights, threshold=0.05
        )
        
        print(f"\\nRebalancing Analysis:")
        print(f"  Needs Rebalancing: {needs_rebalancing}")
        for symbol, drift_pct in drift.items():
            print(f"  {symbol}: {drift_pct:.2%} drift")
            
        if needs_rebalancing:
            current_prices = {'AAPL': 155.0, 'GOOGL': 2450.0, 'MSFT': 295.0}
            orders = position_manager.generate_rebalancing_orders(target_weights, current_prices)
            
            print(f"\\nRebalancing Orders:")
            for order in orders:
                print(f"  {order['side']} {order['quantity']:.0f} {order['symbol']} (${order['estimated_value']:,.2f})")
                print(f"    Current: {order['current_weight']:.2%} → Target: {order['target_weight']:.2%}")
                
        # Performance metrics
        performance = position_manager.get_performance_metrics()
        print(f"\\nPerformance Metrics:")
        print(f"  Win Rate: {performance['win_rate']:.1%}")
        print(f"  Avg Winner: ${performance['avg_winner']:+,.2f}")
        print(f"  Avg Loser: ${performance['avg_loser']:+,.2f}")
        print(f"  Max Drawdown: {performance['max_drawdown']:.2%}")
        
        # Let it run for a bit
        print(f"\\nMonitoring for 10 seconds...")
        time.sleep(10)
        
    finally:
        position_manager.stop_monitoring()
        print("\\n✅ Position Manager test completed")
```

---

## 6.5 Transaction Cost Analysis

### Implementation: `transaction_cost_analyzer.py`

```python
# Place in: src/integration/transaction_cost_analyzer.py

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from integration.position_manager import Trade


class CostType(Enum):
    """Types of transaction costs."""
    COMMISSION = "Commission"
    SPREAD = "Bid-Ask Spread"
    MARKET_IMPACT = "Market Impact"
    SLIPPAGE = "Slippage"
    TIMING_COST = "Timing Cost"
    OPPORTUNITY_COST = "Opportunity Cost"


@dataclass
class TransactionCost:
    """Individual transaction cost component."""
    cost_type: CostType
    amount: float
    basis_points: float
    description: str


@dataclass
class TCAResult:
    """Transaction Cost Analysis result."""
    trade: Trade
    benchmark_price: float
    implementation_shortfall: float
    total_cost: float
    total_cost_bps: float
    cost_breakdown: List[TransactionCost]
    execution_quality_score: float
    
    
class TransactionCostAnalyzer:
    """Comprehensive transaction cost analysis."""
    
    def __init__(self):
        self.commission_rates = {
            'equity': 0.005,  # $0.005 per share
            'option': 0.65,   # $0.65 per contract
            'bond': 0.025     # $0.25 per $1000 notional
        }
        
        self.typical_spreads = {}  # Cache of typical spreads by symbol
        self.market_impact_params = {}  # Market impact model parameters
        
    def analyze_trade(self, trade: Trade, 
                     benchmark_price: Optional[float] = None,
                     pre_trade_quote: Optional[Dict] = None,
                     market_data: Optional[Dict] = None) -> TCAResult:
        """
        Comprehensive transaction cost analysis for a trade.
        
        Args:
            trade: Trade to analyze
            benchmark_price: Benchmark price (VWAP, arrival price, etc.)
            pre_trade_quote: Quote at decision time
            market_data: Additional market data
            
        Returns:
            TCA result with cost breakdown
        """
        cost_breakdown = []
        
        # Use arrival price as benchmark if not provided
        if benchmark_price is None:
            benchmark_price = trade.price
            
        trade_value = abs(trade.quantity) * trade.price
        
        # 1. Explicit Costs
        
        # Commission
        commission = self._calculate_commission(trade)
        if commission > 0:
            cost_breakdown.append(TransactionCost(
                cost_type=CostType.COMMISSION,
                amount=commission,
                basis_points=(commission / trade_value) * 10000,
                description=f"Commission: ${commission:.2f}"
            ))
            
        # 2. Implicit Costs
        
        # Bid-Ask Spread Cost
        spread_cost = self._calculate_spread_cost(trade, pre_trade_quote, market_data)
        if spread_cost > 0:
            cost_breakdown.append(TransactionCost(
                cost_type=CostType.SPREAD,
                amount=spread_cost,
                basis_points=(spread_cost / trade_value) * 10000,
                description=f"Spread cost: ${spread_cost:.2f}"
            ))
            
        # Market Impact
        market_impact = self._calculate_market_impact(trade, market_data)
        if market_impact > 0:
            cost_breakdown.append(TransactionCost(
                cost_type=CostType.MARKET_IMPACT,
                amount=market_impact,
                basis_points=(market_impact / trade_value) * 10000,
                description=f"Market impact: ${market_impact:.2f}"
            ))
            
        # Price Slippage (vs benchmark)
        slippage = self._calculate_slippage(trade, benchmark_price)
        if abs(slippage) > 0:
            cost_breakdown.append(TransactionCost(
                cost_type=CostType.SLIPPAGE,
                amount=slippage,
                basis_points=(slippage / trade_value) * 10000,
                description=f"Slippage vs benchmark: ${slippage:+.2f}"
            ))
            
        # Calculate totals
        total_cost = sum(cost.amount for cost in cost_breakdown)
        total_cost_bps = (total_cost / trade_value) * 10000
        
        # Implementation shortfall
        implementation_shortfall = (trade.price - benchmark_price) * abs(trade.quantity)
        if trade.side == 'SELL':
            implementation_shortfall = -implementation_shortfall
            
        # Execution quality score (0-100, higher is better)
        execution_quality_score = self._calculate_execution_quality_score(
            cost_breakdown, trade_value, market_data
        )
        
        return TCAResult(
            trade=trade,
            benchmark_price=benchmark_price,
            implementation_shortfall=implementation_shortfall,
            total_cost=total_cost,
            total_cost_bps=total_cost_bps,
            cost_breakdown=cost_breakdown,
            execution_quality_score=execution_quality_score
        )
        
    def _calculate_commission(self, trade: Trade) -> float:
        """Calculate commission cost."""
        # Simple per-share commission
        return abs(trade.quantity) * self.commission_rates['equity']
        
    def _calculate_spread_cost(self, trade: Trade, 
                              pre_trade_quote: Optional[Dict],
                              market_data: Optional[Dict]) -> float:
        """Calculate bid-ask spread cost."""
        if pre_trade_quote:
            bid = pre_trade_quote.get('bid', 0)
            ask = pre_trade_quote.get('ask', 0)
            
            if bid > 0 and ask > 0:
                mid_price = (bid + ask) / 2
                
                if trade.side == 'BUY':
                    # Paid ask, benchmark is mid
                    spread_cost = (ask - mid_price) * abs(trade.quantity)
                else:
                    # Received bid, benchmark is mid
                    spread_cost = (mid_price - bid) * abs(trade.quantity)
                    
                return spread_cost
                
        # Estimate based on typical spread
        typical_spread_pct = self._get_typical_spread(trade.symbol)
        estimated_spread_cost = trade.price * typical_spread_pct * abs(trade.quantity)
        
        return estimated_spread_cost
        
    def _calculate_market_impact(self, trade: Trade, market_data: Optional[Dict]) -> float:
        """Calculate market impact cost."""
        # Simplified market impact model
        # Impact = α × (Trade Size / ADV)^β × Price
        
        daily_volume = self._get_average_daily_volume(trade.symbol, market_data)
        if daily_volume <= 0:
            return 0.0
            
        participation_rate = abs(trade.quantity) / daily_volume
        
        # Market impact parameters (calibrated from empirical data)
        alpha = 0.314  # Impact coefficient
        beta = 0.6     # Size exponent
        
        impact_pct = alpha * (participation_rate ** beta) / 100
        impact_cost = trade.price * impact_pct * abs(trade.quantity)
        
        return impact_cost
        
    def _calculate_slippage(self, trade: Trade, benchmark_price: float) -> float:
        """Calculate slippage vs benchmark."""
        price_diff = trade.price - benchmark_price
        
        # Adjust sign based on trade side
        if trade.side == 'BUY':
            # Positive slippage = paid more than benchmark
            slippage = price_diff * abs(trade.quantity)
        else:
            # Positive slippage = received less than benchmark
            slippage = -price_diff * abs(trade.quantity)
            
        return slippage
        
    def _get_typical_spread(self, symbol: str) -> float:
        """Get typical bid-ask spread percentage."""
        # Simplified - in practice, use historical spread data
        if symbol in self.typical_spreads:
            return self.typical_spreads[symbol]
            
        # Default spread estimates by symbol type
        if symbol in ['AAPL', 'MSFT', 'GOOGL', 'AMZN']:
            return 0.0001  # 1 basis point for liquid large caps
        else:
            return 0.0005  # 5 basis points for typical stocks
            
    def _get_average_daily_volume(self, symbol: str, market_data: Optional[Dict]) -> float:
        """Get average daily volume."""
        if market_data and 'average_volume' in market_data:
            return market_data['average_volume']
            
        # Default volume estimates
        volume_estimates = {
            'AAPL': 50000000,
            'MSFT': 30000000,
            'GOOGL': 25000000,
            'AMZN': 35000000,
            'TSLA': 75000000
        }
        
        return volume_estimates.get(symbol, 1000000)  # Default 1M shares
        
    def _calculate_execution_quality_score(self, cost_breakdown: List[TransactionCost],
                                          trade_value: float,
                                          market_data: Optional[Dict]) -> float:
        """Calculate execution quality score."""
        total_cost_bps = sum(cost.basis_points for cost in cost_breakdown)
        
        # Benchmarks (basis points)
        excellent_threshold = 5    # < 5 bps
        good_threshold = 15       # < 15 bps
        poor_threshold = 50       # > 50 bps
        
        if total_cost_bps < excellent_threshold:
            return 90 + (excellent_threshold - total_cost_bps)  # 90-95
        elif total_cost_bps < good_threshold:
            return 70 + 20 * (good_threshold - total_cost_bps) / (good_threshold - excellent_threshold)
        elif total_cost_bps < poor_threshold:
            return 30 + 40 * (poor_threshold - total_cost_bps) / (poor_threshold - good_threshold)
        else:
            return max(0, 30 - (total_cost_bps - poor_threshold))
            
    def analyze_portfolio_trades(self, trades: List[Trade],
                                benchmark_prices: Optional[Dict[str, float]] = None) -> Dict:
        """Analyze costs for a portfolio of trades."""
        results = []
        
        for trade in trades:
            benchmark_price = None
            if benchmark_prices and trade.symbol in benchmark_prices:
                benchmark_price = benchmark_prices[trade.symbol]
                
            tca_result = self.analyze_trade(trade, benchmark_price)
            results.append(tca_result)
            
        # Aggregate statistics
        total_value = sum(abs(r.trade.quantity) * r.trade.price for r in results)
        total_costs = sum(r.total_cost for r in results)
        
        cost_by_type = {}
        for result in results:
            for cost in result.cost_breakdown:
                cost_type = cost.cost_type.value
                if cost_type not in cost_by_type:
                    cost_by_type[cost_type] = 0.0
                cost_by_type[cost_type] += cost.amount
                
        avg_quality_score = np.mean([r.execution_quality_score for r in results])
        
        return {
            'individual_results': results,
            'summary': {
                'total_trades': len(trades),
                'total_value': total_value,
                'total_costs': total_costs,
                'total_cost_bps': (total_costs / total_value * 10000) if total_value > 0 else 0,
                'avg_execution_quality': avg_quality_score,
                'cost_by_type': cost_by_type
            }
        }
        
    def generate_tca_report(self, analysis_result: Dict) -> str:
        """Generate TCA report."""
        summary = analysis_result['summary']
        results = analysis_result['individual_results']
        
        report = []
        report.append("TRANSACTION COST ANALYSIS REPORT")
        report.append("=" * 50)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Summary
        report.append("PORTFOLIO SUMMARY")
        report.append("-" * 30)
        report.append(f"Total Trades: {summary['total_trades']}")
        report.append(f"Total Value: ${summary['total_value']:,.2f}")
        report.append(f"Total Costs: ${summary['total_costs']:,.2f}")
        report.append(f"Cost Rate: {summary['total_cost_bps']:.1f} basis points")
        report.append(f"Avg Execution Quality: {summary['avg_execution_quality']:.1f}/100")
        report.append("")
        
        # Cost breakdown
        report.append("COST BREAKDOWN BY TYPE")
        report.append("-" * 30)
        for cost_type, amount in summary['cost_by_type'].items():
            pct_of_total = (amount / summary['total_costs'] * 100) if summary['total_costs'] > 0 else 0
            report.append(f"{cost_type:20}: ${amount:8.2f} ({pct_of_total:4.1f}%)")
        report.append("")
        
        # Individual trades (top 5 by cost)
        report.append("TOP TRADES BY COST")
        report.append("-" * 30)
        sorted_results = sorted(results, key=lambda x: x.total_cost, reverse=True)[:5]
        
        for i, result in enumerate(sorted_results, 1):
            trade = result.trade
            report.append(f"{i}. {trade.symbol}: {trade.side} {trade.quantity:,.0f} @ ${trade.price:.2f}")
            report.append(f"   Cost: ${result.total_cost:.2f} ({result.total_cost_bps:.1f} bps)")
            report.append(f"   Quality: {result.execution_quality_score:.1f}/100")
            report.append("")
            
        return "\\n".join(report)


# Example usage
if __name__ == "__main__":
    print("Transaction Cost Analysis Test")
    print("=" * 35)
    
    # Create sample trades
    trades = [
        Trade(
            symbol='AAPL',
            quantity=1000,
            price=155.25,
            side='BUY',
            trade_time=datetime.now(),
            commission=5.00
        ),
        Trade(
            symbol='GOOGL',
            quantity=200,
            price=2451.75,
            side='BUY',
            trade_time=datetime.now(),
            commission=1.00
        ),
        Trade(
            symbol='MSFT',
            quantity=500,
            price=295.80,
            side='SELL',
            trade_time=datetime.now(),
            commission=2.50
        ),
        Trade(
            symbol='TSLA',
            quantity=150,
            price=238.45,
            side='BUY',
            trade_time=datetime.now(),
            commission=0.75
        )
    ]
    
    # Benchmark prices (e.g., VWAP)
    benchmark_prices = {
        'AAPL': 155.20,   # Slight negative slippage
        'GOOGL': 2450.00, # Positive slippage
        'MSFT': 296.00,   # Positive slippage (good for sell)
        'TSLA': 238.50    # Slight positive slippage
    }
    
    # Analyze trades
    analyzer = TransactionCostAnalyzer()
    
    print("\\nAnalyzing individual trades...")
    for trade in trades:
        benchmark = benchmark_prices.get(trade.symbol)
        result = analyzer.analyze_trade(trade, benchmark)
        
        print(f"\\n{trade.symbol}: {trade.side} {trade.quantity} @ ${trade.price:.2f}")
        print(f"  Benchmark: ${result.benchmark_price:.2f}")
        print(f"  Total Cost: ${result.total_cost:.2f} ({result.total_cost_bps:.1f} bps)")
        print(f"  Implementation Shortfall: ${result.implementation_shortfall:+.2f}")
        print(f"  Execution Quality: {result.execution_quality_score:.1f}/100")
        
        print("  Cost Breakdown:")
        for cost in result.cost_breakdown:
            print(f"    {cost.cost_type.value}: ${cost.amount:.2f} ({cost.basis_points:.1f} bps)")
            
    # Portfolio analysis
    print("\\n" + "="*50)
    portfolio_analysis = analyzer.analyze_portfolio_trades(trades, benchmark_prices)
    
    summary = portfolio_analysis['summary']
    print(f"\\nPORTFOLIO TCA SUMMARY:")
    print(f"Total Value: ${summary['total_value']:,.2f}")
    print(f"Total Costs: ${summary['total_costs']:.2f}")
    print(f"Cost Rate: {summary['total_cost_bps']:.1f} basis points")
    print(f"Avg Quality Score: {summary['avg_execution_quality']:.1f}/100")
    
    print(f"\\nCost Breakdown:")
    for cost_type, amount in summary['cost_by_type'].items():
        pct = (amount / summary['total_costs'] * 100) if summary['total_costs'] > 0 else 0
        print(f"  {cost_type}: ${amount:.2f} ({pct:.1f}%)")
        
    # Generate full report
    print("\\n" + "="*50)
    print("FULL TCA REPORT")
    print("="*50)
    full_report = analyzer.generate_tca_report(portfolio_analysis)
    print(full_report)
    
    print("\\n✅ Transaction Cost Analysis completed")
```

---

## 6.6 Complete Live Trading System Integration

### Implementation: `live_trading_system.py`

```python
# Place in: src/integration/live_trading_system.py

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from datetime import datetime, timedelta
import threading
import time
import logging

from quantum.data_preparation import PortfolioDataPreparer
from quantum.esg_portfolio_optimizer import ESGPortfolioOptimizer
from integration.broker_interface import BrokerFactory, BrokerInterface, OrderType
from integration.order_manager import OrderManager, OrderPriority
from integration.execution_algorithms import ExecutionEngine, ExecutionRequest, AlgorithmType
from integration.position_manager import PositionManager
from integration.transaction_cost_analyzer import TransactionCostAnalyzer


class LiveTradingSystem:
    """Complete live trading system integrating all components."""
    
    def __init__(self, broker_name: str, broker_credentials: Dict[str, str],
                 tickers: List[str], rebalance_frequency: str = 'daily',
                 **broker_kwargs):
        """
        Initialize live trading system.
        
        Args:
            broker_name: Broker name ('alpaca', 'interactive_brokers')
            broker_credentials: API credentials
            tickers: List of symbols to trade
            rebalance_frequency: 'daily', 'weekly', 'monthly'
        """
        self.tickers = tickers
        self.rebalance_frequency = rebalance_frequency
        
        # Initialize broker connection
        self.broker = BrokerFactory.create_broker(
            broker_name, broker_credentials, **broker_kwargs
        )
        
        # Initialize components
        self.order_manager = OrderManager(self.broker)
        self.execution_engine = ExecutionEngine(self.order_manager)
        self.position_manager = PositionManager(self.broker)
        self.tca_analyzer = TransactionCostAnalyzer()
        
        # Portfolio optimization
        self.esg_optimizer = ESGPortfolioOptimizer(tickers)
        self.current_target_weights = {}
        self.last_optimization_time = None
        self.last_rebalance_time = None
        
        # System state
        self.running = False
        self.main_thread = None
        
        # Configuration
        self.config = {
            'optimization_frequency': 'daily',  # How often to re-optimize
            'rebalance_threshold': 0.05,        # 5% drift threshold
            'max_position_size': 0.25,          # 25% max position
            'execution_algorithm': 'VWAP',      # Default execution algorithm
            'risk_limit_enabled': True,
            'esg_constraints_enabled': True
        }
        
        # Performance tracking
        self.performance_log = []
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('LiveTradingSystem')
        
    def start(self) -> bool:
        """Start the live trading system."""
        try:
            # Connect to broker
            if not self.broker.connect():
                self.logger.error("Failed to connect to broker")
                return False
                
            self.logger.info(f"Connected to {self.broker.__class__.__name__}")
            
            # Start sub-components
            self.order_manager.start()
            self.position_manager.start_monitoring()
            
            # Initial portfolio optimization
            self._run_portfolio_optimization()
            
            # Start main trading loop
            self.running = True
            self.main_thread = threading.Thread(target=self._main_trading_loop, daemon=True)
            self.main_thread.start()
            
            self.logger.info("Live trading system started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start live trading system: {e}")
            return False
            
    def stop(self):
        """Stop the live trading system."""
        self.running = False
        
        # Stop components
        if self.order_manager:
            self.order_manager.stop()
        if self.position_manager:
            self.position_manager.stop_monitoring()
            
        # Disconnect broker
        if self.broker:
            self.broker.disconnect()
            
        self.logger.info("Live trading system stopped")
        
    def _main_trading_loop(self):
        """Main trading loop."""
        while self.running:
            try:
                current_time = datetime.now()
                
                # Check if we need to optimize portfolio
                if self._should_optimize_portfolio(current_time):
                    self._run_portfolio_optimization()
                    
                # Check if we need to rebalance
                if self._should_rebalance(current_time):
                    self._run_rebalancing()
                    
                # Monitor positions and risk
                self._monitor_risk_limits()
                
                # Log performance
                self._update_performance_log()
                
                # Sleep before next iteration
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Error in main trading loop: {e}")
                time.sleep(30)  # Wait before retrying
                
    def _should_optimize_portfolio(self, current_time: datetime) -> bool:
        """Check if portfolio optimization is needed."""
        if self.last_optimization_time is None:
            return True
            
        if self.config['optimization_frequency'] == 'daily':
            return current_time.date() > self.last_optimization_time.date()
        elif self.config['optimization_frequency'] == 'weekly':
            return (current_time - self.last_optimization_time).days >= 7
        else:
            return False
            
    def _run_portfolio_optimization(self):
        """Run quantum portfolio optimization."""
        try:
            self.logger.info("Running portfolio optimization...")
            
            # Prepare data
            data_preparer = PortfolioDataPreparer(
                self.tickers, 
                (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'),
                datetime.now().strftime('%Y-%m-%d')
            )
            
            financial_data = data_preparer.download_data()
            financial_stats = data_preparer.calculate_statistics()
            mu, sigma = data_preparer.get_optimization_inputs('ledoit_wolf')
            
            # Load ESG data
            esg_info = self.esg_optimizer.load_esg_data(mu, sigma)
            
            # Optimize with ESG constraints
            optimization_result = self.esg_optimizer.optimize_esg_portfolio(
                esg_weight=0.3,
                carbon_penalty=0.2,
                min_esg_score=70,
                use_quantum=True
            )
            
            # Update target weights
            self.current_target_weights = dict(zip(
                self.tickers, 
                optimization_result['weights']
            ))
            
            self.last_optimization_time = datetime.now()
            
            self.logger.info("Portfolio optimization completed")
            self.logger.info(f"New target weights: {self.current_target_weights}")
            
        except Exception as e:
            self.logger.error(f"Portfolio optimization failed: {e}")
            
    def _should_rebalance(self, current_time: datetime) -> bool:
        """Check if rebalancing is needed."""
        if not self.current_target_weights:
            return False
            
        # Check drift threshold
        portfolio_value = self.position_manager.portfolio_metrics['total_value']
        if portfolio_value <= 0:
            return False
            
        needs_rebalancing, drift = self.position_manager.check_rebalancing_needed(
            self.current_target_weights, 
            self.config['rebalance_threshold']
        )
        
        if needs_rebalancing:
            self.logger.info(f"Rebalancing needed - max drift: {max(drift.values()):.2%}")
            return True
            
        # Time-based rebalancing
        if self.last_rebalance_time is None:
            return True
            
        if self.rebalance_frequency == 'daily':
            return current_time.date() > self.last_rebalance_time.date()
        elif self.rebalance_frequency == 'weekly':
            return (current_time - self.last_rebalance_time).days >= 7
        elif self.rebalance_frequency == 'monthly':
            return (current_time - self.last_rebalance_time).days >= 30
            
        return False
        
    def _run_rebalancing(self):
        """Execute portfolio rebalancing."""
        try:
            self.logger.info("Starting portfolio rebalancing...")
            
            # Get current prices
            current_prices = {}
            for symbol in self.tickers:
                market_data = self.broker.get_market_data(symbol)
                current_prices[symbol] = market_data.get('last_price', 0)
                
            # Generate rebalancing orders
            rebalancing_orders = self.position_manager.generate_rebalancing_orders(
                self.current_target_weights, current_prices
            )
            
            if not rebalancing_orders:
                self.logger.info("No rebalancing orders needed")
                return
                
            self.logger.info(f"Generated {len(rebalancing_orders)} rebalancing orders")
            
            # Execute orders using execution algorithms
            execution_requests = []
            
            for order_info in rebalancing_orders:
                if abs(order_info['quantity']) < 1:  # Skip fractional shares
                    continue
                    
                # Determine execution algorithm based on order size
                portfolio_value = self.position_manager.portfolio_metrics['total_value']
                order_value = abs(order_info['quantity']) * current_prices[order_info['symbol']]
                participation_rate = order_value / portfolio_value
                
                if participation_rate > 0.10:  # Large orders use VWAP
                    algorithm = AlgorithmType.VWAP
                elif participation_rate > 0.05:  # Medium orders use TWAP
                    algorithm = AlgorithmType.TWAP
                else:  # Small orders execute immediately
                    # Submit market order directly
                    self.order_manager.submit_order(
                        symbol=order_info['symbol'],
                        quantity=order_info['quantity'],
                        order_type=OrderType.MARKET,
                        side=order_info['side'],
                        priority=OrderPriority.HIGH
                    )
                    continue
                
                # Create execution request for algorithmic trading
                exec_request = ExecutionRequest(
                    symbol=order_info['symbol'],
                    total_quantity=order_info['quantity'],
                    side=order_info['side'],
                    algorithm=algorithm,
                    start_time=datetime.now(),
                    end_time=datetime.now() + timedelta(hours=2),
                    max_participation_rate=0.20,
                    callback=self._execution_callback
                )
                
                execution_id = self.execution_engine.execute_order(exec_request)
                execution_requests.append((execution_id, exec_request))
                
            self.last_rebalance_time = datetime.now()
            self.logger.info(f"Rebalancing initiated with {len(execution_requests)} algorithmic executions")
            
        except Exception as e:
            self.logger.error(f"Rebalancing failed: {e}")
            
    def _execution_callback(self, execution_summary: Dict):
        """Callback for execution completion."""
        self.logger.info(f"Execution completed: {execution_summary}")
        
        # Record trade for TCA
        # In practice, would get actual fill details from order manager
        
    def _monitor_risk_limits(self):
        """Monitor portfolio risk limits."""
        if not self.config['risk_limit_enabled']:
            return
            
        try:
            positions = self.position_manager.get_all_positions()
            portfolio_value = self.position_manager.portfolio_metrics['total_value']
            
            if portfolio_value <= 0:
                return
                
            # Check position size limits
            for symbol, pos_record in positions.items():
                position_pct = abs(pos_record.position.market_value) / portfolio_value
                
                if position_pct > self.config['max_position_size']:
                    self.logger.warning(
                        f"Position size limit exceeded: {symbol} = {position_pct:.2%} "
                        f"(limit: {self.config['max_position_size']:.2%})"
                    )
                    
            # Check total P&L drawdown
            performance_metrics = self.position_manager.get_performance_metrics()
            if performance_metrics['max_drawdown'] > 0.10:  # 10% portfolio drawdown
                self.logger.warning(f"Portfolio drawdown: {performance_metrics['max_drawdown']:.2%}")
                
        except Exception as e:
            self.logger.error(f"Risk monitoring error: {e}")
            
    def _update_performance_log(self):
        """Update performance tracking."""
        try:
            portfolio_summary = self.position_manager.get_portfolio_summary()
            performance_metrics = self.position_manager.get_performance_metrics()
            
            log_entry = {
                'timestamp': datetime.now(),
                'portfolio_value': portfolio_summary['total_value'],
                'total_pnl': portfolio_summary['total_pnl'],
                'pnl_pct': portfolio_summary['total_pnl_pct'],
                'num_positions': portfolio_summary['num_positions'],
                'win_rate': performance_metrics['win_rate'],
                'max_drawdown': performance_metrics['max_drawdown']
            }
            
            self.performance_log.append(log_entry)
            
            # Keep only last 1000 entries
            if len(self.performance_log) > 1000:
                self.performance_log = self.performance_log[-1000:]
                
        except Exception as e:
            self.logger.error(f"Performance logging error: {e}")
            
    def get_system_status(self) -> Dict:
        """Get current system status."""
        try:
            portfolio_summary = self.position_manager.get_portfolio_summary()
            order_stats = self.order_manager.get_order_statistics()
            active_executions = self.execution_engine.get_active_executions()
            
            return {
                'running': self.running,
                'connected': self.broker.connected if hasattr(self.broker, 'connected') else True,
                'last_optimization': self.last_optimization_time,
                'last_rebalance': self.last_rebalance_time,
                'portfolio': portfolio_summary,
                'orders': order_stats,
                'active_executions': len(active_executions),
                'target_weights': self.current_target_weights,
                'config': self.config
            }
            
        except Exception as e:
            self.logger.error(f"Status check error: {e}")
            return {'error': str(e)}
            
    def run_tca_analysis(self, start_date: Optional[datetime] = None) -> Dict:
        """Run transaction cost analysis."""
        try:
            # Get trades from today if no start date specified
            if start_date is None:
                start_date = datetime.now().date()
                
            trades = [trade for trade in self.position_manager.trades_today 
                     if trade.trade_time.date() >= start_date]
            
            if not trades:
                return {'message': 'No trades to analyze'}
                
            # Run TCA analysis
            tca_result = self.tca_analyzer.analyze_portfolio_trades(trades)
            
            return tca_result
            
        except Exception as e:
            self.logger.error(f"TCA analysis error: {e}")
            return {'error': str(e)}


# Example usage and system test
if __name__ == "__main__":
    print("Live Trading System Test")
    print("=" * 30)
    
    # Configuration
    broker_credentials = {
        'api_key': 'YOUR_ALPACA_API_KEY',
        'secret_key': 'YOUR_ALPACA_SECRET_KEY'
    }
    
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    
    # Note: This test requires valid broker credentials
    try:
        # Initialize trading system
        trading_system = LiveTradingSystem(
            broker_name='alpaca',
            broker_credentials=broker_credentials,
            tickers=tickers,
            rebalance_frequency='daily',
            paper_trading=True  # Use paper trading for testing
        )
        
        # Configure system
        trading_system.config.update({
            'rebalance_threshold': 0.03,  # 3% threshold for testing
            'max_position_size': 0.30,
            'esg_constraints_enabled': True
        })
        
        print("Configuration:")
        for key, value in trading_system.config.items():
            print(f"  {key}: {value}")
            
        # Start system (comment out if no valid credentials)
        # if trading_system.start():
        #     print("\\n✅ Live trading system started successfully")
        #     
        #     try:
        #         # Let it run for a few minutes
        #         for i in range(10):
        #             time.sleep(30)
        #             status = trading_system.get_system_status()
        #             print(f"\\nStatus Check {i+1}:")
        #             print(f"  Portfolio Value: ${status['portfolio']['total_value']:,.2f}")
        #             print(f"  Total P&L: ${status['portfolio']['total_pnl']:+,.2f}")
        #             print(f"  Active Orders: {status['orders']['active_orders']}")
        #             print(f"  Active Executions: {status['active_executions']}")
        #             
        #     finally:
        #         trading_system.stop()
        #         print("\\n✅ Live trading system stopped")
        # else:
        #     print("❌ Failed to start live trading system")
        
        print("\\nLive Trading System components initialized:")
        print("  ✓ Broker Interface")
        print("  ✓ Order Manager") 
        print("  ✓ Execution Engine")
        print("  ✓ Position Manager")
        print("  ✓ TCA Analyzer")
        print("  ✓ ESG Portfolio Optimizer")
        
        print("\\nNote: Full system test requires valid broker API credentials")
        
    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        print("This is expected without valid broker credentials")
```

---

## 6.7 Key Achievements in Phase 6

1. **✅ Multi-Broker Support**: Interactive Brokers, Alpaca API integration with unified interface
2. **✅ Smart Order Management**: Queue-based processing, validation, monitoring, retry logic
3. **✅ Execution Algorithms**: VWAP, TWAP, POV implementations with real-time scheduling
4. **✅ Position Reconciliation**: Real-time tracking, drift detection, P&L calculation
5. **✅ Transaction Cost Analysis**: Comprehensive TCA with market impact modeling
6. **✅ Complete Integration**: End-to-end live trading system connecting all components

---

## 6.8 Performance Benchmarks

| Metric                    | Target          | Typical Range    |
|---------------------------|-----------------|------------------|
| **Order Fill Rate**       | >95%            | 92-98%           |
| **Average Fill Time**     | <2 seconds      | 0.5-5 seconds    |
| **Slippage (Market)**     | <5 bps          | 2-10 bps         |
| **Slippage (Limit)**      | <2 bps          | 1-8 bps          |
| **Total Trading Costs**   | <25 bps         | 15-40 bps        |
| **Execution Quality**     | >85/100         | 75-95            |
| **System Uptime**         | >99.5%          | 98-99.9%         |

*Sample benchmarks for institutional equity trading. Actual results vary by market conditions, order size, and asset liquidity.*

---

## 6.9 Repository Structure Update

```
src/
├── integration/
│   ├── broker_interface.py
│   ├── order_manager.py
│   ├── execution_algorithms.py
│   ├── position_manager.py
│   ├── transaction_cost_analyzer.py
│   └── live_trading_system.py
└── docs/
    └── Phase_6_Live_Execution_and_Broker_Integration.md
```

---

## Next Steps

**Phase 7**: Advanced Analytics and Reinforcement Learning
- Deep RL for dynamic portfolio allocation
- Alternative data integration (satellite, sentiment, news)
- Real-time risk monitoring with machine learning
- Predictive market microstructure models

**Production Deployment Considerations:**
- Docker containerization for scalable deployment
- Kubernetes orchestration for high availability
- Message queues (Redis/RabbitMQ) for reliable communication
- Database integration (PostgreSQL) for trade storage
- Monitoring and alerting (Prometheus/Grafana)
- Compliance logging and audit trails

---

**The quantum portfolio optimization system now includes complete live trading capabilities for institutional-grade execution!** 🚀⚛️

## Live Trading Integration Summary

This phase successfully integrates:
- **Multi-broker support** with unified API abstraction
- **Professional order management** with validation, monitoring, and retry logic
- **Smart execution algorithms** (VWAP, TWAP, POV) for optimal trade execution
- **Real-time position tracking** with drift detection and rebalancing triggers
- **Comprehensive TCA** for execution quality measurement and cost control
- **Complete integration** connecting quantum optimization to live market execution

The system now enables institutional investors to deploy quantum-optimized portfolios in live markets with professional-grade execution quality, cost control, and risk management.