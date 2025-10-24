# src/integration/order_manager.py
"""
Order management system with fixed imports
"""

import numpy as np
import pandas as pd
import sys
import os
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import threading
import time
import queue
import uuid

# Fix imports by adding project paths
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
src_dir = current_dir.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Now import from broker_interface with relative import
try:
    from integration.broker_interface import BrokerInterface, Order, OrderType, OrderStatus, Position
except ImportError:
    # Fallback to direct import
    from broker_interface import BrokerInterface, Order, OrderType, OrderStatus, Position


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
            account_info = self.broker.get_account()
            
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
        self.lock = threading.Lock()
        
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
        
        print("✅ Order Manager started")
        
    def stop(self):
        """Stop order management."""
        self.running = False
        
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
            
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
            
        print("✅ Order Manager stopped")
        
    def submit_order_request(self, request: OrderRequest) -> bool:
        """Submit order request for processing."""
        # Validate order
        validation = self.validator.validate_order(request.order)
        
        if not validation.is_valid:
            print(f"❌ Order validation failed: {validation}")
            request.order.status = OrderStatus.REJECTED
            self._complete_order(request)
            return False
            
        # Add to queue with priority
        priority_value = (5 - request.priority.value, datetime.now().timestamp())
        self.order_queue.put((priority_value, request))
        
        print(f"✅ Order request queued: {request.order.side} {request.order.quantity} {request.order.symbol}")
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
            return getattr(order, 'order_id', None)
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
                with self.lock:
                    self.active_orders[order_id] = request
                    self.stats['orders_submitted'] += 1
                
                # Call callback if provided
                if request.callback:
                    try:
                        request.callback(order, 'submitted')
                    except Exception as e:
                        print(f"Callback error: {e}")
                        
                print(f"✅ Order submitted: {order_id} - {order.side} {order.quantity} {order.symbol}")
                return
                
            except Exception as e:
                retries += 1
                print(f"❌ Order submission failed (attempt {retries}): {e}")
                
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
                with self.lock:
                    active_order_ids = list(self.active_orders.keys())
                
                for order_id in active_order_ids:
                    request = self.active_orders.get(order_id)
                    if request:
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
                    order.filled_quantity = abs(order.quantity)
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
        
        with self.lock:
            if order_id and order_id in self.active_orders:
                del self.active_orders[order_id]
                
            if order_id:
                self.completed_orders[order_id] = request
            self.order_history.append(request)
        
        if order_id:
            print(f"✅ Order completed: {order_id} - Status: {request.order.status.value}")


# Example usage
if __name__ == "__main__":
    print("Order Manager Test")
    print("=" * 25)
    
    # Mock broker for testing
    class MockBroker(BrokerInterface):
        def __init__(self):
            super().__init__({})
            self.connected = True
            self.order_counter = 1000
            self._order_times = {}
            
        def connect(self) -> bool:
            return True
            
        def disconnect(self):
            pass
            
        def submit_order(self, order: Order) -> str:
            order_id = f"ORDER_{self.order_counter}"
            self.order_counter += 1
            self._order_times[order_id] = time.time()
            return order_id
            
        def cancel_order(self, order_id: str) -> bool:
            return True
            
        def get_order_status(self, order_id: str) -> OrderStatus:
            # Simulate order filling after 3 seconds
            if order_id in self._order_times:
                if time.time() - self._order_times[order_id] > 3:
                    return OrderStatus.FILLED
                else:
                    return OrderStatus.SUBMITTED
            return OrderStatus.SUBMITTED
            
        def get_positions(self) -> List[Position]:
            return []
            
        def get_account(self) -> Dict:
            return {'buying_power': 100000, 'portfolio_value': 500000}
            
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
            print(f"  📢 Callback: {order.symbol} order {status}")
            
        print("\n📝 Submitting test orders...")
        
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
        print("\n👀 Monitoring orders...")
        for i in range(10):
            time.sleep(1)
            stats = order_manager.get_order_statistics()
            active = len(order_manager.get_active_orders())
            print(f"  Active: {active}, Submitted: {stats['orders_submitted']}, Filled: {stats['orders_filled']}")
            
            if stats['orders_filled'] >= 2:
                break
                
        # Final statistics
        final_stats = order_manager.get_order_statistics()
        print(f"\n📊 Final Statistics:")
        print(f"  Orders Submitted: {final_stats['orders_submitted']}")
        print(f"  Orders Filled: {final_stats['orders_filled']}")
        print(f"  Fill Rate: {final_stats['fill_rate']:.1%}")
        print(f"  Total Volume: {final_stats['total_volume']}")
        
    finally:
        order_manager.stop()
        print("\n✅ Order Manager test completed")
