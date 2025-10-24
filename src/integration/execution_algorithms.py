# src/integration/execution_algorithms.py
"""
Execution algorithms with fixed imports
"""

import numpy as np
import pandas as pd
import sys
import os
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, time
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
import threading
import time as time_module

# Fix imports by adding project paths
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
src_dir = current_dir.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Now import with try/except for robustness
try:
    from integration.order_manager import OrderManager, Order, OrderType, OrderPriority
except ImportError:
    from order_manager import OrderManager, Order, OrderType, OrderPriority


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
        
        print(f"✅ Started {self.get_algorithm_name()} execution: {execution_id}")
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
                
                print(f"  ✅ Executed slice {i+1}/{len(schedule)}: {slice_order.quantity} @ ${slice_order.avg_price:.2f}")
                
            except Exception as e:
                print(f"  ❌ Failed to execute slice {i+1}: {e}")
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
            
        print(f"✅ Completed execution {execution_id}: {total_executed}/{request.total_quantity} @ ${avg_price:.2f}")
        
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
            
            expected_volume_pct = volume_profile.get(interval_hour, 0.05)  # Default 5%
            
            # Calculate slice size based on volume profile
            slice_quantity = min(
                remaining_quantity,
                request.total_quantity * expected_volume_pct * request.max_participation_rate
            )
            
            if slice_quantity > 0:
                slice_order = ExecutionSlice(
                    quantity=slice_quantity,
                    target_time=current_time,
                    price_limit=request.price_limit
                )
                schedule.append(slice_order)
                
                remaining_quantity -= slice_quantity
            
            current_time += timedelta(minutes=interval_minutes)
            
        return schedule
        
    def _get_volume_profile(self, symbol: str) -> Dict[int, float]:
        """Get historical intraday volume profile."""
        # Simplified volume profile (in practice, get from market data)
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
    print("Execution Algorithms Test")
    print("=" * 30)
    
    # Mock broker and order manager for testing
    from integration.broker_interface import BrokerInterface, Position
    
    class MockBroker(BrokerInterface):
        def __init__(self):
            super().__init__({})
            self.connected = True
            
        def connect(self) -> bool:
            return True
        def disconnect(self):
            pass
        def submit_order(self, order):
            return f"ORDER_{np.random.randint(1000, 9999)}"
        def cancel_order(self, order_id: str) -> bool:
            return True
        def get_order_status(self, order_id: str):
            from integration.broker_interface import OrderStatus
            return OrderStatus.FILLED
        def get_positions(self):
            return []
        def get_account_info(self):
            return {'buying_power': 100000}
        def get_market_data(self, symbol: str):
            return {'symbol': symbol, 'last_price': 100.0 + np.random.randn()}
    
    broker = MockBroker()
    order_manager = OrderManager(broker)
    order_manager.start()
    
    execution_engine = ExecutionEngine(order_manager)
    
    try:
        # Test VWAP execution
        def execution_callback(summary):
            print(f"\n✅ Execution completed: {summary}")
            
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
        
        print(f"\n🚀 Starting VWAP execution: {vwap_request.total_quantity} {vwap_request.symbol}")
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
        
        print(f"🚀 Starting TWAP execution: {twap_request.total_quantity} {twap_request.symbol}")
        execution_id2 = execution_engine.execute_order(twap_request)
        
        # Monitor executions
        print("\n👀 Monitoring executions...")
        for i in range(15):
            active = execution_engine.get_active_executions()
            print(f"  Active executions: {len(active)}")
            for exec_info in active:
                print(f"    {exec_info['algorithm']}: {exec_info['symbol']} - {exec_info['total_quantity']}")
            
            if len(active) == 0:
                break
                
            time_module.sleep(2)
            
        print("\n✅ Execution algorithms test completed")
        
    finally:
        order_manager.stop()
