# src/integration/position_manager.py
"""
Position management and reconciliation with fixed imports
"""

import numpy as np
import pandas as pd
import sys
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import threading
import time

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
    from integration.broker_interface import BrokerInterface, Position
except ImportError:
    from broker_interface import BrokerInterface, Position


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
        print("✅ Position monitoring started")
        
    def stop_monitoring(self):
        """Stop position monitoring."""
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        print("✅ Position monitoring stopped")
        
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
                # Create empty position record
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
                
                if pos_record.position.avg_cost > 0:
                    trade_pnl = abs(trade.quantity) * (trade.price - pos_record.position.avg_cost)
                    if trade.side == 'SELL':
                        pos_record.realized_pnl_today += trade_pnl
                    else:
                        pos_record.realized_pnl_today -= trade_pnl
                        
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
                account_info = self.broker.get_account()
                
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
    from integration.broker_interface import BrokerInterface
    
    print("Position Manager Test")
    print("=" * 25)
    
    # Mock broker for testing
    class MockBroker(BrokerInterface):
        def __init__(self):
            super().__init__({})
            self.connected = True
            self.mock_positions = [
                Position('AAPL', 100, 150.0, 155.0, 15500, 500),
                Position('GOOGL', 50, 2400.0, 2450.0, 122500, 2500),
                Position('MSFT', -25, 300.0, 295.0, -7375, 125)
            ]
            
        def connect(self) -> bool:
            return True
        def disconnect(self):
            pass
        def submit_order(self, order):
            return "ORDER_123"
        def cancel_order(self, order_id: str) -> bool:
            return True
        def get_order_status(self, order_id: str):
            from integration.broker_interface import OrderStatus
            return OrderStatus.FILLED
        def get_positions(self) -> List[Position]:
            return self.mock_positions
        def get_account(self) -> Dict:
            return {
                'cash': 50000,
                'buying_power': 100000,
                'portfolio_value': 180625
            }
        def get_market_data(self, symbol: str) -> Dict:
            prices = {'AAPL': 155.0, 'GOOGL': 2450.0, 'MSFT': 295.0}
            return {'symbol': symbol, 'last_price': prices.get(symbol, 100.0)}
    
    # Setup
    broker = MockBroker()
    position_manager = PositionManager(broker, update_interval=5)
    
    try:
        # Start monitoring
        position_manager.start_monitoring()
        
        # Wait for initial update
        time.sleep(2)
        
        # Check positions
        print("\n📊 Current Positions:")
        positions = position_manager.get_all_positions()
        for symbol, pos_record in positions.items():
            pos = pos_record.position
            print(f"  {symbol}: {pos.quantity:+.0f} @ ${pos.avg_cost:.2f}")
            print(f"    Market Value: ${pos.market_value:,.2f}, P&L: ${pos.unrealized_pnl:+,.2f}")
            
        # Portfolio summary
        summary = position_manager.get_portfolio_summary()
        print(f"\n💼 Portfolio Summary:")
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
        
        print(f"\n⚖️  Rebalancing Analysis:")
        print(f"  Needs Rebalancing: {needs_rebalancing}")
        for symbol, drift_pct in drift.items():
            print(f"  {symbol}: {drift_pct:.2%} drift")
            
        if needs_rebalancing:
            current_prices = {'AAPL': 155.0, 'GOOGL': 2450.0, 'MSFT': 295.0}
            orders = position_manager.generate_rebalancing_orders(target_weights, current_prices)
            
            print(f"\n📝 Rebalancing Orders:")
            for order in orders:
                print(f"  {order['side']} {order['quantity']:.0f} {order['symbol']} (${order['estimated_value']:,.2f})")
                print(f"    Current: {order['current_weight']:.2%} → Target: {order['target_weight']:.2%}")
                
        # Performance metrics
        performance = position_manager.get_performance_metrics()
        print(f"\n📈 Performance Metrics:")
        print(f"  Win Rate: {performance['win_rate']:.1%}")
        print(f"  Avg Winner: ${performance['avg_winner']:+,.2f}")
        print(f"  Avg Loser: ${performance['avg_loser']:+,.2f}")
        print(f"  Max Drawdown: {performance['max_drawdown']:.2%}")
        
        # Let it run for a bit
        print(f"\n👀 Monitoring for 10 seconds...")
        time.sleep(10)
        
    finally:
        position_manager.stop_monitoring()
        print("\n✅ Position Manager test completed")
