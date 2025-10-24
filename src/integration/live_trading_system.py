# src/integration/live_trading_system.py
"""
Live Trading System
Main orchestration for quantum portfolio optimization and execution
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import time

# Setup paths
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
src_dir = current_dir.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Load environment variables
from dotenv import load_dotenv
env_path = project_root / '.env'
load_dotenv(env_path)

from integration.broker_interface import BrokerFactory, OrderType


class LiveTradingSystem:
    """Main live trading system with quantum optimization."""
    
    def __init__(self, broker_name: str = 'alpaca', tickers: List[str] = None, 
                 rebalance_frequency: str = 'daily'):
        """Initialize the live trading system."""
        
        self.logger = logging.getLogger('LiveTradingSystem')
        
        # Configuration
        self.broker_name = broker_name
        self.tickers = tickers or ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        self.rebalance_frequency = rebalance_frequency
        
        # Trading state
        self.broker = None
        self.connected = False
        self.running = False
        self.last_optimization_time = None
        self.last_rebalance_time = None
        self.current_portfolio = None
        
        # Configuration
        self.config = {
            'rebalance_threshold': 0.15,
            'min_rebalance_interval': 3600,
            'max_position_size': 0.20,
            'min_position_size': 0.01,
            'execution_algorithm': 'VWAP',
            'risk_limit_enabled': True,
            'esg_constraints_enabled': True,
            'cash_buffer': 0.05,
            'optimization_frequency': 'daily'
        }
        
        # Initialize components
        self._init_broker()
        self._init_data_preparer()
        self._init_optimizer()
        self._init_order_manager()
        
        self.logger.info("LiveTradingSystem initialized")
    
    def _init_broker(self):
        """Initialize broker connection."""
        try:
            self.broker = BrokerFactory.create_broker(self.broker_name)
            if self.broker and self.broker.connect():
                self.connected = True
                self.logger.info(f"✅ Connected to {self.broker_name}")
            else:
                self.logger.error(f"❌ Failed to connect to {self.broker_name}")
        except Exception as e:
            self.logger.error(f"Error initializing broker: {e}")
    
    def _init_data_preparer(self):
        """Initialize data preparation module."""
        try:
            from quantum.data_preparation import PortfolioDataPreparer
            self.data_preparer = PortfolioDataPreparer(tickers=self.tickers)
            self.logger.info("✅ Data preparer initialized")
        except Exception as e:
            self.logger.warning(f"Data preparer not available: {e}")
            self.data_preparer = None
    
    def _init_optimizer(self):
        """Initialize quantum optimizer."""
        try:
            from quantum.quantum_optimizer import QuantumPortfolioOptimizer
            self.quantum_optimizer = QuantumPortfolioOptimizer(tickers=self.tickers)
            self.logger.info("✅ Quantum optimizer initialized")
        except Exception as e:
            self.logger.warning(f"Quantum optimizer not available: {e}")
            self.quantum_optimizer = None
    
    def _init_order_manager(self):
        """Initialize order management."""
        try:
            from integration.order_manager import OrderManager
            self.order_manager = OrderManager(broker=self.broker)
            self.logger.info("✅ Order manager initialized")
        except Exception as e:
            self.logger.warning(f"⚠️  Order manager not available")
            self.order_manager = None
    
    def start(self) -> bool:
        """Start the live trading system."""
        if not self.connected:
            self.logger.error("Not connected to broker")
            return False
        
        if not self.data_preparer:
            self.logger.error("Data preparer not available")
            return False
        
        if not self.quantum_optimizer:
            self.logger.error("Quantum optimizer not available")
            return False
        
        self.running = True
        self.logger.info("Live trading system started successfully")
        return True
    
    def stop(self):
        """Stop the live trading system."""
        self.running = False
        if self.broker:
            self.broker.disconnect()
        self.logger.info("Live trading system stopped")
    
    def run_portfolio_optimization(self) -> Optional[Dict]:
        """Run quantum portfolio optimization."""
        try:
            if not self.data_preparer or not self.quantum_optimizer:
                self.logger.warning("Required components not available for optimization")
                return None
            
            # Download data
            self.logger.info("Running portfolio optimization...")
            financial_data = self.data_preparer.download_data()
            
            if financial_data is None or financial_data.empty:
                self.logger.error("Failed to download financial data")
                return None
            
            # Calculate statistics
            stats = self.data_preparer.calculate_statistics()
            
            # Run quantum optimization
            optimal_weights = self.quantum_optimizer.optimize(
                expected_returns=stats['mean_returns'],
                cov_matrix=stats['cov_matrix']
            )
            
            self.last_optimization_time = datetime.now()
            self.logger.info(f"Portfolio optimization completed")
            
            return {
                'weights': optimal_weights,
                'stats': stats,
                'timestamp': self.last_optimization_time
            }
            
        except Exception as e:
            self.logger.error(f"Portfolio optimization failed: {e}")
            return None
    
    def rebalance_portfolio(self, target_weights: Dict[str, float]) -> bool:
        """Rebalance portfolio to target weights."""
        try:
            if not self.broker or not self.connected:
                self.logger.error("Broker not connected")
                return False
            
            if not self.order_manager:
                self.logger.warning("Order manager not available for rebalancing")
                return False
            
            # Get current positions and cash
            positions = self.broker.get_positions()
            account = self.broker.get_account()
            
            current_portfolio_value = account.get('portfolio_value', 0)
            cash = account.get('cash', 0)
            buying_power = account.get('buying_power', 0)
            
            self.logger.info(f"Starting portfolio rebalancing...")
            self.logger.info(f"Current portfolio value: ${current_portfolio_value:,.2f}")
            self.logger.info(f"Cash available: ${cash:,.2f}")
            self.logger.info(f"Target weights: {target_weights}")
            
            # Generate rebalancing orders
            orders = self.order_manager.generate_rebalancing_orders(
                target_weights=target_weights,
                current_positions=positions,
                current_portfolio_value=current_portfolio_value,
                cash=cash,
                buying_power=buying_power
            )
            
            if not orders:
                self.logger.warning("No rebalancing orders generated")
                return False
            
            self.logger.info(f"Generated {len(orders)} rebalancing orders")
            
            # Submit orders
            submitted_count = 0
            for order in orders:
                try:
                    order_type = OrderType.BUY if order['side'] == 'buy' else OrderType.SELL
                    result = self.broker.submit_order(
                        ticker=order['ticker'],
                        quantity=order['quantity'],
                        order_type=order_type
                    )
                    
                    if result:
                        self.logger.info(f"✅ Order submitted: {order['side'].upper()} {order['quantity']} {order['ticker']}")
                        submitted_count += 1
                    else:
                        self.logger.error(f"❌ Failed to submit order: {order['side']} {order['ticker']}")
                
                except Exception as e:
                    self.logger.error(f"Error submitting order: {e}")
            
            self.last_rebalance_time = datetime.now()
            self.logger.info(f"Rebalancing completed: {submitted_count} orders submitted")
            
            return submitted_count > 0
            
        except Exception as e:
            self.logger.error(f"Portfolio rebalancing failed: {e}")
            return False
    
    def get_system_status(self) -> Dict:
        """Get current system status."""
        try:
            account = self.broker.get_account() if self.broker else {}
            positions = self.broker.get_positions() if self.broker else []
            orders = self.broker.get_orders() if self.broker else []
            
            return {
                'connected': self.connected,
                'running': self.running,
                'portfolio': {
                    'total_value': account.get('portfolio_value', 0),
                    'cash': account.get('cash', 0),
                    'buying_power': account.get('buying_power', 0),
                    'total_pnl': account.get('equity', 0) - 100000,  # Assuming $100K starting
                    'total_pnl_pct': (account.get('equity', 0) - 100000) / 100000,
                    'num_positions': len(positions)
                },
                'orders': {
                    'active_orders': len([o for o in orders if o['status'] in ['pending_new', 'accepted', 'partially_filled']]),
                    'orders_submitted': len([o for o in orders if o['status'] != 'cancelled']),
                    'orders_filled': len([o for o in orders if o['status'] == 'filled']),
                    'fill_rate': len([o for o in orders if o['status'] == 'filled']) / max(len(orders), 1)
                },
                'last_optimization': self.last_optimization_time,
                'last_rebalance': self.last_rebalance_time
            }
        except Exception as e:
            self.logger.error(f"Error getting system status: {e}")
            return {}


def main():
    """Main entry point for live trading."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize system
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'JPM', 'JNJ', 'UNH']
    system = LiveTradingSystem(
        broker_name='alpaca',
        tickers=tickers,
        rebalance_frequency='daily'
    )
    
    # Start
    if system.start():
        print("✅ System started")
        
        # Run optimization
        opt_result = system.run_portfolio_optimization()
        
        if opt_result:
            # Rebalance
            system.rebalance_portfolio(opt_result['weights'])
        
        # Get status
        status = system.get_system_status()
        print(f"Status: {status}")
        
        system.stop()
    else:
        print("❌ Failed to start system")


if __name__ == "__main__":
    main()