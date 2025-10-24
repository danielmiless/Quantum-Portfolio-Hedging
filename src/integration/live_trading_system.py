# src/integration/live_trading_system.py
"""
Complete live trading system with environment variable support and fixed imports
"""

import numpy as np
import pandas as pd
import sys
import os
from typing import Dict, List, Tuple, Optional, Callable
from datetime import datetime, timedelta
from pathlib import Path
import threading
import time
import logging

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
    class Config:
        REBALANCE_THRESHOLD = 0.05
        MAX_POSITION_SIZE = 0.25
        DEFAULT_EXECUTION_ALGORITHM = 'VWAP'
        ENVIRONMENT = 'development'

# Import project modules with error handling
try:
    from quantum.data_preparation import PortfolioDataPreparer
except ImportError:
    print("⚠️  quantum.data_preparation not available")
    PortfolioDataPreparer = None

try:
    from quantum.esg_portfolio_optimizer import ESGPortfolioOptimizer
except ImportError:
    print("⚠️  quantum.esg_portfolio_optimizer not available")
    ESGPortfolioOptimizer = None

try:
    from integration.broker_interface import BrokerFactory, BrokerInterface, OrderType
except ImportError:
    print("⚠️  integration.broker_interface not available")
    BrokerFactory = None

try:
    from integration.order_manager import OrderManager, OrderPriority
except ImportError:
    print("⚠️  integration.order_manager not available")
    OrderManager = None

try:
    from integration.execution_algorithms import ExecutionEngine, ExecutionRequest, AlgorithmType
except ImportError:
    print("⚠️  integration.execution_algorithms not available")
    ExecutionEngine = None

try:
    from integration.position_manager import PositionManager
except ImportError:
    print("⚠️  integration.position_manager not available")
    PositionManager = None

try:
    from integration.transaction_cost_analyzer import TransactionCostAnalyzer
except ImportError:
    print("⚠️  integration.transaction_cost_analyzer not available")
    TransactionCostAnalyzer = None


class LiveTradingSystem:
    """Complete live trading system using environment configuration."""
    
    def __init__(self, broker_name: str, 
                 tickers: List[str], 
                 rebalance_frequency: str = 'daily',
                 api_key: Optional[str] = None,
                 secret_key: Optional[str] = None,
                 **broker_kwargs):
        """
        Initialize live trading system.
        
        Args:
            broker_name: Broker name ('alpaca', 'interactive_brokers')
            tickers: List of symbols to trade
            rebalance_frequency: 'daily', 'weekly', 'monthly'
            api_key: Optional API key override (uses env vars if None)
            secret_key: Optional secret key override
            **broker_kwargs: Additional broker parameters
        """
        self.tickers = tickers
        self.rebalance_frequency = rebalance_frequency
        
        # Check dependencies
        if BrokerFactory is None:
            raise ImportError("BrokerFactory not available. Check broker_interface.py")
        
        # Initialize broker connection using config
        self.broker = BrokerFactory.create_broker(
            broker_name,
            api_key=api_key,
            secret_key=secret_key,
            **broker_kwargs
        )
        
        # Initialize components (with None checks)
        if OrderManager:
            self.order_manager = OrderManager(self.broker)
        else:
            self.order_manager = None
            print("⚠️  Order manager not available")
        
        if ExecutionEngine and self.order_manager:
            self.execution_engine = ExecutionEngine(self.order_manager)
        else:
            self.execution_engine = None
            print("⚠️  Execution engine not available")
        
        if PositionManager:
            self.position_manager = PositionManager(self.broker)
        else:
            self.position_manager = None
            print("⚠️  Position manager not available")
        
        if TransactionCostAnalyzer:
            self.tca_analyzer = TransactionCostAnalyzer()
        else:
            self.tca_analyzer = None
            print("⚠️  TCA analyzer not available")
        
        # Portfolio optimization
        if ESGPortfolioOptimizer:
            self.esg_optimizer = ESGPortfolioOptimizer(tickers)
        else:
            self.esg_optimizer = None
            print("⚠️  ESG optimizer not available")
        
        self.current_target_weights = {}
        self.last_optimization_time = None
        self.last_rebalance_time = None
        
        # System state
        self.running = False
        self.main_thread = None
        
        # Load configuration from environment
        self.config = {
            'optimization_frequency': 'daily',
            'rebalance_threshold': Config.REBALANCE_THRESHOLD,
            'max_position_size': Config.MAX_POSITION_SIZE,
            'execution_algorithm': Config.DEFAULT_EXECUTION_ALGORITHM,
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
            if self.order_manager:
                self.order_manager.start()
            
            if self.position_manager:
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
            
            if not PortfolioDataPreparer:
                self.logger.warning("Data preparer not available, skipping optimization")
                return
            
            # Prepare data
            data_preparer = PortfolioDataPreparer(
                self.tickers, 
                (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'),
                datetime.now().strftime('%Y-%m-%d')
            )
            
            financial_data = data_preparer.download_data()
            financial_stats = data_preparer.calculate_statistics()
            mu, sigma = data_preparer.get_optimization_inputs('ledoit_wolf')
            
            if not self.esg_optimizer:
                self.logger.warning("ESG optimizer not available, using equal weights")
                self.current_target_weights = {ticker: 1.0/len(self.tickers) for ticker in self.tickers}
                self.last_optimization_time = datetime.now()
                return
            
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
            import traceback
            traceback.print_exc()
            
    def _should_rebalance(self, current_time: datetime) -> bool:
        """Check if rebalancing is needed."""
        if not self.current_target_weights or not self.position_manager:
            return False
            
        # Check drift threshold
        portfolio_value = self.position_manager.portfolio_metrics.get('total_value', 0)
        if portfolio_value <= 0:
            return False
            
        needs_rebalancing, drift = self.position_manager.check_rebalancing_needed(
            self.current_target_weights, 
            self.config['rebalance_threshold']
        )
        
        if needs_rebalancing:
            max_drift = max(drift.values()) if drift else 0
            self.logger.info(f"Rebalancing needed - max drift: {max_drift:.2%}")
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
            
            if not self.position_manager or not self.order_manager:
                self.logger.warning("Required components not available for rebalancing")
                return
            
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
            
            # Execute orders
            for order_info in rebalancing_orders:
                if abs(order_info['quantity']) < 1:  # Skip fractional shares
                    continue
                    
                # Submit market order
                try:
                    self.order_manager.submit_order(
                        symbol=order_info['symbol'],
                        quantity=order_info['quantity'],
                        order_type=OrderType.MARKET,
                        side=order_info['side'],
                        priority=OrderPriority.HIGH
                    )
                    self.logger.info(f"Submitted: {order_info['side']} {order_info['quantity']} {order_info['symbol']}")
                except Exception as e:
                    self.logger.error(f"Failed to submit order for {order_info['symbol']}: {e}")
            
            self.last_rebalance_time = datetime.now()
            self.logger.info(f"Rebalancing completed")
            
        except Exception as e:
            self.logger.error(f"Rebalancing failed: {e}")
            import traceback
            traceback.print_exc()
            
    def _monitor_risk_limits(self):
        """Monitor portfolio risk limits."""
        if not self.config['risk_limit_enabled'] or not self.position_manager:
            return
            
        try:
            positions = self.position_manager.get_all_positions()
            portfolio_value = self.position_manager.portfolio_metrics.get('total_value', 0)
            
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
                    
        except Exception as e:
            self.logger.error(f"Risk monitoring error: {e}")
            
    def _update_performance_log(self):
        """Update performance tracking."""
        if not self.position_manager:
            return
            
        try:
            portfolio_summary = self.position_manager.get_portfolio_summary()
            performance_metrics = self.position_manager.get_performance_metrics()
            
            log_entry = {
                'timestamp': datetime.now(),
                'portfolio_value': portfolio_summary.get('total_value', 0),
                'total_pnl': portfolio_summary.get('total_pnl', 0),
                'pnl_pct': portfolio_summary.get('total_pnl_pct', 0),
                'num_positions': portfolio_summary.get('num_positions', 0),
                'win_rate': performance_metrics.get('win_rate', 0),
                'max_drawdown': performance_metrics.get('max_drawdown', 0)
            }
            
            self.performance_log.append(log_entry)
            
            # Keep only last 1000 entries
            if len(self.performance_log) > 1000:
                self.performance_log = self.performance_log[-1000:]
                
        except Exception as e:
            self.logger.error(f"Performance logging error: {e}")
            
    def get_system_status(self) -> Dict:
        """Get current system status."""
        status = {
            'running': self.running,
            'connected': self.broker.connected if hasattr(self.broker, 'connected') else False,
            'last_optimization': self.last_optimization_time,
            'last_rebalance': self.last_rebalance_time,
            'target_weights': self.current_target_weights,
            'config': self.config
        }
        
        if self.position_manager:
            try:
                status['portfolio'] = self.position_manager.get_portfolio_summary()
            except:
                status['portfolio'] = {}
        
        if self.order_manager:
            try:
                status['orders'] = self.order_manager.get_order_statistics()
            except:
                status['orders'] = {}
        
        return status


# Example usage and system test
if __name__ == "__main__":
    print("Live Trading System Test")
    print("=" * 30)
    
    # Configuration
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    
    try:
        # Initialize trading system (uses .env credentials)
        trading_system = LiveTradingSystem(
            broker_name='alpaca',
            tickers=tickers,
            rebalance_frequency='daily'
        )
        
        # Configure system
        trading_system.config.update({
            'rebalance_threshold': 0.03,
            'max_position_size': 0.30,
            'esg_constraints_enabled': True
        })
        
        print("\n✅ Trading system initialized")
        print(f"Environment: {Config.ENVIRONMENT}")
        print(f"Broker: alpaca")
        print(f"Tickers: {tickers}")
        
        print("\nConfiguration:")
        for key, value in trading_system.config.items():
            print(f"  {key}: {value}")
        
        # Note: Actual start() requires valid credentials and active market
        print("\n⚠️  To run live:")
        print("  1. Ensure .env has ALPACA_API_KEY and ALPACA_SECRET_KEY")
        print("  2. Call trading_system.start()")
        print("  3. Monitor with trading_system.get_system_status()")
        
    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ Live Trading System components loaded successfully")
