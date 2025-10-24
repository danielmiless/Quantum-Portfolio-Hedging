# src/integration/transaction_cost_analyzer.py
"""
Transaction cost analysis with fixed imports
"""

import numpy as np
import pandas as pd
import sys
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path

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
    from integration.position_manager import Trade
except ImportError:
    from position_manager import Trade


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
                    spread_cost = (ask - mid_price) * abs(trade.quantity)
                else:
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
        
        # Market impact parameters
        alpha = 0.314
        beta = 0.6
        
        impact_pct = alpha * (participation_rate ** beta) / 100
        impact_cost = trade.price * impact_pct * abs(trade.quantity)
        
        return impact_cost
        
    def _calculate_slippage(self, trade: Trade, benchmark_price: float) -> float:
        """Calculate slippage vs benchmark."""
        price_diff = trade.price - benchmark_price
        
        if trade.side == 'BUY':
            slippage = price_diff * abs(trade.quantity)
        else:
            slippage = -price_diff * abs(trade.quantity)
            
        return slippage
        
    def _get_typical_spread(self, symbol: str) -> float:
        """Get typical bid-ask spread percentage."""
        if symbol in self.typical_spreads:
            return self.typical_spreads[symbol]
            
        # Default spread estimates
        if symbol in ['AAPL', 'MSFT', 'GOOGL', 'AMZN']:
            return 0.0001  # 1 basis point
        else:
            return 0.0005  # 5 basis points
            
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
        
        return volume_estimates.get(symbol, 1000000)
        
    def _calculate_execution_quality_score(self, cost_breakdown: List[TransactionCost],
                                          trade_value: float,
                                          market_data: Optional[Dict]) -> float:
        """Calculate execution quality score."""
        total_cost_bps = sum(cost.basis_points for cost in cost_breakdown)
        
        # Benchmarks (basis points)
        excellent_threshold = 5
        good_threshold = 15
        poor_threshold = 50
        
        if total_cost_bps < excellent_threshold:
            return 90 + (excellent_threshold - total_cost_bps)
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
            
        return "\n".join(report)


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
        'AAPL': 155.20,
        'GOOGL': 2450.00,
        'MSFT': 296.00,
        'TSLA': 238.50
    }
    
    # Analyze trades
    analyzer = TransactionCostAnalyzer()
    
    print("\n📊 Analyzing individual trades...")
    for trade in trades:
        benchmark = benchmark_prices.get(trade.symbol)
        result = analyzer.analyze_trade(trade, benchmark)
        
        print(f"\n{trade.symbol}: {trade.side} {trade.quantity} @ ${trade.price:.2f}")
        print(f"  Benchmark: ${result.benchmark_price:.2f}")
        print(f"  Total Cost: ${result.total_cost:.2f} ({result.total_cost_bps:.1f} bps)")
        print(f"  Implementation Shortfall: ${result.implementation_shortfall:+.2f}")
        print(f"  Execution Quality: {result.execution_quality_score:.1f}/100")
        
        print("  Cost Breakdown:")
        for cost in result.cost_breakdown:
            print(f"    {cost.cost_type.value}: ${cost.amount:.2f} ({cost.basis_points:.1f} bps)")
            
    # Portfolio analysis
    print("\n" + "="*50)
    portfolio_analysis = analyzer.analyze_portfolio_trades(trades, benchmark_prices)
    
    summary = portfolio_analysis['summary']
    print(f"\n💼 PORTFOLIO TCA SUMMARY:")
    print(f"Total Value: ${summary['total_value']:,.2f}")
    print(f"Total Costs: ${summary['total_costs']:.2f}")
    print(f"Cost Rate: {summary['total_cost_bps']:.1f} basis points")
    print(f"Avg Quality Score: {summary['avg_execution_quality']:.1f}/100")
    
    print(f"\nCost Breakdown:")
    for cost_type, amount in summary['cost_by_type'].items():
        pct = (amount / summary['total_costs'] * 100) if summary['total_costs'] > 0 else 0
        print(f"  {cost_type}: ${amount:.2f} ({pct:.1f}%)")
        
    # Generate full report
    print("\n" + "="*50)
    print("FULL TCA REPORT")
    print("="*50)
    full_report = analyzer.generate_tca_report(portfolio_analysis)
    print(full_report)
    
    print("\n✅ Transaction Cost Analysis completed")
