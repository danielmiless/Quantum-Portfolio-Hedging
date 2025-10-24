# complete_multi_asset_system.py
"""
Complete multi-asset class optimization pipeline
"""

from quantum.multi_asset_data import MultiAssetDataManager, AssetClassConfig
from quantum.multi_asset_qubo import MultiAssetQUBOBuilder
from quantum.quantum_hardware_interface import DWaveQUBOSolver
import numpy as np


def run_multi_asset_optimization():
    """Complete multi-asset class optimization pipeline."""
    
    print("🌐 Multi-Asset Class Quantum Portfolio Optimization")
    print("=" * 50)
    
    # 1. Setup multi-asset data
    print("\n1️⃣ Multi-Asset Data Preparation")
    manager = MultiAssetDataManager()
    
    # Define asset classes
    manager.add_asset_class(AssetClassConfig(
        name='Equities',
        tickers=['SPY', 'QQQ', 'IWM', 'EFA'],
        currency='USD'
    ))
    
    manager.add_asset_class(AssetClassConfig(
        name='Bonds',
        tickers=['TLT', 'IEF', 'LQD'],
        currency='USD'
    ))
    
    manager.add_asset_class(AssetClassConfig(
        name='Commodities',
        tickers=['GLD', 'USO'],
        currency='USD'
    ))
    
    manager.add_asset_class(AssetClassConfig(
        name='Crypto',
        tickers=['BTC-USD'],
        currency='USD'
    ))
    
    # Download data
    data = manager.download_all_data('2020-01-01', '2024-01-01')
    returns = manager.align_returns()
    stats = manager.calculate_cross_asset_statistics()
    
    print(f"✅ Loaded {returns.shape[1]} assets across {len(manager.asset_classes)} classes")
    
    # 2. Build multi-asset QUBO
    print("\n2️⃣ Multi-Asset QUBO Formulation")
    
    mu = stats['mean_returns'].values
    sigma = stats['covariance'].values
    
    # Asset class mapping
    asset_classes = {}
    for i, col in enumerate(returns.columns):
        for class_name in manager.asset_classes.keys():
            if col.startswith(f"{class_name}_"):
                asset_classes[i] = class_name
                break
    
    # Liquidity scores (simplified)
    liquidity = np.ones(len(mu))
    liquidity[-1] = 0.5  # Lower for crypto
    
    builder = MultiAssetQUBOBuilder(mu, sigma, asset_classes, liquidity)
    
    # Class constraints
    class_limits = {
        'Equities': (0.30, 0.60),
        'Bonds': (0.20, 0.40),
        'Commodities': (0.05, 0.20),
        'Crypto': (0.00, 0.10)
    }
    
    Q, offset = builder.build_qubo_with_class_constraints(class_limits)
    Q = builder.add_liquidity_penalty(Q)
    
    print(f"✅ QUBO matrix built: {Q.shape}")
    
    # 3. Quantum optimization
    print("\n3️⃣ Quantum Optimization")
    
    quantum_solver = DWaveQUBOSolver(use_hardware=False)
    result = quantum_solver.solve(Q, num_reads=1000)
    
    # Decode solution with proper key handling
    n_bits = 4
    weights = np.zeros(len(mu))
    max_val = 2**n_bits - 1
    
    # Handle different result formats
    solution = result.get('best_solution', result.get('solution', np.zeros(len(mu) * n_bits)))
    
    for i in range(len(mu)):
        for k in range(n_bits):
            idx = i * n_bits + k
            if idx < len(solution) and solution[idx] > 0.5:
                weights[i] += 2**k / max_val
    
    # Normalize weights
    if np.sum(weights) > 0:
        weights = weights / np.sum(weights)
    else:
        print("Warning: Zero weights, using equal allocation")
        weights = np.ones(len(mu)) / len(mu)
    
    # Calculate metrics
    portfolio_return = np.dot(weights, mu)
    portfolio_risk = np.sqrt(np.dot(weights, np.dot(sigma, weights)))
    sharpe = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0
    
    print(f"✅ Quantum optimization completed")
    print(f"   Expected Return: {portfolio_return:.2%}")
    print(f"   Portfolio Risk: {portfolio_risk:.2%}")
    print(f"   Sharpe Ratio: {sharpe:.3f}")
    
    # 4. Asset class allocation
    print("\n4️⃣ Asset Class Allocation")
    
    class_weights = manager.get_asset_class_weights(weights)
    
    for class_name, weight in class_weights.items():
        print(f"   {class_name}: {weight:.1%}")
    
    # 5. Individual allocations
    print("\n5️⃣ Individual Asset Allocations")
    
    for i, (asset, weight) in enumerate(zip(returns.columns, weights)):
        if weight > 0.01:
            print(f"   {asset}: {weight:.1%}")
    
    print("\n🎉 Multi-Asset Optimization Complete!")
    
    return {
        'weights': weights,
        'returns': portfolio_return,
        'risk': portfolio_risk,
        'sharpe': sharpe,
        'class_weights': class_weights
    }


if __name__ == "__main__":
    run_multi_asset_optimization()
