# complete_quantum_portfolio_system.py
"""
Complete quantum portfolio optimization system integrating all components
"""

from quantum.data_preparation import PortfolioDataPreparer
from quantum.quantum_hardware_interface import DWaveQUBOSolver, QuantumPortfolioOptimizer
from alt_data.ml_return_forecasting import MLReturnForecaster, ForecastingConfig, QuantumMLPortfolioOptimizer
from quantum.advanced_risk_models import CVaROptimizer, FactorRiskModel
from quantum.multi_objective_optimizer import MultiObjectiveQUBOOptimizer, OptimizationObjectives
from backtest.rebalancing_engine import PortfolioRebalancer, RebalancingConfig

def run_complete_quantum_portfolio_system():
    """
    Demonstration of the complete quantum portfolio optimization system.
    """
    print("🚀 Quantum Portfolio Optimization System")
    print("=" * 50)
    
    # 1. Data Preparation
    print("\n1️⃣ Data Preparation")
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    preparer = PortfolioDataPreparer(tickers, '2020-01-01', '2024-01-01')
    
    try:
        data = preparer.download_data()
        stats = preparer.calculate_statistics()
        print(f"✅ Loaded data for {len(tickers)} assets")
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return
    
    # 2. ML Return Forecasting
    print("\n2️⃣ ML Return Forecasting")
    config = ForecastingConfig(
        lookback_window=60,
        forecast_horizon=5,
        feature_engineering=True,
        ensemble_models=True
    )
    
    forecaster = MLReturnForecaster(config)
    X, y = forecaster.prepare_training_data(data['prices'])
    
    # Train-test split
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    forecaster.train_models(X_train, y_train)
    predictions = forecaster.predict_returns(X_test[-1:])  # Latest prediction
    
    print("✅ ML models trained and forecasts generated")
    
    # 3. Quantum Optimization
    print("\n3️⃣ Quantum Portfolio Optimization")
    quantum_solver = DWaveQUBOSolver(use_hardware=False)  # Use simulator
    quantum_optimizer = QuantumPortfolioOptimizer(quantum_solver)
    
    # Use ML predictions as expected returns
    mu_predicted = predictions['ensemble'][0]
    mu_historical, sigma = preparer.get_optimization_inputs('ledoit_wolf')
    
    # Quantum optimization with ML forecasts
    ml_quantum_optimizer = QuantumMLPortfolioOptimizer(forecaster, quantum_optimizer)
    result = ml_quantum_optimizer.optimize_with_ml_forecasts(
        X_test[-1:], stats['returns'].values, risk_aversion=1.0
    )
    
    print(f"✅ Quantum optimization completed")
    print(f"   Expected Return: {result['return']:.2%}")
    print(f"   Portfolio Risk: {result['risk']:.2%}")
    print(f"   Sharpe Ratio: {result['sharpe_ratio']:.3f}")
    
    # 4. Risk Analysis
    print("\n4️⃣ Advanced Risk Analysis")
    
    # CVaR analysis
    cvar_optimizer = CVaROptimizer(alpha=0.05)
    cvar_result = cvar_optimizer.optimize_cvar(stats['returns'].values)
    
    if cvar_result['success']:
        print(f"✅ CVaR-optimal portfolio (95% CVaR: {cvar_result['cvar']:.2%})")
    
    # Factor model
    factor_model = FactorRiskModel(n_factors=3)
    factor_results = factor_model.fit_factor_model(stats['returns'])
    risk_attribution = factor_model.calculate_risk_attribution(
        result['weights'], factor_results
    )
    
    print(f"✅ Factor model fitted")
    print(f"   Systematic Risk: {risk_attribution['systematic_pct']*100:.1f}%")
    print(f"   Specific Risk: {risk_attribution['specific_pct']*100:.1f}%")
    
    # 5. Portfolio Rebalancing Simulation
    print("\n5️⃣ Dynamic Rebalancing")
    rebalance_config = RebalancingConfig(
        method='threshold',
        threshold=0.05,
        transaction_cost=0.001
    )
    
    rebalancer = PortfolioRebalancer(rebalance_config)
    performance = rebalancer.simulate_rebalancing(
        stats['returns'], result['weights'], initial_value=100000
    )
    
    initial_value = performance['portfolio_value'].iloc[0]
    final_value = performance['portfolio_value'].iloc[-1]
    total_return = (final_value / initial_value - 1) * 100
    n_rebalances = performance['rebalanced'].sum()
    
    print(f"✅ Rebalancing simulation completed")
    print(f"   Total Return: {total_return:.2f}%")
    print(f"   Number of Rebalances: {n_rebalances}")
    
    # 6. Summary Report
    print("\n6️⃣ Final Portfolio Summary")
    print("=" * 30)
    print("Optimal Portfolio Weights:")
    for i, (ticker, weight) in enumerate(zip(tickers, result['weights'])):
        print(f"  {ticker}: {weight:.1%}")
    
    print(f"\nKey Metrics:")
    print(f"  Expected Return: {result['return']:.2%}")
    print(f"  Portfolio Risk: {result['risk']:.2%}")
    print(f"  Sharpe Ratio: {result['sharpe_ratio']:.3f}")
    print(f"  Solver Used: {result['solver_info']['solver']}")
    
    print("\n🎉 Quantum Portfolio Optimization Complete!")


if __name__ == "__main__":
    run_complete_quantum_portfolio_system()
