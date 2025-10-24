# Place in: src/integration/complete_esg_system.py

import numpy as np
import pandas as pd
from quantum.esg_portfolio_optimizer import ESGPortfolioOptimizer
from integration.esg_reporting import ESGReportingEngine
from quantum.data_preparation import PortfolioDataPreparer


def run_complete_esg_optimization():
    """Complete ESG-aware quantum portfolio optimization pipeline."""
    
    print("🌱 ESG & Sustainable Investing Quantum Portfolio Optimization")
    print("=" * 60)
    
    # 1. Setup and Data Preparation
    print("\n1️⃣ ESG Data Preparation")
    
    # Extended universe with ESG diversity
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'JNJ', 'PG', 'TSLA', 'NEE', 'XOM', 'CVX', 'WMT']
    
    # Load financial data
    preparer = PortfolioDataPreparer(tickers, '2020-01-01', '2024-01-01')
    try:
        financial_data = preparer.download_data()
        financial_stats = preparer.calculate_statistics()
        mu, sigma = preparer.get_optimization_inputs('ledoit_wolf')
        print(f"✅ Financial data loaded for {len(tickers)} assets")
    except Exception as e:
        print(f"❌ Error loading financial data: {e}")
        # Use synthetic data
        np.random.seed(42)
        mu = np.random.normal(0.08, 0.04, len(tickers))
        A = np.random.randn(len(tickers), len(tickers))
        sigma = 0.01 * (A @ A.T + np.eye(len(tickers)))
        print("✅ Using synthetic financial data")
    
    # Initialize ESG optimizer
    esg_optimizer = ESGPortfolioOptimizer(tickers)
    esg_info = esg_optimizer.load_esg_data(mu, sigma)
    
    print(f"✅ ESG data loaded for {len(tickers)} assets")
    print("\nESG Overview:")
    print(esg_info['esg_data'][['esg_score', 'carbon_intensity']].round(1))
    
    # 2. ESG Strategy Comparison
    print("\n2️⃣ ESG Strategy Comparison")
    
    esg_strategies = [
        {
            'name': 'Traditional_Portfolio',
            'esg_weight': 0.0,
            'carbon_penalty': 0.0,
            'use_quantum': False
        },
        {
            'name': 'ESG_Integration',
            'esg_weight': 0.3,
            'carbon_penalty': 0.2,
            'use_quantum': True
        },
        {
            'name': 'ESG_Best_in_Class',
            'esg_weight': 0.5,
            'carbon_penalty': 0.4,
            'min_esg_score': 70,
            'use_quantum': True
        },
        {
            'name': 'Climate_Focus',
            'esg_weight': 0.2,
            'carbon_penalty': 0.8,
            'max_carbon_intensity': 100,
            'use_quantum': True
        },
        {
            'name': 'Sustainable_Leader',
            'esg_weight': 0.6,
            'carbon_penalty': 0.5,
            'min_esg_score': 75,
            'max_carbon_intensity': 50,
            'use_quantum': True
        }
    ]
    
    strategy_comparison = esg_optimizer.compare_esg_strategies(esg_strategies)
    
    print("\nESG Strategy Performance:")
    print(strategy_comparison.round(3))
    
    # 3. Optimal ESG Portfolio Selection
    print("\n3️⃣ Optimal ESG Portfolio Analysis")
    
    # Select best balanced strategy
    optimal_esg_portfolio = esg_optimizer.optimize_esg_portfolio(
        esg_weight=0.5,
        carbon_penalty=0.4,
        min_esg_score=70,
        max_carbon_intensity=100,
        use_quantum=True
    )
    
    print(f"Optimal ESG Portfolio Results:")
    print(f"  Expected Return: {optimal_esg_portfolio['expected_return']:.2%}")
    print(f"  Portfolio Risk: {optimal_esg_portfolio['portfolio_risk']:.2%}")
    print(f"  Sharpe Ratio: {optimal_esg_portfolio['sharpe_ratio']:.3f}")
    print(f"  ESG Score: {optimal_esg_portfolio['esg_score']:.1f}/100")
    print(f"  Carbon Intensity: {optimal_esg_portfolio['carbon_intensity']:.1f} tCO2e/$M")
    print(f"  ESG Efficiency: {optimal_esg_portfolio['esg_efficiency']:.2f}")
    
    # 4. Portfolio Allocation Analysis
    print("\n4️⃣ ESG Portfolio Allocation")
    
    weights = optimal_esg_portfolio['weights']
    esg_scores = esg_info['esg_scores']
    carbon_intensities = esg_info['carbon_intensities']
    
    print("Individual Holdings (>1% allocation):")
    for i, ticker in enumerate(tickers):
        if weights[i] > 0.01:
            print(f"  {ticker:5}: {weights[i]:6.1%} | ESG: {esg_scores[i]:4.0f} | Carbon: {carbon_intensities[i]:6.1f}")
    
    # ESG allocation breakdown
    high_esg = np.sum(weights[esg_scores > 80])
    medium_esg = np.sum(weights[(esg_scores >= 60) & (esg_scores <= 80)])
    low_esg = np.sum(weights[esg_scores < 60])
    
    print(f"\nESG Allocation Breakdown:")
    print(f"  High ESG (>80): {high_esg:.1%}")
    print(f"  Medium ESG (60-80): {medium_esg:.1%}")
    print(f"  Low ESG (<60): {low_esg:.1%}")
    
    # Carbon allocation breakdown
    low_carbon = np.sum(weights[carbon_intensities < 50])
    medium_carbon = np.sum(weights[(carbon_intensities >= 50) & (carbon_intensities <= 200)])
    high_carbon = np.sum(weights[carbon_intensities > 200])
    
    print(f"\nCarbon Allocation Breakdown:")
    print(f"  Low Carbon (<50): {low_carbon:.1%}")
    print(f"  Medium Carbon (50-200): {medium_carbon:.1%}")
    print(f"  High Carbon (>200): {high_carbon:.1%}")
    
    # 5. ESG Compliance and Reporting
    print("\n5️⃣ ESG Compliance and Reporting")
    
    reporter = ESGReportingEngine()
    esg_report = reporter.generate_esg_report(
        optimal_esg_portfolio, 
        esg_info['esg_data'], 
        'comprehensive'
    )
    
    print("\nRegulatory Compliance Status:")
    for framework, compliance in esg_report['compliance_check'].items():
        status = compliance.get('recommendation', compliance.get('alignment_level', 'Compliant'))
        print(f"  {framework:15}: {status}")
    
    print("\nSDG Alignment:")
    for sdg, score in esg_report['sdg_alignment']['sdg_alignment_scores'].items():
        print(f"  {sdg:20}: {score}")
    
    # 6. Risk Analysis
    print("\n6️⃣ ESG Risk Analysis")
    
    risk_analysis = esg_report['risk_analysis']
    print(f"ESG Risk Level: {risk_analysis['esg_risk_level']}")
    print(f"ESG Risk Concentration: {risk_analysis['esg_risk_concentration']}")
    print(f"Portfolio Diversification: {risk_analysis['diversification_score']}")
    
    # 7. Performance Summary
    print("\n7️⃣ ESG vs Traditional Comparison")
    
    traditional = strategy_comparison[strategy_comparison['strategy'] == 'Traditional_Portfolio'].iloc[0]
    sustainable = strategy_comparison[strategy_comparison['strategy'] == 'Sustainable_Leader'].iloc[0]
    
    print("Metric Comparison (Traditional vs Sustainable):")
    print(f"  Return:      {traditional['expected_return']:.2%} vs {sustainable['expected_return']:.2%}")
    print(f"  Risk:        {traditional['portfolio_risk']:.2%} vs {sustainable['portfolio_risk']:.2%}")
    print(f"  Sharpe:      {traditional['sharpe_ratio']:.3f} vs {sustainable['sharpe_ratio']:.3f}")
    print(f"  ESG Score:   {traditional['esg_score']:.1f} vs {sustainable['esg_score']:.1f}")
    print(f"  Carbon:      {traditional['carbon_intensity']:.1f} vs {sustainable['carbon_intensity']:.1f}")
    
    # Calculate ESG premium/discount
    return_diff = sustainable['expected_return'] - traditional['expected_return']
    esg_premium = f"{return_diff:.1%}" if return_diff >= 0 else f"{abs(return_diff):.1%} discount"
    
    print(f"\nESG Investment Impact:")
    print(f"  Return Impact: {esg_premium}")
    print(f"  ESG Improvement: +{sustainable['esg_score'] - traditional['esg_score']:.1f} points")
    print(f"  Carbon Reduction: -{traditional['carbon_intensity'] - sustainable['carbon_intensity']:.1f} tCO2e/$M")
    
    # 8. Export Report
    # print("\n8️⃣ Report Generation")
    
    # report_filename = reporter.export_report(esg_report, 'json')
    # csv_filename = reporter.export_report(esg_report, 'csv')
    
    # print(f"✅ ESG reports generated:")
    # print(f"   JSON: {report_filename}")
    # print(f"   CSV:  {csv_filename}")
    
    print("\n🎉 ESG Quantum Portfolio Optimization Complete!")
    
    return {
        'optimal_portfolio': optimal_esg_portfolio,
        'strategy_comparison': strategy_comparison,
        'esg_report': esg_report,
        'tickers': tickers,
        'weights': weights
    }


if __name__ == "__main__":
    results = run_complete_esg_optimization()
