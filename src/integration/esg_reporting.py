# Place in: src/integration/esg_reporting.py

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


class ESGReportingEngine:
    """
    ESG reporting and compliance monitoring.
    """
    
    def __init__(self):
        self.compliance_frameworks = {
            'EU_SFDR': self._check_sfdr_compliance,
            'UN_SDG': self._check_sdg_alignment,
            'TCFD': self._check_tcfd_compliance,
            'US_SEC': self._check_sec_climate_disclosure
        }
    
    def generate_esg_report(self, portfolio_results: Dict,
                           esg_data: pd.DataFrame,
                           report_type: str = 'comprehensive') -> Dict:
        """
        Generate comprehensive ESG report.
        
        Args:
            portfolio_results: Results from ESG optimization
            esg_data: ESG data DataFrame
            report_type: Type of report ('summary', 'comprehensive', 'regulatory')
            
        Returns:
            ESG report dictionary
        """
        weights = portfolio_results['weights']
        
        report = {
            'report_date': datetime.now().strftime('%Y-%m-%d'),
            'report_type': report_type,
            'portfolio_summary': self._create_portfolio_summary(portfolio_results),
            'esg_breakdown': self._create_esg_breakdown(weights, esg_data),
            'carbon_analysis': self._create_carbon_analysis(weights, esg_data),
            'sdg_alignment': self._create_sdg_analysis(weights, esg_data),
            'compliance_check': self._check_regulatory_compliance(portfolio_results, esg_data)
        }
        
        if report_type == 'comprehensive':
            report.update({
                'peer_comparison': self._create_peer_comparison(portfolio_results),
                'esg_trends': self._analyze_esg_trends(esg_data),
                'risk_analysis': self._create_esg_risk_analysis(weights, esg_data)
            })
        
        return report
    
    def _create_portfolio_summary(self, results: Dict) -> Dict:
        """Create high-level portfolio summary."""
        return {
            'total_assets': len(results['weights']),
            'expected_return': f"{results['expected_return']:.2%}",
            'portfolio_risk': f"{results['portfolio_risk']:.2%}",
            'sharpe_ratio': f"{results['sharpe_ratio']:.3f}",
            'esg_score': f"{results['esg_score']:.1f}/100",
            'carbon_intensity': f"{results['carbon_intensity']:.1f} tCO2e/$M",
            'optimization_method': results.get('optimization_method', 'Unknown')
        }
    
    def _create_esg_breakdown(self, weights: np.ndarray, esg_data: pd.DataFrame) -> Dict:
        """Create detailed ESG breakdown."""
        esg_scores = esg_data['esg_score'].values
        
        # ESG distribution
        high_esg_weight = np.sum(weights[esg_scores > 80])
        medium_esg_weight = np.sum(weights[(esg_scores >= 60) & (esg_scores <= 80)])
        low_esg_weight = np.sum(weights[esg_scores < 60])
        
        # Top ESG holdings
        top_esg_indices = np.argsort(esg_scores)[-5:][::-1]  # Top 5
        top_holdings = []
        for idx in top_esg_indices:
            if weights[idx] > 0.01:  # Only if weight > 1%
                top_holdings.append({
                    'ticker': esg_data.index[idx],
                    'weight': f"{weights[idx]:.1%}",
                    'esg_score': f"{esg_scores[idx]:.1f}",
                    'contribution': weights[idx] * esg_scores[idx]
                })
        
        return {
            'esg_distribution': {
                'high_esg_allocation': f"{high_esg_weight:.1%}",
                'medium_esg_allocation': f"{medium_esg_weight:.1%}",
                'low_esg_allocation': f"{low_esg_weight:.1%}"
            },
            'top_esg_holdings': top_holdings,
            'weighted_avg_esg': f"{np.dot(weights, esg_scores):.1f}"
        }
    
    def _create_carbon_analysis(self, weights: np.ndarray, esg_data: pd.DataFrame) -> Dict:
        """Create carbon footprint analysis."""
        carbon_intensities = esg_data['carbon_intensity'].values
        
        # Carbon distribution
        low_carbon = np.sum(weights[carbon_intensities < 50])
        medium_carbon = np.sum(weights[(carbon_intensities >= 50) & (carbon_intensities <= 200)])
        high_carbon = np.sum(weights[carbon_intensities > 200])
        
        # Carbon contributors
        carbon_contribution = weights * carbon_intensities
        high_carbon_contributors = []
        
        for i in np.argsort(carbon_contribution)[-5:][::-1]:
            if weights[i] > 0.01:
                high_carbon_contributors.append({
                    'ticker': esg_data.index[i],
                    'weight': f"{weights[i]:.1%}",
                    'carbon_intensity': f"{carbon_intensities[i]:.1f}",
                    'carbon_contribution': f"{carbon_contribution[i]:.1f}"
                })
        
        return {
            'carbon_distribution': {
                'low_carbon_allocation': f"{low_carbon:.1%}",
                'medium_carbon_allocation': f"{medium_carbon:.1%}",
                'high_carbon_allocation': f"{high_carbon:.1%}"
            },
            'waci': f"{np.dot(weights, carbon_intensities):.1f} tCO2e/$M",
            'high_carbon_contributors': high_carbon_contributors
        }
    
    def _create_sdg_analysis(self, weights: np.ndarray, esg_data: pd.DataFrame) -> Dict:
        """Create UN SDG alignment analysis."""
        # Simplified SDG mapping based on ESG scores
        e_scores = esg_data.get('e_score', esg_data['esg_score'] * 0.9).values
        s_scores = esg_data.get('s_score', esg_data['esg_score'] * 1.1).values
        
        sdg_scores = {
            'SDG_7_Clean_Energy': np.dot(weights, np.minimum(100, e_scores * 1.2)),
            'SDG_8_Decent_Work': np.dot(weights, np.minimum(100, s_scores * 1.1)),
            'SDG_13_Climate_Action': np.dot(weights, np.minimum(100, e_scores * 1.3)),
            'SDG_16_Peace_Justice': np.dot(weights, esg_data['esg_score'].values)
        }
        
        return {
            'sdg_alignment_scores': {k: f"{v:.1f}/100" for k, v in sdg_scores.items()},
            'overall_sdg_alignment': f"{np.mean(list(sdg_scores.values())):.1f}/100"
        }
    
    def _check_regulatory_compliance(self, results: Dict, esg_data: pd.DataFrame) -> Dict:
        """Check compliance with various frameworks."""
        compliance_results = {}
        
        for framework, check_func in self.compliance_frameworks.items():
            compliance_results[framework] = check_func(results, esg_data)
        
        return compliance_results
    
    def _check_sfdr_compliance(self, results: Dict, esg_data: pd.DataFrame) -> Dict:
        """Check EU SFDR (Sustainable Finance Disclosure Regulation) compliance."""
        esg_score = results['esg_score']
        carbon_intensity = results['carbon_intensity']
        
        # Simplified SFDR Article 8/9 criteria
        article_8_compliant = esg_score >= 60 and carbon_intensity <= 200
        article_9_compliant = esg_score >= 80 and carbon_intensity <= 100
        
        return {
            'framework': 'EU SFDR',
            'article_6_baseline': True,  # Basic disclosure
            'article_8_compliant': article_8_compliant,  # ESG promotion
            'article_9_compliant': article_9_compliant,  # Sustainable investment
            'recommendation': 'Article 9' if article_9_compliant else 'Article 8' if article_8_compliant else 'Article 6'
        }
    
    def _check_sdg_alignment(self, results: Dict, esg_data: pd.DataFrame) -> Dict:
        """Check UN SDG alignment."""
        esg_score = results['esg_score']
        
        alignment_level = 'High' if esg_score >= 80 else 'Medium' if esg_score >= 60 else 'Low'
        
        return {
            'framework': 'UN SDG',
            'alignment_level': alignment_level,
            'primary_sdgs': ['SDG 13 (Climate Action)', 'SDG 8 (Decent Work)', 'SDG 7 (Clean Energy)'],
            'sdg_score': f"{esg_score:.1f}/100"
        }
    
    def _check_tcfd_compliance(self, results: Dict, esg_data: pd.DataFrame) -> Dict:
        """Check TCFD (Task Force on Climate-related Financial Disclosures) compliance."""
        carbon_intensity = results['carbon_intensity']
        
        return {
            'framework': 'TCFD',
            'climate_risk_assessed': True,
            'carbon_disclosure': 'Full' if carbon_intensity < 100 else 'Partial',
            'transition_risk': 'Low' if carbon_intensity < 50 else 'Medium' if carbon_intensity < 200 else 'High',
            'scenario_analysis_needed': carbon_intensity > 100
        }
    
    def _check_sec_climate_disclosure(self, results: Dict, esg_data: pd.DataFrame) -> Dict:
        """Check US SEC climate disclosure requirements."""
        return {
            'framework': 'US SEC Climate',
            'scope_1_2_disclosure': True,  # Simplified
            'scope_3_assessment': results['carbon_intensity'] > 50,
            'climate_targets': results['carbon_intensity'] < 100,
            'governance_disclosure': results['esg_score'] > 70
        }
    
    def _create_peer_comparison(self, results: Dict) -> Dict:
        """Create peer comparison (simplified)."""
        # Benchmark against typical market metrics
        market_esg_avg = 65
        market_carbon_avg = 150
        market_sharpe_avg = 0.6
        
        return {
            'esg_vs_market': f"+{results['esg_score'] - market_esg_avg:.1f} points",
            'carbon_vs_market': f"{((results['carbon_intensity'] / market_carbon_avg) - 1)*100:+.1f}%",
            'sharpe_vs_market': f"+{results['sharpe_ratio'] - market_sharpe_avg:.3f}",
            'esg_percentile': min(99, max(1, int((results['esg_score'] / market_esg_avg) * 50)))
        }
    
    def _analyze_esg_trends(self, esg_data: pd.DataFrame) -> Dict:
        """Analyze ESG trends (simplified)."""
        return {
            'improving_esg': len(esg_data[esg_data['esg_score'] > 75]),
            'declining_esg': len(esg_data[esg_data['esg_score'] < 50]),
            'trend_summary': 'Positive ESG momentum across portfolio'
        }
    
    def _create_esg_risk_analysis(self, weights: np.ndarray, esg_data: pd.DataFrame) -> Dict:
        """Create ESG risk analysis."""
        esg_scores = esg_data['esg_score'].values
        
        # ESG risk concentration
        low_esg_exposure = np.sum(weights[esg_scores < 60])
        
        return {
            'esg_risk_concentration': f"{low_esg_exposure:.1%}",
            'esg_risk_level': 'Low' if low_esg_exposure < 0.2 else 'Medium' if low_esg_exposure < 0.4 else 'High',
            'diversification_score': f"{(1 - np.sum(weights**2)) * 100:.1f}/100"
        }
    
    def export_report(self, report: Dict, format: str = 'json', filename: Optional[str] = None) -> str:
        """Export report to file."""
        if filename is None:
            filename = f"esg_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if format == 'json':
            import json
            filename += '.json'
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        
        elif format == 'csv':
            # Convert key metrics to CSV
            filename += '.csv'
            summary_data = []
            for section, data in report.items():
                if isinstance(data, dict):
                    for key, value in data.items():
                        summary_data.append({
                            'section': section,
                            'metric': key,
                            'value': value
                        })
            
            pd.DataFrame(summary_data).to_csv(filename, index=False)
        
        return filename


# Example usage
if __name__ == "__main__":
    from quantum.esg_portfolio_optimizer import ESGPortfolioOptimizer
    
    print("ESG Reporting Engine Test")
    print("=" * 30)
    
    # Sample portfolio results (from optimizer)
    portfolio_results = {
        'weights': np.array([0.25, 0.25, 0.20, 0.15, 0.10, 0.05]),
        'expected_return': 0.105,
        'portfolio_risk': 0.142,
        'sharpe_ratio': 0.739,
        'esg_score': 76.2,
        'carbon_intensity': 45.8,
        'optimization_method': 'Quantum ESG'
    }
    
    # Sample ESG data
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'XOM']
    esg_data = pd.DataFrame({
        'esg_score': [82, 79, 88, 65, 72, 38],
        'e_score': [85, 82, 92, 58, 89, 22],
        's_score': [78, 75, 84, 68, 65, 48],
        'g_score': [84, 80, 88, 69, 62, 45],
        'carbon_intensity': [12.3, 8.7, 5.2, 45.1, 2.1, 890.2]
    }, index=tickers)
    
    # Generate report
    reporter = ESGReportingEngine()
    report = reporter.generate_esg_report(portfolio_results, esg_data, 'comprehensive')
    
    # Print key sections
    print("Portfolio Summary:")
    for key, value in report['portfolio_summary'].items():
        print(f"  {key}: {value}")
    
    print(f"\nESG Breakdown:")
    print(f"  Weighted Avg ESG: {report['esg_breakdown']['weighted_avg_esg']}")
    print(f"  High ESG Allocation: {report['esg_breakdown']['esg_distribution']['high_esg_allocation']}")
    
    print(f"\nCarbon Analysis:")
    print(f"  WACI: {report['carbon_analysis']['waci']}")
    print(f"  Low Carbon Allocation: {report['carbon_analysis']['carbon_distribution']['low_carbon_allocation']}")
    
    print(f"\nCompliance Status:")
    for framework, status in report['compliance_check'].items():
        print(f"  {framework}: {status.get('recommendation', status.get('alignment_level', 'Compliant'))}")
    
    # Export report
    # filename = reporter.export_report(report, 'json')
    # print(f"\nReport exported to: {filename}")
