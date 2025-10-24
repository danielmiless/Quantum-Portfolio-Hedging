# src/quantum/quantum_hardware_interface.py
"""
Quantum hardware interfaces with environment variable support and fixed imports
"""

import numpy as np
import sys
import os
from typing import Dict, List, Tuple, Optional
from abc import ABC, abstractmethod
from pathlib import Path
import warnings

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
        DWAVE_API_TOKEN = os.getenv('DWAVE_API_TOKEN')
        DWAVE_SOLVER = os.getenv('DWAVE_SOLVER', 'Advantage_system4.1')

# D-Wave Ocean SDK (Quantum Annealer Backend)
try:
    from dwave.system import DWaveSampler, EmbeddingComposite
    from dwave.samplers import SimulatedAnnealingSampler
    import dimod
    DWAVE_AVAILABLE = True
except ImportError:
    DWAVE_AVAILABLE = False
    warnings.warn("D-Wave Ocean SDK not available. Install with: pip install dwave-ocean-sdk")

# IBM Qiskit (Gate-based Quantum Backend)
try:
    from qiskit import QuantumCircuit, transpile, Aer
    from qiskit.algorithms.optimizers import COBYLA, SPSA
    from qiskit.algorithms import QAOA, VQE
    from qiskit.opflow import PauliSumOp
    from qiskit.utils import QuantumInstance
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    warnings.warn("Qiskit not available. Install with: pip install qiskit")


class QuantumSolver(ABC):
    """Abstract base class for quantum solvers."""
    
    @abstractmethod
    def solve(self, Q: np.ndarray, num_reads: int = 1000) -> Dict:
        """Solve QUBO problem."""
        pass


class DWaveQUBOSolver(QuantumSolver):
    """D-Wave quantum annealing solver using environment variables."""
    
    def __init__(self, use_hardware: bool = False, 
                 chain_strength: Optional[float] = None,
                 token: Optional[str] = None,
                 solver: Optional[str] = None):
        """
        Initialize D-Wave solver.
        
        Args:
            use_hardware: Use actual D-Wave hardware vs simulator
            chain_strength: Strength of chains in embedding
            token: D-Wave API token (uses Config.DWAVE_API_TOKEN if None)
            solver: Solver name (uses Config.DWAVE_SOLVER if None)
        """
        self.use_hardware = use_hardware and DWAVE_AVAILABLE
        self.chain_strength = chain_strength
        
        # Use environment variable if token not provided
        self.token = token or Config.DWAVE_API_TOKEN
        self.solver_name = solver or Config.DWAVE_SOLVER
        
        if self.use_hardware:
            if not DWAVE_AVAILABLE:
                warnings.warn("D-Wave Ocean SDK not available. Falling back to simulator.")
                self.use_hardware = False
                self.sampler = SimulatedAnnealingSampler()
            else:
                try:
                    # Try to connect to D-Wave hardware
                    if self.token:
                        self.sampler = EmbeddingComposite(
                            DWaveSampler(token=self.token, solver=self.solver_name)
                        )
                        print(f"✅ Connected to D-Wave hardware: {self.solver_name}")
                    else:
                        # Try to use default config or env var
                        self.sampler = EmbeddingComposite(DWaveSampler())
                        print("✅ Connected to D-Wave hardware (using default config)")
                except Exception as e:
                    warnings.warn(f"D-Wave hardware connection failed: {e}. Using simulator.")
                    self.use_hardware = False
                    self.sampler = SimulatedAnnealingSampler()
        else:
            # Always use simulator when use_hardware=False
            if DWAVE_AVAILABLE:
                self.sampler = SimulatedAnnealingSampler()
                print("Using D-Wave simulator (set use_hardware=True for quantum hardware)")
            else:
                warnings.warn("D-Wave SDK not available. Install with: pip install dwave-ocean-sdk")
                self.sampler = None
    
    def solve(self, Q: np.ndarray, num_reads: int = 1000) -> Dict:
        """
        Solve QUBO problem using D-Wave.
        
        Args:
            Q: QUBO matrix
            num_reads: Number of samples to take
            
        Returns:
            Dictionary with solution and metadata
        """
        if self.sampler is None:
            raise RuntimeError("No D-Wave sampler available")
        
        # Convert numpy array to QUBO dictionary
        qubo_dict = {}
        n = Q.shape[0]
        
        for i in range(n):
            for j in range(i, n):
                if Q[i, j] != 0:
                    qubo_dict[(i, j)] = Q[i, j]
        
        # Create BQM
        bqm = dimod.BinaryQuadraticModel.from_qubo(qubo_dict)
        
        # Sample
        if self.use_hardware:
            # Hardware sampler
            sample_kwargs = {'num_reads': num_reads}
            if self.chain_strength is not None:
                sample_kwargs['chain_strength'] = self.chain_strength
            sampleset = self.sampler.sample(bqm, **sample_kwargs)
        else:
            # Simulator
            sampleset = self.sampler.sample(bqm, num_reads=num_reads)
        
        # Extract best solution
        best_sample = sampleset.first.sample
        best_energy = sampleset.first.energy
        
        # Convert to array
        solution = np.array([best_sample[i] for i in range(n)])
        
        # Collect all solutions
        all_solutions = []
        for sample, energy in sampleset.data(['sample', 'energy']):
            sol_array = np.array([sample[i] for i in range(n)])
            all_solutions.append({
                'solution': sol_array,
                'energy': energy
            })
        
        return {
            'best_solution': solution,
            'best_energy': best_energy,
            'all_solutions': all_solutions,
            'num_reads': num_reads,
            'solver': 'D-Wave Hardware' if self.use_hardware else 'D-Wave Simulator',
            'hardware': self.use_hardware
        }


class QuantumPortfolioOptimizer:
    """Quantum portfolio optimizer using D-Wave or Qiskit."""
    
    def __init__(self, quantum_solver: QuantumSolver):
        """
        Initialize optimizer.
        
        Args:
            quantum_solver: Quantum solver instance
        """
        self.quantum_solver = quantum_solver
    
    def optimize_portfolio(self, mu: np.ndarray, sigma: np.ndarray,
                          risk_aversion: float = 1.0,
                          n_bits: int = 4) -> Dict:
        """
        Optimize portfolio using quantum hardware.
        
        Args:
            mu: Expected returns
            sigma: Covariance matrix
            risk_aversion: Risk aversion parameter
            n_bits: Bits per asset for discretization
            
        Returns:
            Optimization results with quantum solution
        """
        # Try to import multi_objective_optimizer
        try:
            # Try relative import first
            try:
                from multi_objective_optimizer import MultiObjectiveQUBOOptimizer, OptimizationObjectives
            except ImportError:
                # Try absolute import from quantum module
                from quantum.multi_objective_optimizer import MultiObjectiveQUBOOptimizer, OptimizationObjectives
        except ImportError:
            # Create a simplified QUBO builder as fallback
            return self._fallback_portfolio_optimization(mu, sigma, risk_aversion, n_bits)
        
        # Build QUBO matrix
        optimizer = MultiObjectiveQUBOOptimizer(mu, sigma)
        objectives = OptimizationObjectives(
            maximize_return=True,
            minimize_risk=True,
            risk_aversion=risk_aversion
        )
        
        Q, offset = optimizer.build_multi_objective_qubo(objectives, n_bits)
        
        # Solve on quantum hardware
        quantum_result = self.quantum_solver.solve(Q, num_reads=1000)
        
        # Decode solution
        weights = optimizer.decode_solution(quantum_result['best_solution'], n_bits)
        
        # Calculate portfolio metrics
        portfolio_return = np.dot(weights, mu)
        portfolio_risk = np.sqrt(np.dot(weights, np.dot(sigma, weights)))
        sharpe_ratio = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0
        
        return {
            'weights': weights,
            'return': portfolio_return,
            'risk': portfolio_risk,
            'sharpe_ratio': sharpe_ratio,
            'quantum_energy': quantum_result['best_energy'],
            'solver_info': {
                'solver': quantum_result['solver'],
                'hardware': quantum_result['hardware'],
                'n_solutions': len(quantum_result['all_solutions'])
            }
        }
    
    def _fallback_portfolio_optimization(self, mu: np.ndarray, sigma: np.ndarray,
                                       risk_aversion: float, n_bits: int) -> Dict:
        """Simplified portfolio optimization without multi_objective_optimizer."""
        print("⚠️  Using fallback optimization (multi_objective_optimizer not found)")
        
        n_assets = len(mu)
        n_vars = n_assets * n_bits
        Q = np.zeros((n_vars, n_vars))
        max_val = 2**n_bits - 1
        
        # Simple QUBO: maximize return - risk_aversion * risk
        for i in range(n_assets):
            for k in range(n_bits):
                idx = i * n_bits + k
                bit_value = 2**k / max_val
                # Return term (maximize)
                Q[idx, idx] -= mu[i] * bit_value
                
        # Risk term (minimize)
        for i in range(n_assets):
            for j in range(n_assets):
                for k1 in range(n_bits):
                    for k2 in range(n_bits):
                        idx1 = i * n_bits + k1
                        idx2 = j * n_bits + k2
                        bit_val1 = 2**k1 / max_val
                        bit_val2 = 2**k2 / max_val
                        
                        if idx1 <= idx2:
                            Q[idx1, idx2] += risk_aversion * sigma[i, j] * bit_val1 * bit_val2
        
        # Budget constraint
        budget_penalty = 5.0
        for i in range(n_assets):
            for j in range(n_assets):
                for k1 in range(n_bits):
                    for k2 in range(n_bits):
                        idx1 = i * n_bits + k1
                        idx2 = j * n_bits + k2
                        bit_val1 = 2**k1 / max_val
                        bit_val2 = 2**k2 / max_val
                        
                        if idx1 <= idx2:
                            Q[idx1, idx2] += budget_penalty * bit_val1 * bit_val2
        
        for i in range(n_assets):
            for k in range(n_bits):
                idx = i * n_bits + k
                bit_value = 2**k / max_val
                Q[idx, idx] -= 2 * budget_penalty * bit_value
        
        # Solve
        quantum_result = self.quantum_solver.solve(Q, num_reads=1000)
        
        # Decode weights
        solution = quantum_result['best_solution']
        weights = np.zeros(n_assets)
        
        for i in range(n_assets):
            for k in range(n_bits):
                idx = i * n_bits + k
                if idx < len(solution) and solution[idx] > 0.5:
                    weights[i] += 2**k / max_val
        
        # Normalize weights
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            weights = np.ones(n_assets) / n_assets
        
        # Calculate metrics
        portfolio_return = np.dot(weights, mu)
        portfolio_risk = np.sqrt(np.dot(weights, np.dot(sigma, weights)))
        sharpe_ratio = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0
        
        return {
            'weights': weights,
            'return': portfolio_return,
            'risk': portfolio_risk,
            'sharpe_ratio': sharpe_ratio,
            'quantum_energy': quantum_result['best_energy'],
            'solver_info': {
                'solver': quantum_result['solver'],
                'hardware': quantum_result['hardware'],
                'n_solutions': len(quantum_result['all_solutions'])
            }
        }


# Example usage
if __name__ == "__main__":
    print("Quantum Hardware Interface Test")
    print("=" * 45)
    
    # Test with simulator (no API token required)
    print("\n1. Testing D-Wave Simulator:")
    try:
        solver = DWaveQUBOSolver(use_hardware=False)
        
        # Simple QUBO test
        Q = np.array([
            [-1, 2, 2],
            [2, -1, 2],
            [2, 2, -1]
        ])
        
        result = solver.solve(Q, num_reads=100)
        print(f"  Solution: {result['best_solution']}")
        print(f"  Energy: {result['best_energy']:.4f}")
        print(f"  Solver: {result['solver']}")
        
    except Exception as e:
        print(f"  ❌ Simulator test failed: {e}")
    
    # Test with hardware (requires API token)
    print("\n2. Testing D-Wave Hardware:")
    if Config.DWAVE_API_TOKEN:
        try:
            solver = DWaveQUBOSolver(use_hardware=True)
            result = solver.solve(Q, num_reads=100)
            print(f"  Solution: {result['best_solution']}")
            print(f"  Energy: {result['best_energy']:.4f}")
            print(f"  Solver: {result['solver']}")
            
        except Exception as e:
            print(f"  ⚠️  Hardware test failed: {e}")
            print(f"  This is normal if you don't have D-Wave access")
    else:
        print("  ⚠️  D-Wave API token not found")
        print("  Set DWAVE_API_TOKEN in .env file to test hardware")
    
    # Test portfolio optimization
    print("\n3. Testing Quantum Portfolio Optimization:")
    try:
        solver = DWaveQUBOSolver(use_hardware=False)
        optimizer = QuantumPortfolioOptimizer(solver)
        
        # Sample portfolio data
        n_assets = 5
        np.random.seed(42)  # For reproducible results
        mu = np.random.normal(0.08, 0.04, n_assets)
        A = np.random.randn(n_assets, n_assets)
        sigma = 0.01 * (A @ A.T + np.eye(n_assets))
        
        print("  Sample portfolio:")
        print(f"    Expected returns: {mu}")
        print(f"    Risk matrix shape: {sigma.shape}")
        
        result = optimizer.optimize_portfolio(mu, sigma, risk_aversion=1.0)
        
        print(f"  ✅ Optimization Results:")
        print(f"    Optimal Weights: {result['weights']}")
        print(f"    Expected Return: {result['return']:.2%}")
        print(f"    Portfolio Risk: {result['risk']:.2%}")
        print(f"    Sharpe Ratio: {result['sharpe_ratio']:.3f}")
        print(f"    Solver: {result['solver_info']['solver']}")
        
    except Exception as e:
        print(f"  ❌ Portfolio optimization failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ Quantum hardware interface test completed")
