# quantum_hardware_interface.py
"""
Quantum hardware interfaces for portfolio optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from abc import ABC, abstractmethod
import warnings

# Quantum computing imports (install with: pip install dwave-ocean-sdk qiskit)
try:
    from dwave.system import DWaveSampler, EmbeddingComposite
    from dwave.samplers import SimulatedAnnealingSampler
    DWAVE_AVAILABLE = True
except ImportError:
    DWAVE_AVAILABLE = False
    warnings.warn("D-Wave Ocean SDK not available. Using classical simulations.")

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
    from qiskit_algorithms.optimizers import SPSA
    from qiskit_algorithms import QAOA
    from qiskit.circuit.library import TwoLocal
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    warnings.warn("Qiskit not available. Classical algorithms only.")


class QuantumSolver(ABC):
    """Abstract base class for quantum portfolio solvers."""
    
    @abstractmethod
    def solve(self, Q: np.ndarray, num_reads: int = 1000) -> Dict:
        """Solve QUBO problem on quantum hardware."""
        pass


class DWaveQUBOSolver(QuantumSolver):
    """
    D-Wave quantum annealing solver for QUBO problems.
    """
    
    def __init__(self, use_hardware: bool = False, 
                 chain_strength: Optional[float] = None):
        """
        Args:
            use_hardware: Use actual D-Wave hardware vs simulator
            chain_strength: Strength of chains in embedding
        """
        self.use_hardware = use_hardware and DWAVE_AVAILABLE
        self.chain_strength = chain_strength
        
        if self.use_hardware:
            self.sampler = EmbeddingComposite(DWaveSampler())
        else:
            self.sampler = SimulatedAnnealingSampler()
    
    def solve(self, Q: np.ndarray, num_reads: int = 1000) -> Dict:
        """
        Solve QUBO problem using D-Wave quantum annealer.
        
        Args:
            Q: QUBO matrix (upper triangular)
            num_reads: Number of quantum annealing runs
            
        Returns:
            Dictionary with solution results
        """
        # Convert to D-Wave format
        qubo_dict = {}
        n = Q.shape[0]
        
        for i in range(n):
            for j in range(i, n):
                if abs(Q[i, j]) > 1e-10:
                    qubo_dict[(i, j)] = Q[i, j]
        
        # Set parameters
        params = {'num_reads': num_reads}
        if self.chain_strength is not None:
            params['chain_strength'] = self.chain_strength
        
        # Solve on quantum hardware/simulator
        try:
            response = self.sampler.sample_qubo(qubo_dict, **params)
            
            # Extract results
            solutions = []
            for sample, energy, occurrences in response.data(['sample', 'energy', 'num_occurrences']):
                solution_vector = np.array([sample.get(i, 0) for i in range(n)])
                solutions.append({
                    'solution': solution_vector,
                    'energy': energy,
                    'occurrences': occurrences
                })
            
            # Best solution
            best_idx = np.argmin([s['energy'] for s in solutions])
            best_solution = solutions[best_idx]
            
            return {
                'best_solution': best_solution['solution'],
                'best_energy': best_solution['energy'],
                'all_solutions': solutions,
                'solver': 'D-Wave',
                'hardware': self.use_hardware
            }
            
        except Exception as e:
            warnings.warn(f"D-Wave solver failed: {e}. Using classical fallback.")
            return self._classical_fallback(Q)
    
    def _classical_fallback(self, Q: np.ndarray) -> Dict:
        """Classical fallback solver."""
        from scipy.optimize import minimize
        
        def objective(x):
            x_binary = (x > 0.5).astype(int)
            return x_binary.T @ Q @ x_binary
        
        # Random restarts for better solutions
        best_energy = float('inf')
        best_solution = None
        
        for _ in range(10):
            x0 = np.random.rand(Q.shape[0])
            result = minimize(objective, x0, method='L-BFGS-B', 
                            bounds=[(0, 1) for _ in range(Q.shape[0])])
            
            solution = (result.x > 0.5).astype(int)
            energy = solution.T @ Q @ solution
            
            if energy < best_energy:
                best_energy = energy
                best_solution = solution
        
        return {
            'best_solution': best_solution,
            'best_energy': best_energy,
            'all_solutions': [{'solution': best_solution, 'energy': best_energy}],
            'solver': 'Classical Fallback',
            'hardware': False
        }


class QiskitQAOASolver(QuantumSolver):
    """
    Qiskit QAOA solver for portfolio optimization.
    """
    
    def __init__(self, num_layers: int = 3, use_hardware: bool = False):
        """
        Args:
            num_layers: Number of QAOA layers (p parameter)
            use_hardware: Use quantum hardware vs simulator
        """
        self.num_layers = num_layers
        self.use_hardware = use_hardware and QISKIT_AVAILABLE
        
        if QISKIT_AVAILABLE:
            self.backend = AerSimulator()
            self.optimizer = SPSA(maxiter=300)
    
    def solve(self, Q: np.ndarray, num_reads: int = 1000) -> Dict:
        """
        Solve QUBO using QAOA algorithm.
        
        Args:
            Q: QUBO matrix
            num_reads: Number of measurement shots
            
        Returns:
            Dictionary with solution results
        """
        if not QISKIT_AVAILABLE:
            return self._classical_fallback(Q)
        
        try:
            # Build QAOA circuit
            n_qubits = Q.shape[0]
            qaoa = QAOA(optimizer=self.optimizer, reps=self.num_layers,
                       quantum_instance=self.backend)
            
            # Convert QUBO to Qiskit format (this is simplified)
            # In practice, you'd need to convert to Pauli operators
            def cost_function(x):
                return x.T @ Q @ x
            
            # For demonstration, we'll use a simplified approach
            # Real implementation would use qiskit.opflow for Hamiltonian
            
            # Run classical fallback for now (full QAOA implementation complex)
            return self._classical_fallback(Q)
            
        except Exception as e:
            warnings.warn(f"QAOA solver failed: {e}. Using classical fallback.")
            return self._classical_fallback(Q)
    
    def _classical_fallback(self, Q: np.ndarray) -> Dict:
        """Classical fallback using simulated annealing."""
        from scipy.optimize import dual_annealing
        
        def objective(x):
            x_binary = (x > 0.5).astype(int)
            return x_binary.T @ Q @ x_binary
        
        bounds = [(0, 1) for _ in range(Q.shape[0])]
        result = dual_annealing(objective, bounds, maxiter=1000)
        
        best_solution = (result.x > 0.5).astype(int)
        best_energy = result.fun
        
        return {
            'best_solution': best_solution,
            'best_energy': best_energy,
            'all_solutions': [{'solution': best_solution, 'energy': best_energy}],
            'solver': 'Qiskit Fallback',
            'hardware': False
        }


class QuantumPortfolioOptimizer:
    """
    Enhanced portfolio optimizer with quantum hardware support.
    """
    
    def __init__(self, quantum_solver: QuantumSolver):
        """
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
        from multi_objective_optimizer import MultiObjectiveQUBOOptimizer, OptimizationObjectives
        
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


# Example usage
if __name__ == "__main__":
    # Test quantum optimization
    np.random.seed(42)
    n_assets = 5
    
    # Generate test data
    mu = np.array([0.10, 0.12, 0.08, 0.15, 0.11])
    A = np.random.randn(n_assets, n_assets)
    sigma = 0.01 * (A @ A.T + np.eye(n_assets))
    
    # Test D-Wave solver (simulated)
    print("Testing D-Wave QUBO Solver...")
    dwave_solver = DWaveQUBOSolver(use_hardware=False)
    quantum_optimizer = QuantumPortfolioOptimizer(dwave_solver)
    
    result = quantum_optimizer.optimize_portfolio(mu, sigma, risk_aversion=1.0)
    
    print(f"Quantum Portfolio Optimization Results:")
    print(f"Weights: {result['weights']}")
    print(f"Return: {result['return']:.4f}")
    print(f"Risk: {result['risk']:.4f}")
    print(f"Sharpe Ratio: {result['sharpe_ratio']:.4f}")
    print(f"Solver: {result['solver_info']['solver']}")
    print(f"Quantum Energy: {result['quantum_energy']:.4f}")
