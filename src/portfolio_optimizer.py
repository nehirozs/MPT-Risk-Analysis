"""
Portfolio optimization module implementing Modern Portfolio Theory.

This module provides functions for portfolio statistics and optimization
algorithms including Minimum Variance and Maximum Sharpe Ratio portfolios.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Tuple, Optional, List, Dict


class PortfolioOptimizer:
    """Portfolio optimization using Modern Portfolio Theory."""
    
    def __init__(self, returns_df: pd.DataFrame, risk_free_rate: float = 0.0, regularization: float = 1e-5):
        """
        Initialize PortfolioOptimizer.
        
        Parameters:
        -----------
        returns_df : pd.DataFrame
            DataFrame with asset returns (columns are assets, index is dates)
        risk_free_rate : float, optional
            Risk-free rate (default: 0.0)
        regularization : float, optional
            Regularization parameter for covariance matrix diagonal (default: 1e-5)
            Helps with numerical stability for ill-conditioned matrices
        """
        self.returns_df = returns_df
        self.assets = returns_df.columns.tolist()
        self.n_assets = len(self.assets)
        self.risk_free_rate = risk_free_rate
        self.regularization = regularization
        
        # Calculate expected returns and covariance matrix
        self.mean_returns = self.calculate_expected_returns(returns_df)
        self.cov_matrix = self.calculate_covariance_matrix(returns_df)
        
        # Apply regularization to covariance matrix for numerical stability
        if regularization > 0:
            self.cov_matrix = self.cov_matrix + regularization * np.eye(self.n_assets)
    
    @property
    def num_assets(self) -> int:
        """Number of assets in the portfolio (alias for n_assets)."""
        return self.n_assets
    
    @property
    def tickers(self) -> List[str]:
        """List of asset tickers (alias for assets)."""
        return self.assets
    
    @staticmethod
    def calculate_expected_returns(returns_df: pd.DataFrame) -> np.ndarray:
        """
        Calculate expected (mean) returns for each asset, annualized.
        
        Assumes returns are daily and annualizes by multiplying by 252 trading days.
        
        Parameters:
        -----------
        returns_df : pd.DataFrame
            DataFrame with asset returns (assumed to be daily)
        
        Returns:
        --------
        np.ndarray
            Array of annualized expected returns
        """
        return returns_df.mean().values * 252
    
    @staticmethod
    def calculate_covariance_matrix(returns_df: pd.DataFrame) -> np.ndarray:
        """
        Calculate covariance matrix of returns, annualized.
        
        Assumes returns are daily and annualizes by multiplying by 252 trading days.
        
        Parameters:
        -----------
        returns_df : pd.DataFrame
            DataFrame with asset returns (assumed to be daily)
        
        Returns:
        --------
        np.ndarray
            Annualized covariance matrix
        """
        return returns_df.cov().values * 252
    
    @staticmethod
    def portfolio_return(weights: np.ndarray, mean_returns: np.ndarray) -> float:
        """
        Calculate portfolio expected return.
        
        Formula: R_p = sum(w_i * R_i)
        
        Parameters:
        -----------
        weights : np.ndarray
            Portfolio weights
        mean_returns : np.ndarray
            Expected returns for each asset
        
        Returns:
        --------
        float
            Portfolio expected return
        """
        return np.dot(weights, mean_returns)
    
    @staticmethod
    def portfolio_volatility(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
        """
        Calculate portfolio volatility (standard deviation).
        
        Formula: σ_p = sqrt(w^T Σ w)
        
        Parameters:
        -----------
        weights : np.ndarray
            Portfolio weights
        cov_matrix : np.ndarray
            Covariance matrix
        
        Returns:
        --------
        float
            Portfolio volatility
        """
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    @staticmethod
    def sharpe_ratio(
        weights: np.ndarray,
        mean_returns: np.ndarray,
        cov_matrix: np.ndarray,
        risk_free_rate: float = 0.0
    ) -> float:
        """
        Calculate Sharpe ratio for a portfolio.
        
        Formula: SR = (R_p - R_f) / σ_p
        
        Parameters:
        -----------
        weights : np.ndarray
            Portfolio weights
        mean_returns : np.ndarray
            Expected returns for each asset
        cov_matrix : np.ndarray
            Covariance matrix
        risk_free_rate : float
            Risk-free rate (default: 0.0)
        
        Returns:
        --------
        float
            Sharpe ratio
        """
        portfolio_ret = PortfolioOptimizer.portfolio_return(weights, mean_returns)
        portfolio_vol = PortfolioOptimizer.portfolio_volatility(weights, cov_matrix)
        
        if portfolio_vol == 0:
            return 0.0
        
        return (portfolio_ret - risk_free_rate) / portfolio_vol
    
    def minimum_variance_portfolio(self) -> pd.Series:
        """
        Find the minimum variance portfolio with robust optimization.
        
        Minimizes: σ_p^2 = w^T Σ w
        Subject to: sum(w_i) = 1, w_i >= 0
        
        Uses multiple optimization methods for robustness and includes
        regularization for numerical stability.
        
        Returns:
        --------
        pd.Series
            Optimal weights for minimum variance portfolio
        """
        def objective(weights):
            return self.portfolio_volatility(weights, self.cov_matrix) ** 2
        
        # Constraints
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        
        # Bounds: weights between 0 and 1 (no shorting)
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        
        # Initial guess: equal weights
        initial_weights = np.array([1 / self.n_assets] * self.n_assets)
        
        # Try multiple optimization methods for robustness
        methods_to_try = ['SLSQP', 'trust-constr']
        best_result = None
        best_variance = np.inf
        
        for method in methods_to_try:
            try:
                result = minimize(
                    objective,
                    initial_weights,
                    method=method,
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 500, 'ftol': 1e-9} if method == 'SLSQP' else {'maxiter': 500}
                )
                
                # Check if optimization succeeded and if result is better
                if result.success:
                    variance = objective(result.x)
                    if variance < best_variance:
                        best_variance = variance
                        best_result = result
                # Also try to use result even if not fully successful but has valid weights
                elif result.x is not None and np.all(result.x >= 0) and np.abs(np.sum(result.x) - 1) < 0.01:
                    variance = objective(result.x)
                    if variance < best_variance:
                        best_variance = variance
                        best_result = result
            except Exception:
                # Continue to next method if this one fails
                continue
        
        # If we found a valid result, use it
        if best_result is not None and best_result.x is not None:
            weights = best_result.x
            # Ensure weights are non-negative and sum to 1
            weights = np.maximum(weights, 0)
            weights = weights / np.sum(weights)
            return pd.Series(weights, index=self.assets)
        
        # If all methods failed, raise error
        raise ValueError("Optimization failed: All methods failed to converge")
    
    def maximum_sharpe_ratio_portfolio(
        self,
        risk_free_rate: Optional[float] = None
    ) -> pd.Series:
        """
        Find the maximum Sharpe ratio portfolio.
        
        Maximizes: SR = (R_p - R_f) / σ_p
        Subject to: sum(w_i) = 1, w_i >= 0
        
        Parameters:
        -----------
        risk_free_rate : float, optional
            Risk-free rate (default: uses self.risk_free_rate if set, otherwise 0.0)
        
        Returns:
        --------
        pd.Series
            Optimal weights for maximum Sharpe ratio portfolio
        """
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
        
        def negative_sharpe(weights):
            return -self.sharpe_ratio(weights, self.mean_returns, self.cov_matrix, risk_free_rate)
        
        # Constraints
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        
        # Bounds: weights between 0 and 1 (no shorting)
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        
        # Initial guess: equal weights
        initial_weights = np.array([1 / self.n_assets] * self.n_assets)
        
        # Optimize (minimize negative Sharpe = maximize Sharpe)
        result = minimize(
            negative_sharpe,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if not result.success:
            raise ValueError(f"Optimization failed: {result.message}")
        
        return pd.Series(result.x, index=self.assets)
    
    def maximum_sharpe_portfolio(self, risk_free_rate: Optional[float] = None) -> pd.Series:
        """
        Alias for maximum_sharpe_ratio_portfolio().
        
        Find the maximum Sharpe ratio portfolio.
        
        Parameters:
        -----------
        risk_free_rate : float, optional
            Risk-free rate (default: uses self.risk_free_rate if set, otherwise 0.0)
        
        Returns:
        --------
        pd.Series
            Optimal weights for maximum Sharpe ratio portfolio
        """
        return self.maximum_sharpe_ratio_portfolio(risk_free_rate=risk_free_rate)
    
    def efficient_frontier(
        self,
        n_portfolios: Optional[int] = None,
        num_portfolios: Optional[int] = None,
        risk_free_rate: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Generate efficient frontier portfolios.
        
        Parameters:
        -----------
        n_portfolios : int, optional
            Number of portfolios to generate along the frontier (default: 50)
        num_portfolios : int, optional
            Alias for n_portfolios
        risk_free_rate : float, optional
            Risk-free rate for Sharpe ratio calculation (default: uses self.risk_free_rate if set, otherwise 0.0)
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with columns: 'Return', 'Volatility', 'Sharpe', 'weights'
        """
        # Handle parameter aliases
        if num_portfolios is not None:
            n_portfolios = num_portfolios
        if n_portfolios is None:
            n_portfolios = 50
        
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
        
        # Find min and max returns
        min_ret = self.mean_returns.min()
        max_ret = self.mean_returns.max()
        
        # Generate target returns
        target_returns = np.linspace(min_ret, max_ret, n_portfolios)
        
        efficient_portfolios = []
        
        for target_return in target_returns:
            try:
                weights = self._optimize_for_target_return(target_return)
                portfolio_ret = self.portfolio_return(weights, self.mean_returns)
                portfolio_vol = self.portfolio_volatility(weights, self.cov_matrix)
                sharpe = self.sharpe_ratio(weights, self.mean_returns, self.cov_matrix, risk_free_rate)
                
                efficient_portfolios.append({
                    'return': portfolio_ret,
                    'volatility': portfolio_vol,
                    'sharpe': sharpe,
                    'weights': weights.copy()
                })
            except:
                # Skip infeasible portfolios
                continue
        
        frontier_df = pd.DataFrame(efficient_portfolios)
        
        # Rename columns to capitalized versions for compatibility
        if not frontier_df.empty:
            frontier_df = frontier_df.rename(columns={
                'return': 'Return',
                'volatility': 'Volatility',
                'sharpe': 'Sharpe'
            })
        
        return frontier_df
    
    def _optimize_for_target_return(self, target_return: float) -> np.ndarray:
        """
        Find portfolio weights for a given target return.
        
        Parameters:
        -----------
        target_return : float
            Target portfolio return
        
        Returns:
        --------
        np.ndarray
            Optimal weights
        """
        def objective(weights):
            return self.portfolio_volatility(weights, self.cov_matrix) ** 2
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w: self.portfolio_return(w, self.mean_returns) - target_return}
        ]
        
        # Bounds
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        
        # Initial guess
        initial_weights = np.array([1 / self.n_assets] * self.n_assets)
        
        # Optimize
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if not result.success:
            raise ValueError(f"Optimization failed for target return {target_return}")
        
        return result.x
    
    def minimum_variance_portfolio_constrained(
        self,
        max_position_size: Optional[float] = None,
        min_position_size: Optional[float] = None,
        sector_limits: Optional[Dict[str, float]] = None,
        sector_mapping: Optional[Dict[str, str]] = None,
        max_turnover: Optional[float] = None,
        current_weights: Optional[np.ndarray] = None,
        min_holdings: Optional[int] = None,
        max_holdings: Optional[int] = None
    ) -> pd.Series:
        """
        Find minimum variance portfolio with real-world constraints.
        
        Parameters:
        -----------
        max_position_size : float, optional
            Maximum weight per asset (e.g., 0.20 for 20%)
        min_position_size : float, optional
            Minimum weight per asset if held (e.g., 0.05 for 5%)
        sector_limits : Dict[str, float], optional
            Maximum weight per sector (e.g., {'Tech': 0.40})
        sector_mapping : Dict[str, str], optional
            Asset to sector mapping (e.g., {'AAPL': 'Tech'})
        max_turnover : float, optional
            Maximum turnover from current weights (e.g., 0.30 for 30%)
        current_weights : np.ndarray, optional
            Current portfolio weights for turnover constraint
        min_holdings : int, optional
            Minimum number of assets to hold
        max_holdings : int, optional
            Maximum number of assets to hold
        
        Returns:
        --------
        pd.Series
            Optimal constrained weights
        """
        def objective(weights):
            return self.portfolio_volatility(weights, self.cov_matrix) ** 2
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        
        # Bounds
        if max_position_size is not None:
            bounds = tuple((0, max_position_size) for _ in range(self.n_assets))
        else:
            bounds = tuple((0, 1) for _ in range(self.n_assets))
        
        # Sector limits
        if sector_limits is not None and sector_mapping is not None:
            for sector, max_weight in sector_limits.items():
                sector_assets = [i for i, asset in enumerate(self.assets) 
                                if sector_mapping.get(asset) == sector]
                if sector_assets:
                    constraints.append({
                        'type': 'ineq',
                        'fun': lambda w, sa=sector_assets, mw=max_weight: 
                               mw - np.sum([w[i] for i in sa])
                    })
        
        # Turnover constraint
        if max_turnover is not None and current_weights is not None:
            constraints.append({
                'type': 'ineq',
                'fun': lambda w: max_turnover - np.sum(np.abs(w - current_weights))
            })
        
        # Min holdings constraint (approximate using L1 norm)
        if min_holdings is not None:
            # Approximate using sum of weights above threshold
            threshold = 1.0 / self.n_assets * 0.5
            constraints.append({
                'type': 'ineq',
                'fun': lambda w: np.sum(w >= threshold) - min_holdings
            })
        
        initial_weights = np.array([1 / self.n_assets] * self.n_assets)
        
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if not result.success:
            raise ValueError(f"Constrained optimization failed: {result.message}")
        
        # Enforce min holdings and max holdings
        weights = pd.Series(result.x, index=self.assets)
        
        if min_holdings is not None or max_holdings is not None:
            # Set small weights to zero
            weights[weights < 0.001] = 0
            weights = weights / weights.sum()  # Renormalize
            
            if min_holdings is not None and (weights > 0.001).sum() < min_holdings:
                # Add smallest positions to meet minimum
                n_needed = min_holdings - (weights > 0.001).sum()
                zero_positions = weights[weights < 0.001].index[:n_needed]
                for pos in zero_positions:
                    weights[pos] = 1.0 / min_holdings
                weights = weights / weights.sum()
            
            if max_holdings is not None and (weights > 0.001).sum() > max_holdings:
                # Keep only top N positions
                top_positions = weights.nlargest(max_holdings).index
                weights[~weights.index.isin(top_positions)] = 0
                weights = weights / weights.sum()
        
        # Apply min position size if specified
        if min_position_size is not None:
            weights[weights > 0] = np.maximum(weights[weights > 0], min_position_size)
            weights = weights / weights.sum()
        
        return weights
    
    def maximum_sharpe_ratio_portfolio_constrained(
        self,
        risk_free_rate: float = 0.0,
        max_position_size: Optional[float] = None,
        sector_limits: Optional[Dict[str, float]] = None,
        sector_mapping: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> pd.Series:
        """
        Find maximum Sharpe ratio portfolio with constraints.
        
        Parameters:
        -----------
        risk_free_rate : float
            Risk-free rate
        max_position_size : float, optional
            Maximum weight per asset
        sector_limits : Dict[str, float], optional
            Maximum weight per sector
        sector_mapping : Dict[str, str], optional
            Asset to sector mapping
        **kwargs
            Additional constraints (passed to minimum_variance_portfolio_constrained)
        
        Returns:
        --------
        pd.Series
            Optimal constrained weights
        """
        def negative_sharpe(weights):
            return -self.sharpe_ratio(weights, self.mean_returns, self.cov_matrix, risk_free_rate)
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        
        # Bounds
        if max_position_size is not None:
            bounds = tuple((0, max_position_size) for _ in range(self.n_assets))
        else:
            bounds = tuple((0, 1) for _ in range(self.n_assets))
        
        # Sector limits
        if sector_limits is not None and sector_mapping is not None:
            for sector, max_weight in sector_limits.items():
                sector_assets = [i for i, asset in enumerate(self.assets) 
                                if sector_mapping.get(asset) == sector]
                if sector_assets:
                    constraints.append({
                        'type': 'ineq',
                        'fun': lambda w, sa=sector_assets, mw=max_weight: 
                               mw - np.sum([w[i] for i in sa])
                    })
        
        initial_weights = np.array([1 / self.n_assets] * self.n_assets)
        
        result = minimize(
            negative_sharpe,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if not result.success:
            raise ValueError(f"Constrained optimization failed: {result.message}")
        
        weights = pd.Series(result.x, index=self.assets)
        weights[weights < 0.001] = 0
        weights = weights / weights.sum()
        
        return weights
    
    def risk_parity_portfolio(self) -> pd.Series:
        """
        Calculate Risk Parity portfolio weights.
        
        Risk parity allocates portfolio risk equally across all assets,
        rather than allocating capital equally. This typically results in
        more balanced risk contribution from each asset.
        
        Minimizes: sum((risk_contribution_i - target_risk)²)
        where risk_contribution_i = w_i * (Σw)_i / σ_p
        
        Returns:
        --------
        pd.Series
            Risk parity portfolio weights
        """
        def risk_parity_objective(weights):
            """Objective: minimize sum of squared differences in risk contributions."""
            # Portfolio volatility
            portfolio_vol = self.portfolio_volatility(weights, self.cov_matrix)
            
            if portfolio_vol < 1e-8:
                return 1e10  # Penalty for near-zero volatility
            
            # Marginal contribution to risk (MCTR) for each asset
            mctr = np.dot(self.cov_matrix, weights) / portfolio_vol
            
            # Risk contribution for each asset
            risk_contrib = weights * mctr
            
            # Target risk contribution (equal for all assets)
            target_risk = 1.0 / self.n_assets
            
            # Sum of squared differences from target
            objective = np.sum((risk_contrib - target_risk) ** 2)
            
            return objective
        
        # Constraints
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        
        # Bounds: weights between 0 and 1 (no shorting)
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        
        # Initial guess: equal weights
        initial_weights = np.array([1 / self.n_assets] * self.n_assets)
        
        # Optimize
        result = minimize(
            risk_parity_objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if not result.success:
            raise ValueError(f"Risk parity optimization failed: {result.message}")
        
        return pd.Series(result.x, index=self.assets)
    
    def black_litterman_portfolio(
        self,
        market_weights: Optional[np.ndarray] = None,
        market_cap_weights: Optional[pd.Series] = None,
        views: Optional[Dict[str, Tuple[float, np.ndarray]]] = None,
        risk_aversion: float = 3.0,
        tau: float = 0.05,
        omega: Optional[np.ndarray] = None
    ) -> pd.Series:
        """
        Calculate Black-Litterman optimal portfolio weights.
        
        Black-Litterman model combines market equilibrium returns with
        investor views to generate expected returns that lead to more
        stable and intuitive portfolio allocations.
        
        Parameters:
        -----------
        market_weights : np.ndarray, optional
            Market portfolio weights (default: equal weights)
        market_cap_weights : pd.Series, optional
            Market capitalization weights as alternative to market_weights
        views : Dict[str, Tuple[float, np.ndarray]], optional
            Investor views as dict of {view_name: (return, pick_vector)}
            where pick_vector indicates which assets are in the view
            Example: {'view1': (0.05, np.array([1, -1, 0]))} means
            asset 1 outperforms asset 2 by 5%
        risk_aversion : float
            Risk aversion parameter (lambda, default: 3.0)
        tau : float
            Scaling factor for uncertainty in prior (default: 0.05)
        omega : np.ndarray, optional
            Uncertainty matrix for views (default: diagonal matrix)
        
        Returns:
        --------
        pd.Series
            Black-Litterman optimal portfolio weights
        """
        # Default to equal weights if no market weights provided
        if market_weights is None:
            if market_cap_weights is not None:
                market_weights = market_cap_weights.values
            else:
                market_weights = np.array([1.0 / self.n_assets] * self.n_assets)
        
        # Calculate equilibrium returns (Pi)
        # Pi = δ * Σ * w_market
        # where δ (delta) is risk aversion parameter
        equilibrium_returns = risk_aversion * np.dot(self.cov_matrix, market_weights)
        
        # If no views provided, return market portfolio
        if views is None or len(views) == 0:
            return pd.Series(market_weights, index=self.assets)
        
        # Build view matrix P and view vector Q
        n_views = len(views)
        P = np.zeros((n_views, self.n_assets))
        Q = np.zeros(n_views)
        
        for idx, (view_name, (return_view, pick_vector)) in enumerate(views.items()):
            # Normalize pick vector to sum to 0 for relative views
            if pick_vector.sum() == 0:
                # Relative view
                P[idx, :] = pick_vector
            else:
                # Absolute view (less common)
                P[idx, :] = pick_vector / pick_vector.sum()
            Q[idx] = return_view
        
        # Build uncertainty matrix Omega (diagonal if not provided)
        if omega is None:
            # Use simplified approach: tau * P * Σ * P^T
            omega = np.diag(np.diag(tau * np.dot(P, np.dot(self.cov_matrix, P.T))))
        else:
            omega = np.array(omega)
            if omega.ndim == 1:
                omega = np.diag(omega)
        
        # Calculate Black-Litterman expected returns
        # μ_BL = [(τΣ)^(-1) + P^T Ω^(-1) P]^(-1) * [(τΣ)^(-1) Pi + P^T Ω^(-1) Q]
        
        # Inverse of tau * covariance matrix
        tau_cov_inv = np.linalg.inv(tau * self.cov_matrix)
        
        # Inverse of omega
        omega_inv = np.linalg.inv(omega)
        
        # Combine pieces
        bl_cov_inv = tau_cov_inv + np.dot(np.dot(P.T, omega_inv), P)
        bl_cov = np.linalg.inv(bl_cov_inv)
        
        bl_returns = np.dot(
            bl_cov,
            np.dot(tau_cov_inv, equilibrium_returns) + np.dot(np.dot(P.T, omega_inv), Q)
        )
        
        # Optimize portfolio using BL expected returns
        def negative_sharpe_bl(weights):
            # Use BL returns instead of mean returns
            portfolio_ret = np.dot(weights, bl_returns)
            portfolio_vol = self.portfolio_volatility(weights, self.cov_matrix)
            
            if portfolio_vol == 0:
                return 1e10
            
            return -portfolio_ret / portfolio_vol
        
        # Constraints
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        
        # Bounds: weights between 0 and 1 (no shorting)
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        
        # Initial guess: market weights
        initial_weights = market_weights.copy()
        
        # Optimize
        result = minimize(
            negative_sharpe_bl,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if not result.success:
            raise ValueError(f"Black-Litterman optimization failed: {result.message}")
        
        weights = pd.Series(result.x, index=self.assets)
        weights[weights < 0.001] = 0
        weights = weights / weights.sum()
        
        return weights
    
    def monte_carlo_simulation(
        self,
        weights: np.ndarray,
        n_simulations: int = 10000,
        time_horizon: int = 252,
        method: str = 'geometric_brownian'
    ) -> Dict[str, np.ndarray]:
        """
        Perform Monte Carlo simulation for portfolio forecasting.
        
        Simulates future portfolio returns using various stochastic processes
        to generate probability distributions of future values.
        
        Parameters:
        -----------
        weights : np.ndarray
            Portfolio weights
        n_simulations : int
            Number of Monte Carlo simulations to run (default: 10000)
        time_horizon : int
            Number of periods ahead to forecast (default: 252 for 1 year of daily data)
        method : str
            Simulation method: 'geometric_brownian' or 'bootstrap' (default: 'geometric_brownian')
        
        Returns:
        --------
        Dict[str, np.ndarray]
            Dictionary containing:
            - 'simulated_returns': array of final period returns for each simulation
            - 'simulated_values': array of portfolio values at time_horizon
            - 'paths': array of shape (n_simulations, time_horizon) with return paths
            - 'cumulative_returns': array of cumulative returns for each path
        """
        # Portfolio expected return and volatility (already annualized)
        portfolio_return_annual = self.portfolio_return(weights, self.mean_returns)
        portfolio_vol_annual = self.portfolio_volatility(weights, self.cov_matrix)
        
        # Annualization factor (assumes returns are daily by default)
        periods_per_year = 252
        
        if method == 'geometric_brownian':
            # Geometric Brownian Motion: dS = μS dt + σS dW
            # For returns: r_t = μ dt + σ dW
            
            # Convert annualized returns to daily for simulation
            # portfolio_return_annual is already annualized, so divide by periods_per_year to get daily
            daily_return = portfolio_return_annual / periods_per_year
            daily_vol = portfolio_vol_annual / np.sqrt(periods_per_year)
            
            # For GBM, we'll use these daily values directly
            annual_return = portfolio_return_annual  # Keep annual for reference
            annual_vol = portfolio_vol_annual  # Keep annual for reference
            
            # Time step (assuming daily data)
            dt = 1.0 / periods_per_year
            
            # Initialize arrays
            paths = np.zeros((n_simulations, time_horizon))
            
            # Generate random shocks
            random_shocks = np.random.normal(0, 1, (n_simulations, time_horizon))
            
            # Simulate paths
            # dt is already 1/252 (daily time step)
            dt = 1.0 / periods_per_year
            for t in range(time_horizon):
                # GBM: dr = μ dt + σ sqrt(dt) dW
                # Using daily_return and daily_vol since we're simulating daily returns
                paths[:, t] = daily_return * 1.0 + daily_vol * random_shocks[:, t]
        
        elif method == 'bootstrap':
            # Bootstrap method: resample historical returns
            if len(self.returns_df) < time_horizon:
                raise ValueError(f"Need at least {time_horizon} historical periods for bootstrap")
            
            # Calculate portfolio returns from historical data
            portfolio_returns_hist = (self.returns_df.values @ weights).flatten()
            
            # Resample random blocks
            paths = np.zeros((n_simulations, time_horizon))
            for sim in range(n_simulations):
                # Randomly sample returns with replacement
                sampled_indices = np.random.choice(len(portfolio_returns_hist), size=time_horizon)
                paths[sim, :] = portfolio_returns_hist[sampled_indices]
        
        else:
            raise ValueError(f"Unknown method: {method}. Use 'geometric_brownian' or 'bootstrap'")
        
        # Calculate cumulative returns for each path
        cumulative_returns = np.cumprod(1 + paths, axis=1) - 1
        
        # Final period returns
        simulated_returns = cumulative_returns[:, -1]
        
        # Portfolio values (assuming starting value of 1)
        simulated_values = 1 + simulated_returns
        
        return {
            'simulated_returns': simulated_returns,
            'simulated_values': simulated_values,
            'paths': paths,
            'cumulative_returns': cumulative_returns,
            'mean_return': portfolio_return_annual,
            'volatility': portfolio_vol_annual,
            'time_horizon': time_horizon,
            'n_simulations': n_simulations
        }