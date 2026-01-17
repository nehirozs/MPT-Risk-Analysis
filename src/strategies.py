"""
Trading strategy implementations for portfolio management.

This module contains various portfolio allocation strategies including
Buy & Hold, Momentum, Mean Reversion, and Minimum Variance.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional

# Support both relative and absolute imports
try:
    from .portfolio_optimizer import PortfolioOptimizer
except ImportError:
    from portfolio_optimizer import PortfolioOptimizer


class Strategy:
    """Base class for trading strategies."""
    
    def __init__(self, name: str):
        """
        Initialize strategy.
        
        Parameters:
        -----------
        name : str
            Strategy name
        """
        self.name = name
    
    def get_weights(
        self,
        returns_df: pd.DataFrame,
        current_date: pd.Timestamp,
        lookback_window: Optional[int] = None
    ) -> pd.Series:
        """
        Calculate portfolio weights for given date.
        
        Parameters:
        -----------
        returns_df : pd.DataFrame
            Historical returns DataFrame
        current_date : pd.Timestamp
            Current date for weight calculation
        lookback_window : int, optional
            Number of periods to look back
        
        Returns:
        --------
        pd.Series
            Portfolio weights (should sum to 1)
        """
        raise NotImplementedError("Subclasses must implement get_weights method")


class BuyAndHold(Strategy):
    """Equal-weighted buy-and-hold strategy with periodic rebalancing."""
    
    def __init__(self):
        """Initialize Buy & Hold strategy."""
        super().__init__("Buy & Hold")
    
    def get_weights(
        self,
        returns_df: pd.DataFrame,
        current_date: pd.Timestamp = None,
        lookback_window: Optional[int] = None
    ) -> pd.Series:
        """
        Calculate equal weights for all assets.
        
        Parameters:
        -----------
        returns_df : pd.DataFrame
            Historical returns DataFrame
        current_date : pd.Timestamp
            Not used for this strategy
        lookback_window : int, optional
            Not used for this strategy
        
        Returns:
        --------
        pd.Series
            Equal weights for all assets
        """
        n_assets = len(returns_df.columns)
        equal_weight = 1.0 / n_assets
        return pd.Series([equal_weight] * n_assets, index=returns_df.columns)


class Momentum(Strategy):
    """
    Momentum strategy: select top N stocks based on past K-month returns.
    """
    
    def __init__(self, top_n: int = 5, lookback_months: int = 3):
        """
        Initialize Momentum strategy.
        
        Parameters:
        -----------
        top_n : int
            Number of top stocks to select (default: 5)
        lookback_months : int
            Number of months to look back for momentum (default: 3)
        """
        super().__init__("Momentum")
        self.top_n = top_n
        self.lookback_months = lookback_months
    
    def get_weights(
        self,
        returns_df: pd.DataFrame,
        current_date: pd.Timestamp,
        lookback_window: Optional[int] = None
    ) -> pd.Series:
        """
        Calculate weights based on momentum.
        
        Parameters:
        -----------
        returns_df : pd.DataFrame
            Historical returns DataFrame
        current_date : pd.Timestamp
            Current date for weight calculation
        lookback_window : int, optional
            Number of periods to look back (overrides lookback_months if provided)
        
        Returns:
        --------
        pd.Series
            Weights (equal-weighted for top N stocks)
        """
        # Use lookback_window if provided, otherwise use lookback_months
        if lookback_window is None:
            # For monthly data, use lookback_months
            # For daily data, approximate (assuming ~21 trading days per month)
            if returns_df.index.freq == 'D' or returns_df.index.freq is None:
                lookback_window = self.lookback_months * 21
            else:
                lookback_window = self.lookback_months
        
        # Get data up to (but not including) current_date
        historical_data = returns_df[returns_df.index < current_date]
        
        if len(historical_data) < lookback_window:
            # Not enough data, return equal weights
            n_assets = len(returns_df.columns)
            return pd.Series([1.0 / n_assets] * n_assets, index=returns_df.columns)
        
        # Get recent data
        recent_data = historical_data.tail(lookback_window)
        
        # Calculate cumulative returns over lookback period
        cumulative_returns = (1 + recent_data).prod() - 1
        
        # Select top N stocks
        top_stocks = cumulative_returns.nlargest(min(self.top_n, len(cumulative_returns)))
        
        # Create weights: equal for top N, zero for others
        weights = pd.Series(0.0, index=returns_df.columns)
        if len(top_stocks) > 0:
            equal_weight = 1.0 / len(top_stocks)
            weights[top_stocks.index] = equal_weight
        
        return weights


class MeanReversion(Strategy):
    """
    Mean reversion strategy: inverse volatility weighting.
    Lower volatility stocks get higher weights.
    """
    
    def __init__(self, lookback_months: int = 12):
        """
        Initialize Mean Reversion strategy.
        
        Parameters:
        -----------
        lookback_months : int
            Number of months to look back for volatility calculation (default: 12)
        """
        super().__init__("Mean Reversion")
        self.lookback_months = lookback_months
    
    def get_weights(
        self,
        returns_df: pd.DataFrame,
        current_date: pd.Timestamp,
        lookback_window: Optional[int] = None
    ) -> pd.Series:
        """
        Calculate weights using inverse volatility.
        
        Parameters:
        -----------
        returns_df : pd.DataFrame
            Historical returns DataFrame
        current_date : pd.Timestamp
            Current date for weight calculation
        lookback_window : int, optional
            Number of periods to look back (overrides lookback_months if provided)
        
        Returns:
        --------
        pd.Series
            Inverse volatility weights
        """
        # Use lookback_window if provided, otherwise use lookback_months
        if lookback_window is None:
            if returns_df.index.freq == 'D' or returns_df.index.freq is None:
                lookback_window = self.lookback_months * 21
            else:
                lookback_window = self.lookback_months
        
        # Get data up to (but not including) current_date
        historical_data = returns_df[returns_df.index < current_date]
        
        if len(historical_data) < lookback_window:
            # Not enough data, return equal weights
            n_assets = len(returns_df.columns)
            return pd.Series([1.0 / n_assets] * n_assets, index=returns_df.columns)
        
        # Get recent data
        recent_data = historical_data.tail(lookback_window)
        
        # Calculate volatilities (standard deviations)
        volatilities = recent_data.std()
        
        # Avoid division by zero
        volatilities = volatilities.replace(0, np.inf)
        
        # Inverse volatility weights
        inverse_vol = 1.0 / volatilities
        
        # Normalize to sum to 1
        weights = inverse_vol / inverse_vol.sum()
        
        return weights


class MinimumVariance(Strategy):
    """
    Minimum variance strategy: rebalance using optimization
    to minimize portfolio variance.
    """
    
    def __init__(self, lookback_months: int = 12):
        """
        Initialize Minimum Variance strategy.
        
        Parameters:
        -----------
        lookback_months : int
            Number of months to look back for covariance calculation (default: 12)
        """
        super().__init__("Minimum Variance")
        self.lookback_months = lookback_months
    
    def get_weights(
        self,
        returns_df: pd.DataFrame,
        current_date: pd.Timestamp,
        lookback_window: Optional[int] = None
    ) -> pd.Series:
        """
        Calculate weights using minimum variance optimization.
        
        Parameters:
        -----------
        returns_df : pd.DataFrame
            Historical returns DataFrame
        current_date : pd.Timestamp
            Current date for weight calculation
        lookback_window : int, optional
            Number of periods to look back (overrides lookback_months if provided)
        
        Returns:
        --------
        pd.Series
            Minimum variance optimal weights
        """
        # Use lookback_window if provided, otherwise use lookback_months
        if lookback_window is None:
            if returns_df.index.freq == 'D' or returns_df.index.freq is None:
                lookback_window = self.lookback_months * 21
            else:
                lookback_window = self.lookback_months
        
        # Get data up to (but not including) current_date
        historical_data = returns_df[returns_df.index < current_date]
        
        if len(historical_data) < lookback_window:
            # Not enough data, return equal weights
            n_assets = len(returns_df.columns)
            return pd.Series([1.0 / n_assets] * n_assets, index=returns_df.columns)
        
        # Get recent data
        recent_data = historical_data.tail(lookback_window)
        
        try:
            # Create optimizer with regularization for numerical stability
            optimizer = PortfolioOptimizer(recent_data, regularization=1e-5)
            weights = optimizer.minimum_variance_portfolio()
            return weights
        except Exception as e:
            # If optimization fails, use inverse volatility weighting as intelligent fallback
            # This is better than equal weights as it still considers risk
            try:
                volatilities = recent_data.std()
                # Avoid division by zero
                volatilities = volatilities.replace(0, np.inf)
                # Inverse volatility weights (lower vol = higher weight)
                inverse_vol = 1.0 / volatilities
                weights = inverse_vol / inverse_vol.sum()
                # Only print warning if it's not a silent fallback
                if "Optimization failed" not in str(e):
                    print(f"Optimization failed for {current_date}: {e}. Using inverse volatility weights.")
                return weights
            except Exception:
                # Final fallback: equal weights if inverse vol also fails
                print(f"Optimization failed for {current_date}: {e}. Using equal weights.")
                n_assets = len(returns_df.columns)
                return pd.Series([1.0 / n_assets] * n_assets, index=returns_df.columns)


def create_strategy(strategy_name: str, **kwargs) -> Strategy:
    """
    Factory function to create strategy instances.
    
    Parameters:
    -----------
    strategy_name : str
        Name of strategy ('buy_hold', 'momentum', 'mean_reversion', 'min_variance')
    **kwargs
        Strategy-specific parameters
    
    Returns:
    --------
    Strategy
        Strategy instance
    """
    strategies = {
        'buy_hold': BuyAndHold,
        'momentum': Momentum,
        'mean_reversion': MeanReversion,
        'min_variance': MinimumVariance
    }
    
    strategy_class = strategies.get(strategy_name.lower())
    if strategy_class is None:
        raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(strategies.keys())}")
    
    return strategy_class(**kwargs)
