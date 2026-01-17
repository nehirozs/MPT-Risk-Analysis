"""
Backtesting engine for portfolio strategies.

This module provides a comprehensive backtesting framework that simulates
trading strategies over historical data and calculates performance metrics.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple

# Support both relative and absolute imports
try:
    from .strategies import Strategy
    from .risk_metrics import (
        calculate_risk_metrics,
        maximum_drawdown,
        downside_deviation,
        sortino_ratio
    )
except ImportError:
    from strategies import Strategy
    from risk_metrics import (
        calculate_risk_metrics,
        maximum_drawdown,
        downside_deviation,
        sortino_ratio
    )


class Backtester:
    """Backtesting engine for portfolio strategies."""
    
    def __init__(
        self,
        returns_df_or_strategy,
        returns_df: Optional[pd.DataFrame] = None,
        initial_capital: float = 100000,
        rebalance_frequency: str = 'M',  # 'D', 'W', 'M', 'Q', 'Y'
        transaction_cost: float = 0.0,  # Percentage (e.g., 0.001 for 0.1%)
        strategy: Optional[Strategy] = None
    ):
        """
        Initialize Backtester.
        
        Can be called in two ways:
        1. Backtester(returns_df, initial_capital=..., ...) - original style
        2. Backtester(strategy, returns_df, initial_capital=..., ...) - notebook style
        
        Parameters:
        -----------
        returns_df_or_strategy : pd.DataFrame or Strategy
            DataFrame with asset returns OR Strategy instance (if Strategy, returns_df must be provided as keyword)
        returns_df : pd.DataFrame, optional
            DataFrame with asset returns (required if first arg is Strategy)
        initial_capital : float
            Initial portfolio capital (default: 100000)
        rebalance_frequency : str
            Rebalancing frequency: 'D', 'W', 'M', 'Q', 'Y' (default: 'M')
        transaction_cost : float
            Transaction cost as percentage (default: 0.0)
        strategy : Strategy, optional
            Trading strategy (alternative to passing as first arg)
        """
        # Handle two calling conventions
        if isinstance(returns_df_or_strategy, Strategy):
            # Notebook style: Backtester(strategy, returns_df, ...)
            self.strategy = returns_df_or_strategy
            if returns_df is None:
                raise ValueError("If first argument is Strategy, returns_df must be provided as keyword argument")
            self.returns_df = returns_df
        elif isinstance(returns_df_or_strategy, pd.DataFrame):
            # Original style: Backtester(returns_df, ...)
            self.returns_df = returns_df_or_strategy
            self.strategy = strategy  # May be None
        else:
            raise TypeError(f"First argument must be pd.DataFrame or Strategy, got {type(returns_df_or_strategy)}")
        
        self.initial_capital = initial_capital
        self.rebalance_frequency = rebalance_frequency
        self.transaction_cost = transaction_cost
        
        # Create rebalancing dates
        self.rebalance_dates = self._get_rebalance_dates()
    
    def _get_rebalance_dates(self) -> pd.DatetimeIndex:
        """
        Generate rebalancing dates based on frequency.
        Handles both timezone-aware and timezone-naive DatetimeIndex.
        """
        dates = self.returns_df.index
        
        # Check if dates are timezone-aware
        is_tz_aware = dates.tz is not None
        
        if self.rebalance_frequency == 'D':
            return dates
        elif self.rebalance_frequency == 'W':
            # Weekly rebalancing (end of week)
            week_ends = dates.to_period('W').to_timestamp('W')
            rebalance_dates_list = []
            for week_end in week_ends.unique():
                # Make week_end timezone-aware if needed
                if is_tz_aware and week_end.tz is None:
                    week_end = week_end.tz_localize(dates.tz)
                elif not is_tz_aware and week_end.tz is not None:
                    week_end = week_end.tz_localize(None)
                
                week_mask = (dates >= week_end - pd.Timedelta(days=6)) & (dates <= week_end)
                if week_mask.any():
                    rebalance_dates_list.append(dates[week_mask][-1])
            return pd.DatetimeIndex(rebalance_dates_list)
        elif self.rebalance_frequency == 'M':
            # Monthly rebalancing (end of month)
            month_ends = dates.to_period('M').to_timestamp('M')
            rebalance_dates_list = []
            for month_end in month_ends.unique():
                # Create the start of month date
                start_of_month = month_end.replace(day=1)
                
                # Make both dates timezone-aware if needed
                if is_tz_aware:
                    if month_end.tz is None:
                        month_end = month_end.tz_localize(dates.tz)
                    if start_of_month.tz is None:
                        start_of_month = start_of_month.tz_localize(dates.tz)
                else:
                    # Ensure both are timezone-naive
                    if month_end.tz is not None:
                        month_end = month_end.tz_localize(None)
                    if start_of_month.tz is not None:
                        start_of_month = start_of_month.tz_localize(None)
                
                # Find dates in this month
                month_mask = (dates >= start_of_month) & (dates <= month_end)
                if month_mask.any():
                    rebalance_dates_list.append(dates[month_mask][-1])
            return pd.DatetimeIndex(rebalance_dates_list)
        elif self.rebalance_frequency == 'Q':
            # Quarterly rebalancing (end of quarter)
            quarter_ends = dates.to_period('Q').to_timestamp('Q')
            rebalance_dates_list = []
            for quarter_end in quarter_ends.unique():
                # Create the start of quarter date
                start_of_quarter = quarter_end.replace(day=1, month=((quarter_end.month-1)//3)*3+1)
                
                # Make both dates timezone-aware if needed
                if is_tz_aware:
                    if quarter_end.tz is None:
                        quarter_end = quarter_end.tz_localize(dates.tz)
                    if start_of_quarter.tz is None:
                        start_of_quarter = start_of_quarter.tz_localize(dates.tz)
                else:
                    # Ensure both are timezone-naive
                    if quarter_end.tz is not None:
                        quarter_end = quarter_end.tz_localize(None)
                    if start_of_quarter.tz is not None:
                        start_of_quarter = start_of_quarter.tz_localize(None)
                
                quarter_mask = (dates >= start_of_quarter) & (dates <= quarter_end)
                if quarter_mask.any():
                    rebalance_dates_list.append(dates[quarter_mask][-1])
            return pd.DatetimeIndex(rebalance_dates_list)
        elif self.rebalance_frequency == 'Y':
            # Yearly rebalancing (end of year)
            year_ends = dates.to_period('Y').to_timestamp('Y')
            rebalance_dates_list = []
            for year_end in year_ends.unique():
                # Create the start of year date
                start_of_year = year_end.replace(month=1, day=1)
                
                # Make both dates timezone-aware if needed
                if is_tz_aware:
                    if year_end.tz is None:
                        year_end = year_end.tz_localize(dates.tz)
                    if start_of_year.tz is None:
                        start_of_year = start_of_year.tz_localize(dates.tz)
                else:
                    # Ensure both are timezone-naive
                    if year_end.tz is not None:
                        year_end = year_end.tz_localize(None)
                    if start_of_year.tz is not None:
                        start_of_year = start_of_year.tz_localize(None)
                
                year_mask = (dates >= start_of_year) & (dates <= year_end)
                if year_mask.any():
                    rebalance_dates_list.append(dates[year_mask][-1])
            return pd.DatetimeIndex(rebalance_dates_list)
        else:
            raise ValueError(f"Invalid rebalance frequency: {self.rebalance_frequency}")
    
    def run(
        self,
        strategy: Optional[Strategy] = None,
        market_returns: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Run backtest for a given strategy.
        
        Parameters:
        -----------
        strategy : Strategy, optional
            Trading strategy to backtest (if None, uses strategy from __init__)
        market_returns : pd.Series, optional
            Market benchmark returns for comparison
        
        Returns:
        --------
        pd.DataFrame
            Backtest results with columns:
            - date: Date
            - portfolio_value: Portfolio value over time
            - returns: Portfolio returns
            - weights: Portfolio weights at each rebalance
            - trades: Number of trades (positions changed)
        """
        # Use strategy from parameter or from __init__
        if strategy is None:
            if self.strategy is None:
                raise ValueError("Strategy must be provided either in __init__ or as parameter to run()")
            strategy = self.strategy
        
        # Initialize tracking variables
        portfolio_value = [self.initial_capital]
        portfolio_returns = [0.0]
        dates = [self.returns_df.index[0]]
        weights_history = []
        current_weights = None
        
        # Filter rebalance dates to those within data range
        valid_rebalance_dates = [
            d for d in self.rebalance_dates 
            if d >= self.returns_df.index[0] and d <= self.returns_df.index[-1]
        ]
        
        if not valid_rebalance_dates:
            # If no rebalance dates, use first date
            valid_rebalance_dates = [self.returns_df.index[0]]
        
        # Normalize timezones for consistent comparison
        # Convert to naive for comparison if there's a mismatch
        returns_tz = self.returns_df.index.tz
        rebalance_tz = valid_rebalance_dates[0].tz if valid_rebalance_dates else None
        
        if returns_tz != rebalance_tz:
            # Normalize both to naive for comparison
            valid_rebalance_dates_set = {d.tz_localize(None) if d.tz else d for d in valid_rebalance_dates}
            normalize_current_date = lambda d: d.tz_localize(None) if d.tz else d
        else:
            valid_rebalance_dates_set = set(valid_rebalance_dates)
            normalize_current_date = lambda d: d
        
        # Iterate through trading days
        prev_date = self.returns_df.index[0]
        
        for current_date in self.returns_df.index[1:]:
            # Check if it's a rebalancing date (using normalized comparison)
            current_date_normalized = normalize_current_date(current_date)
            if current_date_normalized in valid_rebalance_dates_set:
                # Get new weights from strategy
                new_weights = strategy.get_weights(
                    self.returns_df,
                    current_date
                )
                
                # Calculate transaction cost
                if current_weights is not None:
                    # Calculate weight changes
                    weight_changes = np.abs(new_weights - current_weights).sum()
                    cost = portfolio_value[-1] * weight_changes * self.transaction_cost
                    portfolio_value[-1] -= cost
                
                current_weights = new_weights
            
            # If no weights set yet, use equal weights
            if current_weights is None:
                n_assets = len(self.returns_df.columns)
                current_weights = pd.Series(
                    [1.0 / n_assets] * n_assets,
                    index=self.returns_df.columns
                )
            
            # Calculate portfolio return for this period
            period_returns = self.returns_df.loc[current_date]
            # Ensure indices align for dot product
            aligned_weights = current_weights.reindex(period_returns.index, fill_value=0.0)
            # Handle NaN values in returns - use 0 for NaN returns, but adjust weights accordingly
            valid_mask = ~period_returns.isna()
            if valid_mask.any():
                # If some returns are valid, calculate portfolio return using only valid assets
                # Renormalize weights for valid assets
                valid_weights = aligned_weights[valid_mask]
                valid_weights_sum = valid_weights.sum()
                if valid_weights_sum > 0:
                    valid_weights = valid_weights / valid_weights_sum
                    portfolio_return = (valid_weights * period_returns[valid_mask]).sum()
                else:
                    portfolio_return = 0.0
            else:
                # All returns are NaN - use 0
                portfolio_return = 0.0
            
            # Additional check for NaN/inf values
            if pd.isna(portfolio_return) or np.isinf(portfolio_return):
                portfolio_return = 0.0
            
            # Update portfolio value
            new_value = portfolio_value[-1] * (1 + portfolio_return)
            portfolio_value.append(new_value)
            portfolio_returns.append(portfolio_return)
            dates.append(current_date)
            
            # Store weights at rebalancing
            if current_date_normalized in valid_rebalance_dates_set:
                weights_history.append({
                    'date': current_date,
                    'weights': current_weights.copy()
                })
        
        # Create results DataFrame
        results = pd.DataFrame({
            'date': dates,
            'portfolio_value': portfolio_value,
            'returns': portfolio_returns
        })
        results.set_index('date', inplace=True)
        
        # Store weights history
        self.weights_history = pd.DataFrame(weights_history)
        
        # Calculate performance metrics
        self.performance_metrics = self._calculate_performance_metrics(
            results['returns'],
            results['portfolio_value'],
            market_returns
        )
        
        # Return dictionary with results DataFrame and performance metrics for notebook compatibility
        return {
            'portfolio_values': results['portfolio_value'],
            'portfolio_value': results['portfolio_value'],  # Alias for compatibility
            'returns': results['returns'],
            'performance_metrics': self.performance_metrics.to_dict(),
            'data': results  # Full DataFrame for advanced usage
        }
    
    def _calculate_performance_metrics(
        self,
        returns: pd.Series,
        portfolio_values: pd.Series,
        market_returns: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Calculate comprehensive performance metrics.
        
        Parameters:
        -----------
        returns : pd.Series
            Portfolio returns
        portfolio_values : pd.Series
            Portfolio values over time
        market_returns : pd.Series, optional
            Market benchmark returns
        
        Returns:
        --------
        pd.Series
            Performance metrics
        """
        metrics = {}
        
        # Filter out NaN values for calculations
        valid_returns = returns.dropna()
        if len(valid_returns) == 0:
            # If no valid returns, return zero metrics
            metrics['Total_Return'] = 0.0
            metrics['Annualized_Return'] = 0.0
            metrics['Annualized_Volatility'] = 0.0
            metrics['Sharpe_Ratio'] = 0.0
            metrics['Max_Drawdown'] = 0.0
            metrics['Calmar_Ratio'] = 0.0
            metrics['Sortino_Ratio'] = 0.0
            metrics['Win_Rate'] = 0.0
            # Add final value even when no returns
            if len(portfolio_values) > 0:
                final_value = portfolio_values.iloc[-1]
            else:
                final_value = self.initial_capital
            metrics['final_value'] = final_value
            metrics['final_portfolio_value'] = final_value
            # Add lowercase aliases
            metrics['total_return'] = 0.0
            metrics['annualized_return'] = 0.0
            metrics['annualized_volatility'] = 0.0
            metrics['sharpe_ratio'] = 0.0
            metrics['max_drawdown'] = 0.0
            metrics['calmar_ratio'] = 0.0
            metrics['win_rate'] = 0.0
            return pd.Series(metrics)
        
        # Total return
        cumulative_return = (1 + valid_returns).prod() - 1
        metrics['Total_Return'] = cumulative_return
        
        # Annualized return
        periods_per_year = self._get_periods_per_year(returns)
        n_periods = len(valid_returns)
        if n_periods > 0:
            annualized_return = (1 + cumulative_return) ** (periods_per_year / n_periods) - 1
        else:
            annualized_return = 0.0
        metrics['Annualized_Return'] = annualized_return
        
        # Annualized volatility
        annualized_vol = valid_returns.std() * np.sqrt(periods_per_year)
        if pd.isna(annualized_vol):
            annualized_vol = 0.0
        metrics['Annualized_Volatility'] = annualized_vol
        
        # Sharpe ratio (assuming risk-free rate = 0)
        if annualized_vol > 0:
            metrics['Sharpe_Ratio'] = annualized_return / annualized_vol
        else:
            metrics['Sharpe_Ratio'] = np.inf if annualized_return > 0 else -np.inf
        
        # Maximum drawdown
        metrics['Max_Drawdown'] = maximum_drawdown(returns)
        
        # Calmar ratio
        if abs(metrics['Max_Drawdown']) > 0:
            metrics['Calmar_Ratio'] = annualized_return / abs(metrics['Max_Drawdown'])
        else:
            metrics['Calmar_Ratio'] = np.inf if annualized_return > 0 else -np.inf
        
        # Sortino ratio
        metrics['Sortino_Ratio'] = sortino_ratio(returns, annualized=True)
        
        # Win rate
        positive_returns = (valid_returns > 0).sum()
        total_returns = len(valid_returns[valid_returns != 0])  # Exclude zeros
        if total_returns > 0:
            metrics['Win_Rate'] = positive_returns / total_returns
        else:
            metrics['Win_Rate'] = 0.0
        
        # Add risk metrics
        risk_metrics = calculate_risk_metrics(valid_returns, market_returns)
        metrics.update(risk_metrics.to_dict())
        
        # Calculate final portfolio value
        if len(portfolio_values) > 0:
            final_value = portfolio_values.iloc[-1]
        else:
            final_value = self.initial_capital
        metrics['final_value'] = final_value
        metrics['final_portfolio_value'] = final_value  # Alias
        
        # Add lowercase aliases for notebook compatibility
        metrics['total_return'] = metrics.get('Total_Return', cumulative_return)
        metrics['annualized_return'] = metrics.get('Annualized_Return', annualized_return)
        metrics['annualized_volatility'] = annualized_vol
        metrics['sharpe_ratio'] = metrics.get('Sharpe_Ratio', metrics.get('Sharpe_Ratio', 0.0))
        metrics['max_drawdown'] = metrics.get('Max_Drawdown', metrics.get('Max_Drawdown', 0.0))
        metrics['calmar_ratio'] = metrics.get('Calmar_Ratio', metrics.get('Calmar_Ratio', 0.0))
        metrics['win_rate'] = metrics['Win_Rate']
        
        return pd.Series(metrics)
    
    def _get_periods_per_year(self, returns: pd.Series) -> int:
        """Estimate periods per year from returns frequency."""
        if len(returns) < 2:
            return 252
        
        time_deltas = returns.index.to_series().diff().dropna()
        avg_delta = time_deltas.median()
        
        if pd.Timedelta(days=1) * 0.9 <= avg_delta <= pd.Timedelta(days=1) * 1.1:
            return 252  # Daily
        elif pd.Timedelta(weeks=1) * 0.9 <= avg_delta <= pd.Timedelta(weeks=1) * 1.1:
            return 52  # Weekly
        elif pd.Timedelta(days=30) * 0.9 <= avg_delta <= pd.Timedelta(days=30) * 1.1:
            return 12  # Monthly
        else:
            return 252


def compare_strategies(
    returns_df: pd.DataFrame,
    strategies: Dict[str, Strategy],
    initial_capital: float = 100000,
    rebalance_frequency: str = 'M',
    market_returns: Optional[pd.Series] = None
) -> pd.DataFrame:
    """
    Compare multiple strategies through backtesting.
    
    Parameters:
    -----------
    returns_df : pd.DataFrame
        DataFrame with asset returns
    strategies : Dict[str, Strategy]
        Dictionary mapping strategy names to Strategy instances
    initial_capital : float
        Initial portfolio capital
    rebalance_frequency : str
        Rebalancing frequency
    market_returns : pd.Series, optional
        Market benchmark returns
    
    Returns:
    --------
    pd.DataFrame
        Comparison of strategy results with columns for each strategy
    """
    results = {}
    
    for strategy_name, strategy in strategies.items():
        backtester = Backtester(
            returns_df,
            initial_capital,
            rebalance_frequency
        )
        backtest_results = backtester.run(strategy, market_returns)
        results[strategy_name] = backtest_results['portfolio_value']
    
    comparison_df = pd.DataFrame(results)
    return comparison_df
