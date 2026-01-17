"""
Visualization module for portfolio analysis and backtesting results.

This module provides plotting functions for efficient frontier,
performance charts, risk analysis, and strategy comparisons.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, Dict, List

# Support both relative and absolute imports
try:
    from .portfolio_optimizer import PortfolioOptimizer
except ImportError:
    from portfolio_optimizer import PortfolioOptimizer


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def plot_efficient_frontier(
    optimizer: Optional[PortfolioOptimizer] = None,
    frontier: Optional[pd.DataFrame] = None,
    mean_returns: Optional[np.ndarray] = None,
    cov_matrix: Optional[np.ndarray] = None,
    tickers: Optional[List[str]] = None,
    min_var_weights: Optional[pd.Series] = None,
    max_sharpe_weights: Optional[pd.Series] = None,
    risk_free_rate: float = 0.0,
    n_portfolios: int = 50,
    show_individual_assets: bool = True,
    show_min_var: bool = True,
    show_max_sharpe: bool = True,
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot efficient frontier with individual assets and optimal portfolios.
    
    Can be called in two ways:
    1. With optimizer: plot_efficient_frontier(optimizer, ...)
    2. With pre-computed frontier: plot_efficient_frontier(frontier=df, mean_returns=..., ...)
    
    Parameters:
    -----------
    optimizer : PortfolioOptimizer, optional
        PortfolioOptimizer instance (if not provided, must provide frontier and related params)
    frontier : pd.DataFrame, optional
        Pre-computed efficient frontier DataFrame with columns 'Return', 'Volatility', 'Sharpe'
    mean_returns : np.ndarray, optional
        Expected returns array (required if frontier is provided)
    cov_matrix : np.ndarray, optional
        Covariance matrix (required if frontier is provided)
    tickers : List[str], optional
        List of asset tickers (required if frontier is provided)
    min_var_weights : pd.Series, optional
        Minimum variance portfolio weights
    max_sharpe_weights : pd.Series, optional
        Maximum Sharpe ratio portfolio weights
    risk_free_rate : float
        Risk-free rate for Sharpe ratio calculation
    n_portfolios : int
        Number of portfolios to generate along frontier
    show_individual_assets : bool
        Whether to show individual assets on plot
    show_min_var : bool
        Whether to mark minimum variance portfolio
    show_max_sharpe : bool
        Whether to mark maximum Sharpe ratio portfolio
    ax : plt.Axes, optional
        Matplotlib axes to plot on
    
    Returns:
    --------
    plt.Axes
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Handle two calling conventions - check if first positional arg is a DataFrame
    # If optimizer parameter is None but frontier is provided positionally, handle it
    if optimizer is None and frontier is not None:
        # New calling convention: use pre-computed frontier
        frontier_df = frontier
        # Only require mean_returns/cov_matrix/tickers if we need to show individual assets or calculate portfolio positions
        needs_calc = show_individual_assets or (show_min_var and min_var_weights is not None) or (show_max_sharpe and max_sharpe_weights is not None)
        if needs_calc and (mean_returns is None or cov_matrix is None or tickers is None):
            raise ValueError("If showing individual assets or portfolio markers, mean_returns, cov_matrix, and tickers must be provided")
        mean_returns_arr = mean_returns
        cov_matrix_arr = cov_matrix
        tickers_list = tickers if tickers is not None else []
        show_min_var_flag = show_min_var and min_var_weights is not None and mean_returns is not None and cov_matrix is not None
        show_max_sharpe_flag = show_max_sharpe and max_sharpe_weights is not None and mean_returns is not None and cov_matrix is not None
        min_var_w = min_var_weights
        max_sharpe_w = max_sharpe_weights
        optimizer_obj = None
    elif optimizer is not None:
        # Original calling convention: use optimizer (first arg could be optimizer or frontier)
        # Check type to handle if first positional arg is a DataFrame
        if isinstance(optimizer, pd.DataFrame):
            # First positional arg is actually a DataFrame
            frontier_df = optimizer
            # Only require mean_returns/cov_matrix/tickers if we need to show individual assets or calculate portfolio positions
            needs_calc = show_individual_assets or (show_min_var and min_var_weights is not None) or (show_max_sharpe and max_sharpe_weights is not None)
            if needs_calc and (mean_returns is None or cov_matrix is None or tickers is None):
                raise ValueError("If showing individual assets or portfolio markers, mean_returns, cov_matrix, and tickers must be provided")
            mean_returns_arr = mean_returns
            cov_matrix_arr = cov_matrix
            tickers_list = tickers if tickers is not None else []
            show_min_var_flag = show_min_var and min_var_weights is not None and mean_returns is not None and cov_matrix is not None
            show_max_sharpe_flag = show_max_sharpe and max_sharpe_weights is not None and mean_returns is not None and cov_matrix is not None
            min_var_w = min_var_weights
            max_sharpe_w = max_sharpe_weights
            optimizer_obj = None
        else:
            # First positional arg is actually an optimizer
            optimizer_obj = optimizer
            frontier_df = optimizer_obj.efficient_frontier(n_portfolios, risk_free_rate=risk_free_rate)
            mean_returns_arr = optimizer_obj.mean_returns
            cov_matrix_arr = optimizer_obj.cov_matrix
            tickers_list = optimizer_obj.tickers
            show_min_var_flag = show_min_var
            show_max_sharpe_flag = show_max_sharpe
            min_var_w = None
            max_sharpe_w = None
    else:
        raise ValueError("Must provide either optimizer or frontier")
    
    # Plot efficient frontier
    scatter = ax.scatter(
        frontier_df['Volatility'],
        frontier_df['Return'],
        c=frontier_df['Sharpe'],
        cmap='viridis',
        alpha=0.6,
        s=30,
        label='Efficient Frontier'
    )
    
    # Colorbar for Sharpe ratio
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Sharpe Ratio', rotation=270, labelpad=15)
    
    # Plot individual assets
    if show_individual_assets and mean_returns_arr is not None and cov_matrix_arr is not None and len(tickers_list) > 0:
        individual_vols = []
        individual_rets = []
        for i, asset in enumerate(tickers_list):
            # Create unit weight vector for this asset
            unit_weights = np.zeros(len(tickers_list))
            unit_weights[i] = 1.0
            vol = PortfolioOptimizer.portfolio_volatility(unit_weights, cov_matrix_arr)
            ret = mean_returns_arr[i]
            individual_vols.append(vol)
            individual_rets.append(ret)
        
        ax.scatter(
            individual_vols,
            individual_rets,
            marker='x',
            s=100,
            color='red',
            label='Individual Assets',
            zorder=5
        )
        
        # Add asset labels
        for i, asset in enumerate(tickers_list):
            ax.annotate(
                asset,
                (individual_vols[i], individual_rets[i]),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=9
            )
    
    # Mark minimum variance portfolio
    if show_min_var_flag and mean_returns_arr is not None and cov_matrix_arr is not None:
        try:
            if optimizer_obj is not None:
                weights = optimizer_obj.minimum_variance_portfolio()
            else:
                weights = min_var_w
            
            if weights is not None:
                ret = PortfolioOptimizer.portfolio_return(weights.values, mean_returns_arr)
                vol = PortfolioOptimizer.portfolio_volatility(weights.values, cov_matrix_arr)
                ax.scatter(
                    vol,
                    ret,
                    marker='*',
                    s=300,
                    color='gold',
                    label='Min Variance Portfolio',
                    zorder=5,
                    edgecolors='black',
                    linewidths=1
                )
        except:
            pass
    
    # Mark maximum Sharpe ratio portfolio
    if show_max_sharpe_flag and mean_returns_arr is not None and cov_matrix_arr is not None:
        try:
            if optimizer_obj is not None:
                weights = optimizer_obj.maximum_sharpe_ratio_portfolio(risk_free_rate)
            else:
                weights = max_sharpe_w
            
            if weights is not None:
                ret = PortfolioOptimizer.portfolio_return(weights.values, mean_returns_arr)
                vol = PortfolioOptimizer.portfolio_volatility(weights.values, cov_matrix_arr)
                ax.scatter(
                    vol,
                    ret,
                    marker='*',
                    s=300,
                    color='lime',
                    label='Max Sharpe Portfolio',
                    zorder=5,
                    edgecolors='black',
                    linewidths=1
                )
        except:
            pass
    
    ax.set_xlabel('Volatility (Risk)', fontsize=12)
    ax.set_ylabel('Expected Return', fontsize=12)
    ax.set_title('Efficient Frontier', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_portfolio_performance(
    backtest_results: pd.DataFrame,
    benchmark_returns: Optional[pd.Series] = None,
    title: str = 'Portfolio Performance',
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot cumulative portfolio performance over time.
    
    Parameters:
    -----------
    backtest_results : pd.DataFrame
        Backtest results with 'portfolio_value' column
    benchmark_returns : pd.Series, optional
        Benchmark returns for comparison
    title : str
        Plot title
    ax : plt.Axes, optional
        Matplotlib axes to plot on
    
    Returns:
    --------
    plt.Axes
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot portfolio value
    ax.plot(
        backtest_results.index,
        backtest_results['portfolio_value'],
        label='Portfolio',
        linewidth=2
    )
    
    # Plot benchmark if provided
    if benchmark_returns is not None:
        # Align dates
        aligned = pd.DataFrame({
            'portfolio': backtest_results['returns'],
            'benchmark': benchmark_returns
        }).dropna()
        
        if len(aligned) > 0:
            benchmark_values = (1 + aligned['benchmark']).cumprod() * backtest_results['portfolio_value'].iloc[0]
            ax.plot(
                aligned.index,
                benchmark_values,
                label='Benchmark',
                linewidth=2,
                linestyle='--'
            )
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    return ax


def plot_drawdown(
    backtest_results: pd.DataFrame,
    title: str = 'Portfolio Drawdown',
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot drawdown chart for portfolio.
    
    Parameters:
    -----------
    backtest_results : pd.DataFrame
        Backtest results with 'returns' column
    title : str
        Plot title
    ax : plt.Axes, optional
        Matplotlib axes to plot on
    
    Returns:
    --------
    plt.Axes
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    # Calculate cumulative returns
    cumulative = (1 + backtest_results['returns']).cumprod()
    
    # Calculate running maximum
    running_max = cumulative.expanding().max()
    
    # Calculate drawdown
    drawdown = (cumulative - running_max) / running_max * 100  # Percentage
    
    # Plot drawdown
    ax.fill_between(
        drawdown.index,
        drawdown,
        0,
        color='red',
        alpha=0.3,
        label='Drawdown'
    )
    ax.plot(drawdown.index, drawdown, color='red', linewidth=1)
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Drawdown (%)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    return ax


def plot_strategy_comparison(
    comparison_df: pd.DataFrame,
    title: str = 'Strategy Comparison',
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot comparison of multiple strategies.
    
    Parameters:
    -----------
    comparison_df : pd.DataFrame
        DataFrame with portfolio values for each strategy (columns are strategies)
    title : str
        Plot title
    ax : plt.Axes, optional
        Matplotlib axes to plot on
    
    Returns:
    --------
    plt.Axes
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    # Normalize to starting value
    for col in comparison_df.columns:
        normalized = comparison_df[col] / comparison_df[col].iloc[0] * 100
        ax.plot(comparison_df.index, normalized, label=col, linewidth=2)
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Normalized Value (Starting = 100)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    return ax


def plot_correlation_heatmap(
    returns_df: pd.DataFrame,
    title: str = 'Asset Correlation Matrix',
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot correlation heatmap of asset returns.
    
    Parameters:
    -----------
    returns_df : pd.DataFrame
        DataFrame with asset returns
    title : str
        Plot title
    ax : plt.Axes, optional
        Matplotlib axes to plot on
    
    Returns:
    --------
    plt.Axes
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Calculate correlation matrix
    corr_matrix = returns_df.corr()
    
    # Plot heatmap
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax
    )
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    return ax


def plot_return_distribution(
    returns: pd.Series,
    title: str = 'Return Distribution',
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot histogram of return distribution.
    
    Parameters:
    -----------
    returns : pd.Series
        Portfolio returns
    title : str
        Plot title
    ax : plt.Axes, optional
        Matplotlib axes to plot on
    
    Returns:
    --------
    plt.Axes
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot histogram
    ax.hist(returns * 100, bins=50, alpha=0.7, edgecolor='black')
    
    # Add vertical line at mean
    mean_return = returns.mean() * 100
    ax.axvline(mean_return, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_return:.2f}%')
    
    ax.set_xlabel('Return (%)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_rolling_sharpe(
    backtest_results: pd.DataFrame,
    window: int = 60,
    risk_free_rate: float = 0.0,
    title: str = 'Rolling Sharpe Ratio',
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot rolling Sharpe ratio over time.
    
    Parameters:
    -----------
    backtest_results : pd.DataFrame
        Backtest results with 'returns' column
    window : int
        Rolling window size (default: 60)
    risk_free_rate : float
        Risk-free rate (default: 0.0)
    title : str
        Plot title
    ax : plt.Axes, optional
        Matplotlib axes to plot on
    
    Returns:
    --------
    plt.Axes
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    # Calculate rolling Sharpe ratio
    rolling_mean = backtest_results['returns'].rolling(window=window).mean()
    rolling_std = backtest_results['returns'].rolling(window=window).std()
    rolling_sharpe = (rolling_mean - risk_free_rate / 252) / rolling_std * np.sqrt(252)
    
    ax.plot(rolling_sharpe.index, rolling_sharpe, linewidth=2)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.axhline(y=1, color='green', linestyle='--', linewidth=1, label='Sharpe = 1')
    ax.axhline(y=2, color='blue', linestyle='--', linewidth=1, label='Sharpe = 2')
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Rolling Sharpe Ratio', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    return ax


def plot_performance_metrics(
    metrics_dict: Dict[str, pd.Series],
    title: str = 'Performance Metrics Comparison',
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot bar chart comparing performance metrics across strategies.
    
    Parameters:
    -----------
    metrics_dict : Dict[str, pd.Series]
        Dictionary mapping strategy names to performance metrics Series
    title : str
        Plot title
    ax : plt.Axes, optional
        Matplotlib axes to plot on
    
    Returns:
    --------
    plt.Axes
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create DataFrame from metrics
    metrics_df = pd.DataFrame(metrics_dict).T
    
    # Select key metrics to plot
    key_metrics = ['Annualized_Return', 'Sharpe_Ratio', 'Max_Drawdown', 'Win_Rate']
    available_metrics = [m for m in key_metrics if m in metrics_df.columns]
    
    if not available_metrics:
        raise ValueError("No matching metrics found in metrics_dict")
    
    # Plot grouped bar chart
    x = np.arange(len(metrics_df.index))
    width = 0.8 / len(available_metrics)
    
    for i, metric in enumerate(available_metrics):
        offset = (i - len(available_metrics) / 2 + 0.5) * width
        ax.bar(x + offset, metrics_df[metric], width, label=metric)
    
    ax.set_xlabel('Strategy', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_df.index, rotation=45, ha='right')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, axis='y')
    
    return ax


def plot_cumulative_returns_vs_benchmark(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    title: str = 'Cumulative Returns vs Benchmark',
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot cumulative returns for portfolio vs benchmark.
    
    Parameters:
    -----------
    portfolio_returns : pd.Series
        Portfolio returns
    benchmark_returns : pd.Series
        Benchmark returns (e.g., S&P 500)
    title : str
        Plot title
    ax : plt.Axes, optional
        Matplotlib axes to plot on
    
    Returns:
    --------
    plt.Axes
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    # Align dates
    aligned = pd.DataFrame({
        'portfolio': portfolio_returns,
        'benchmark': benchmark_returns
    }).dropna()
    
    # Calculate cumulative returns
    portfolio_cum = (1 + aligned['portfolio']).cumprod()
    benchmark_cum = (1 + aligned['benchmark']).cumprod()
    
    # Plot
    ax.plot(aligned.index, portfolio_cum, label='Portfolio', linewidth=2)
    ax.plot(aligned.index, benchmark_cum, label='Benchmark (S&P 500)', 
            linewidth=2, linestyle='--', alpha=0.8)
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cumulative Return', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    return ax


def plot_portfolio_composition(
    weights_history: pd.DataFrame,
    title: str = 'Portfolio Composition Over Time',
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot portfolio composition over time as stacked area chart.
    
    Parameters:
    -----------
    weights_history : pd.DataFrame
        DataFrame with 'date' column and weight columns for each asset
    title : str
        Plot title
    ax : plt.Axes, optional
        Matplotlib axes to plot on
    
    Returns:
    --------
    plt.Axes
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 8))
    
    # Extract dates and weights
    if 'date' in weights_history.columns:
        dates = pd.to_datetime(weights_history['date'])
        weight_data = weights_history.drop('date', axis=1)
    else:
        dates = weights_history.index
        if 'weights' in weights_history.columns:
            # If weights are in a single column as dict/Series
            weights_list = weights_history['weights'].tolist()
            if len(weights_list) > 0 and isinstance(weights_list[0], pd.Series):
                weight_data = pd.DataFrame(weights_list, index=dates)
            elif len(weights_list) > 0 and isinstance(weights_list[0], dict):
                weight_data = pd.DataFrame(weights_list, index=dates)
            else:
                weight_data = weights_history
        else:
            weight_data = weights_history
    
    weight_cols = [col for col in weight_data.columns if col != 'date']
    
    if len(weight_cols) > 0:
        # Plot stacked area chart
        ax.stackplot(
            dates,
            *[weight_data[col] * 100 for col in weight_cols],
            labels=weight_cols,
            alpha=0.7
        )
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Portfolio Weight (%)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    plt.xticks(rotation=45)
    
    return ax


def plot_drawdown_duration(
    returns: pd.Series,
    title: str = 'Drawdown Duration Analysis',
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot drawdown duration analysis showing time to recovery.
    
    Parameters:
    -----------
    returns : pd.Series
        Portfolio returns
    title : str
        Plot title
    ax : plt.Axes, optional
        Matplotlib axes to plot on
    
    Returns:
    --------
    plt.Axes
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    # Calculate cumulative returns
    cumulative = (1 + returns).cumprod()
    
    # Calculate running maximum
    running_max = cumulative.expanding().max()
    
    # Calculate drawdown
    drawdown = (cumulative - running_max) / running_max * 100
    
    # Plot drawdown
    ax.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
    ax.plot(drawdown.index, drawdown, color='red', linewidth=1)
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Drawdown (%)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.xticks(rotation=45)
    
    return ax


def plot_rolling_volatility(
    returns: pd.Series,
    window: int = 252,
    benchmark_returns: Optional[pd.Series] = None,
    title: str = 'Rolling Volatility',
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot rolling volatility over time.
    
    Parameters:
    -----------
    returns : pd.Series
        Portfolio returns
    window : int
        Rolling window size (default: 252 for daily annualized)
    benchmark_returns : pd.Series, optional
        Benchmark returns for comparison
    title : str
        Plot title
    ax : plt.Axes, optional
        Matplotlib axes to plot on
    
    Returns:
    --------
    plt.Axes
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    # Estimate periods per year
    if len(returns) < 2:
        periods_per_year = 252
    else:
        time_deltas = returns.index.to_series().diff().dropna()
        avg_delta = time_deltas.median()
        if pd.Timedelta(days=1) * 0.9 <= avg_delta <= pd.Timedelta(days=1) * 1.1:
            periods_per_year = 252
        elif pd.Timedelta(weeks=1) * 0.9 <= avg_delta <= pd.Timedelta(weeks=1) * 1.1:
            periods_per_year = 52
        elif pd.Timedelta(days=30) * 0.9 <= avg_delta <= pd.Timedelta(days=30) * 1.1:
            periods_per_year = 12
        else:
            periods_per_year = 252
    
    # Calculate rolling volatility
    rolling_vol = returns.rolling(window=window).std() * np.sqrt(periods_per_year)
    
    # Plot portfolio volatility
    ax.plot(rolling_vol.index, rolling_vol * 100, label='Portfolio', 
            linewidth=2, color='blue')
    
    # Plot benchmark volatility if provided
    if benchmark_returns is not None:
        # Align dates
        aligned = pd.DataFrame({
            'portfolio': returns,
            'benchmark': benchmark_returns
        }).dropna()
        
        benchmark_rolling_vol = aligned['benchmark'].rolling(window=window).std() * np.sqrt(periods_per_year)
        ax.plot(benchmark_rolling_vol.index, benchmark_rolling_vol * 100, 
                label='Benchmark', linewidth=2, linestyle='--', color='orange')
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Annualized Volatility (%)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    return ax
