"""
Risk metrics module for portfolio risk analysis.

This module provides various risk measures including VaR, CVaR,
Beta, and other downside risk metrics.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict
from scipy import stats


def historical_var(
    returns: pd.Series,
    confidence_level: float = 0.95
) -> float:
    """
    Calculate Historical Value at Risk (VaR).
    
    VaR is the maximum loss expected over a given time period
    at a given confidence level.
    
    Parameters:
    -----------
    returns : pd.Series
        Portfolio returns
    confidence_level : float
        Confidence level (e.g., 0.95 for 95%, 0.99 for 99%)
    
    Returns:
    --------
    float
        Historical VaR (negative value indicating loss)
    """
    if confidence_level <= 0 or confidence_level >= 1:
        raise ValueError("confidence_level must be between 0 and 1")
    
    percentile = (1 - confidence_level) * 100
    var = np.percentile(returns, percentile)
    return var


def parametric_var(
    returns: pd.Series,
    confidence_level: float = 0.95
) -> float:
    """
    Calculate Parametric VaR (assuming normal distribution).
    
    Formula: VaR = μ - z_α * σ
    where z_α is the z-score for the confidence level
    
    Parameters:
    -----------
    returns : pd.Series
        Portfolio returns
    confidence_level : float
        Confidence level (e.g., 0.95 for 95%, 0.99 for 99%)
    
    Returns:
    --------
    float
        Parametric VaR (negative value indicating loss)
    """
    if confidence_level <= 0 or confidence_level >= 1:
        raise ValueError("confidence_level must be between 0 and 1")
    
    mean = returns.mean()
    std = returns.std()
    
    # Z-score for confidence level (one-tailed)
    z_score = stats.norm.ppf(1 - confidence_level)
    
    var = mean + z_score * std
    return var


def conditional_var(
    returns: pd.Series,
    confidence_level: float = 0.95
) -> float:
    """
    Calculate Conditional Value at Risk (CVaR) / Expected Shortfall.
    
    CVaR is the expected loss given that the loss exceeds VaR.
    
    Parameters:
    -----------
    returns : pd.Series
        Portfolio returns
    confidence_level : float
        Confidence level (e.g., 0.95 for 95%, 0.99 for 99%)
    
    Returns:
    --------
    float
        Conditional VaR (negative value indicating expected loss)
    """
    if confidence_level <= 0 or confidence_level >= 1:
        raise ValueError("confidence_level must be between 0 and 1")
    
    # Calculate VaR threshold
    var_threshold = historical_var(returns, confidence_level)
    
    # Get returns below VaR threshold
    tail_returns = returns[returns <= var_threshold]
    
    if len(tail_returns) == 0:
        return var_threshold
    
    # Average of tail returns
    cvar = tail_returns.mean()
    return cvar


def calculate_var(
    returns: pd.Series,
    confidence_level: float = 0.95,
    method: str = 'historical'
) -> float:
    """
    Calculate Value at Risk using specified method.
    
    Parameters:
    -----------
    returns : pd.Series
        Portfolio returns
    confidence_level : float
        Confidence level (default: 0.95)
    method : str
        Method: 'historical' or 'parametric' (default: 'historical')
    
    Returns:
    --------
    float
        VaR value
    """
    if method.lower() == 'historical':
        return historical_var(returns, confidence_level)
    elif method.lower() == 'parametric':
        return parametric_var(returns, confidence_level)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'historical' or 'parametric'")


def beta(
    portfolio_returns: pd.Series,
    market_returns: pd.Series
) -> float:
    """
    Calculate Beta of portfolio relative to market.
    
    Beta measures the sensitivity of portfolio returns to market returns.
    Beta = Cov(r_p, r_m) / Var(r_m)
    
    Parameters:
    -----------
    portfolio_returns : pd.Series
        Portfolio returns
    market_returns : pd.Series
        Market benchmark returns
    
    Returns:
    --------
    float
        Beta value
    """
    # Align dates
    aligned_data = pd.DataFrame({
        'portfolio': portfolio_returns,
        'market': market_returns
    }).dropna()
    
    if len(aligned_data) < 2:
        return np.nan
    
    portfolio = aligned_data['portfolio']
    market = aligned_data['market']
    
    # Calculate covariance and variance
    covariance = np.cov(portfolio, market)[0, 1]
    market_variance = market.var()
    
    if market_variance == 0:
        return np.nan
    
    beta_value = covariance / market_variance
    return beta_value


def downside_deviation(
    returns: pd.Series,
    target_return: float = 0.0
) -> float:
    """
    Calculate downside deviation.
    
    Downside deviation measures volatility of negative returns.
    
    Parameters:
    -----------
    returns : pd.Series
        Portfolio returns
    target_return : float
        Target return threshold (default: 0.0)
    
    Returns:
    --------
    float
        Downside deviation
    """
    # Only consider returns below target
    downside_returns = returns[returns < target_return] - target_return
    
    if len(downside_returns) == 0:
        return 0.0
    
    # Calculate standard deviation of downside returns
    downside_std = np.sqrt(np.mean(downside_returns ** 2))
    return downside_std


def sortino_ratio(
    returns: pd.Series,
    target_return: float = 0.0,
    annualized: bool = True
) -> float:
    """
    Calculate Sortino ratio.
    
    Sortino ratio = (Return - Target) / Downside Deviation
    Similar to Sharpe ratio but only penalizes downside volatility.
    
    Parameters:
    -----------
    returns : pd.Series
        Portfolio returns
    target_return : float
        Target return (default: 0.0)
    annualized : bool
        Whether to annualize the ratio (default: True)
    
    Returns:
    --------
    float
        Sortino ratio
    """
    mean_return = returns.mean()
    dd = downside_deviation(returns, target_return)
    
    if dd == 0:
        return np.inf if mean_return > target_return else -np.inf
    
    if annualized:
        # Annualize based on frequency
        periods_per_year = _get_periods_per_year(returns)
        mean_return = mean_return * periods_per_year
        dd = dd * np.sqrt(periods_per_year)
    
    sortino = (mean_return - target_return) / dd
    return sortino


def _get_periods_per_year(returns: pd.Series) -> int:
    """
    Estimate number of periods per year from returns index.
    
    Parameters:
    -----------
    returns : pd.Series
        Returns series with datetime index
    
    Returns:
    --------
    int
        Estimated periods per year
    """
    if len(returns) < 2:
        return 252  # Default to daily
    
    # Calculate average time between observations
    time_deltas = returns.index.to_series().diff().dropna()
    avg_delta = time_deltas.median()
    
    # Estimate frequency
    if pd.Timedelta(days=1) * 0.9 <= avg_delta <= pd.Timedelta(days=1) * 1.1:
        return 252  # Daily
    elif pd.Timedelta(weeks=1) * 0.9 <= avg_delta <= pd.Timedelta(weeks=1) * 1.1:
        return 52  # Weekly
    elif pd.Timedelta(days=30) * 0.9 <= avg_delta <= pd.Timedelta(days=30) * 1.1:
        return 12  # Monthly
    else:
        return 252  # Default to daily


def maximum_drawdown(returns: pd.Series) -> float:
    """
    Calculate maximum drawdown.
    
    Maximum drawdown is the largest peak-to-trough decline.
    
    Parameters:
    -----------
    returns : pd.Series
        Portfolio returns
    
    Returns:
    --------
    float
        Maximum drawdown (negative value)
    """
    # Calculate cumulative returns
    cumulative = (1 + returns).cumprod()
    
    # Calculate running maximum
    running_max = cumulative.expanding().max()
    
    # Calculate drawdown
    drawdown = (cumulative - running_max) / running_max
    
    # Return maximum drawdown
    max_dd = drawdown.min()
    return max_dd


def calculate_risk_metrics(
    returns: pd.Series,
    market_returns: Optional[pd.Series] = None
) -> pd.Series:
    """
    Calculate comprehensive risk metrics for a portfolio.
    
    Parameters:
    -----------
    returns : pd.Series
        Portfolio returns
    market_returns : pd.Series, optional
        Market benchmark returns for beta calculation
    
    Returns:
    --------
    pd.Series
        Dictionary of risk metrics
    """
    metrics = {}
    
    # VaR metrics
    metrics['VaR_95'] = historical_var(returns, 0.95)
    metrics['VaR_99'] = historical_var(returns, 0.99)
    metrics['CVaR_95'] = conditional_var(returns, 0.95)
    metrics['CVaR_99'] = conditional_var(returns, 0.99)
    
    # Volatility metrics
    periods_per_year = _get_periods_per_year(returns)
    metrics['Volatility'] = returns.std() * np.sqrt(periods_per_year)
    metrics['Downside_Deviation'] = downside_deviation(returns) * np.sqrt(periods_per_year)
    
    # Ratio metrics
    metrics['Sortino_Ratio'] = sortino_ratio(returns, annualized=True)
    
    # Drawdown
    metrics['Max_Drawdown'] = maximum_drawdown(returns)
    
    # Beta (if market returns provided)
    if market_returns is not None:
        metrics['Beta'] = beta(returns, market_returns)
    
    return pd.Series(metrics)


def calculate_transaction_costs(
    current_weights: np.ndarray,
    new_weights: np.ndarray,
    portfolio_value: float,
    cost_per_trade_bps: float = 10.0
) -> float:
    """
    Calculate transaction costs for rebalancing.
    
    Formula: Cost = Value * sum(|w_new - w_old|) * cost_per_trade
    
    Parameters:
    -----------
    current_weights : np.ndarray
        Current portfolio weights
    new_weights : np.ndarray
        Target portfolio weights
    portfolio_value : float
        Current portfolio value
    cost_per_trade_bps : float
        Transaction cost in basis points (default: 10 bps = 0.001)
    
    Returns:
    --------
    float
        Total transaction cost
    """
    weight_change = np.abs(new_weights - current_weights).sum()
    cost_bps = cost_per_trade_bps
    cost = portfolio_value * weight_change * (cost_bps / 10000)
    return cost


def rebalancing_analysis(
    weights_history: pd.DataFrame,
    returns_df: pd.DataFrame,
    cost_per_trade_bps: float = 10.0
) -> pd.Series:
    """
    Analyze rebalancing costs vs portfolio drift.
    
    Parameters:
    -----------
    weights_history : pd.DataFrame
        DataFrame with 'date' and 'weights' columns
    returns_df : pd.DataFrame
        Historical returns DataFrame
    cost_per_trade_bps : float
        Transaction cost in basis points
    
    Returns:
    --------
    pd.Series
        Rebalancing metrics
    """
    metrics = {}
    
    if len(weights_history) < 2:
        return pd.Series({})
    
    total_cost = 0.0
    total_turnover = 0.0
    
    for i in range(1, len(weights_history)):
        prev_date = weights_history.iloc[i-1]['date']
        curr_date = weights_history.iloc[i]['date']
        
        prev_weights = weights_history.iloc[i-1]['weights']
        curr_weights = weights_history.iloc[i]['weights']
        
        # Calculate drift (how much weights changed due to returns)
        period_returns = returns_df.loc[prev_date:curr_date]
        if len(period_returns) > 0:
            # Simulate portfolio value change
            period_cumulative = (1 + period_returns).prod()
            drifted_weights = prev_weights * period_cumulative
            drifted_weights = drifted_weights / drifted_weights.sum()
            
            # Calculate turnover
            turnover = np.abs(curr_weights - prev_weights).sum()
            total_turnover += turnover
            
            # Estimate portfolio value (approximate)
            portfolio_value = 100000  # Approximate
            cost = calculate_transaction_costs(
                prev_weights.values,
                curr_weights.values,
                portfolio_value,
                cost_per_trade_bps
            )
            total_cost += cost
    
    metrics['Total_Rebalancing_Cost'] = total_cost
    metrics['Average_Turnover'] = total_turnover / max(1, len(weights_history) - 1)
    metrics['Number_of_Rebalances'] = len(weights_history) - 1
    
    return pd.Series(metrics)


def calculate_factor_exposure(
    portfolio_returns: pd.Series,
    market_returns: pd.Series,
    sector_returns: Optional[pd.DataFrame] = None
) -> pd.Series:
    """
    Calculate factor exposures including market beta and sector tilts.
    
    Parameters:
    -----------
    portfolio_returns : pd.Series
        Portfolio returns
    market_returns : pd.Series
        Market benchmark returns
    sector_returns : pd.DataFrame, optional
        Sector return time series (columns are sectors)
    
    Returns:
    --------
    pd.Series
        Factor exposure metrics
    """
    metrics = {}
    
    # Market beta
    metrics['Market_Beta'] = beta(portfolio_returns, market_returns)
    
    # Sector exposures if provided
    if sector_returns is not None:
        for sector in sector_returns.columns:
            sector_beta = beta(portfolio_returns, sector_returns[sector])
            metrics[f'{sector}_Exposure'] = sector_beta
    
    return pd.Series(metrics)


def rolling_sharpe_ratio(
    returns: pd.Series,
    window: int = 252,
    risk_free_rate: float = 0.0
) -> pd.Series:
    """
    Calculate rolling Sharpe ratio over time windows.
    
    Formula: SR_t = (R_t - R_f) / σ_t
    where R_t and σ_t are rolling mean and std
    
    Parameters:
    -----------
    returns : pd.Series
        Portfolio returns
    window : int
        Rolling window size (default: 252 for daily annualized)
    risk_free_rate : float
        Annualized risk-free rate (default: 0.0)
    
    Returns:
    --------
    pd.Series
        Rolling Sharpe ratio
    """
    periods_per_year = _get_periods_per_year(returns)
    rolling_mean = returns.rolling(window=window).mean()
    rolling_std = returns.rolling(window=window).std()
    
    # Annualize
    annualized_mean = rolling_mean * periods_per_year
    annualized_std = rolling_std * np.sqrt(periods_per_year)
    
    rolling_sharpe = (annualized_mean - risk_free_rate) / annualized_std
    return rolling_sharpe


def rolling_sortino_ratio(
    returns: pd.Series,
    window: int = 252,
    target_return: float = 0.0
) -> pd.Series:
    """
    Calculate rolling Sortino ratio over time windows.
    
    Parameters:
    -----------
    returns : pd.Series
        Portfolio returns
    window : int
        Rolling window size
    target_return : float
        Target return threshold
    
    Returns:
    --------
    pd.Series
        Rolling Sortino ratio
    """
    periods_per_year = _get_periods_per_year(returns)
    rolling_mean = returns.rolling(window=window).mean()
    
    # Rolling downside deviation
    rolling_dd = returns.rolling(window=window).apply(
        lambda x: downside_deviation(x, target_return)
    )
    
    # Annualize
    annualized_mean = rolling_mean * periods_per_year
    annualized_dd = rolling_dd * np.sqrt(periods_per_year)
    
    rolling_sortino = (annualized_mean - target_return) / annualized_dd
    return rolling_sortino


def correlation_by_period(
    returns_df: pd.DataFrame,
    period: str = 'M'
) -> pd.DataFrame:
    """
    Calculate correlation breakdown by time periods.
    
    Parameters:
    -----------
    returns_df : pd.DataFrame
        Returns DataFrame
    period : str
        Period grouping: 'M' (monthly), 'Q' (quarterly), 'Y' (yearly)
    
    Returns:
    --------
    pd.DataFrame
        Correlation matrices by period
    """
    correlations = {}
    
    for period_start, period_data in returns_df.groupby(returns_df.index.to_period(period)):
        corr_matrix = period_data.corr()
        correlations[str(period_start)] = corr_matrix
    
    return pd.DataFrame(correlations)


def performance_attribution(
    portfolio_weights: pd.Series,
    asset_returns: pd.Series,
    benchmark_weights: Optional[pd.Series] = None
) -> pd.Series:
    """
    Calculate performance attribution (allocation, selection, interaction effects).
    
    Formula:
    - Allocation Effect = sum((w_p - w_b) * R_b)
    - Selection Effect = sum(w_b * (R_p - R_b))
    - Interaction Effect = sum((w_p - w_b) * (R_p - R_b))
    
    Where:
    - w_p = portfolio weights
    - w_b = benchmark weights
    - R_p = asset returns in portfolio
    - R_b = asset returns in benchmark
    
    Parameters:
    -----------
    portfolio_weights : pd.Series
        Portfolio weights for each asset
    asset_returns : pd.Series
        Asset returns for the period
    benchmark_weights : pd.Series, optional
        Benchmark weights (default: equal weights)
    
    Returns:
    --------
    pd.Series
        Attribution metrics
    """
    # Align indices
    common_assets = portfolio_weights.index.intersection(asset_returns.index)
    portfolio_weights = portfolio_weights[common_assets]
    asset_returns = asset_returns[common_assets]
    
    if benchmark_weights is None:
        # Equal-weighted benchmark
        benchmark_weights = pd.Series(
            1.0 / len(common_assets), 
            index=common_assets
        )
    else:
        benchmark_weights = benchmark_weights[common_assets]
    
    # Portfolio return
    portfolio_return = (portfolio_weights * asset_returns).sum()
    
    # Benchmark return
    benchmark_return = (benchmark_weights * asset_returns).sum()
    
    # Allocation effect: (w_p - w_b) * R_b
    # Using portfolio asset returns as proxy for benchmark asset returns
    allocation_effect = ((portfolio_weights - benchmark_weights) * asset_returns).sum()
    
    # Selection effect: w_b * (R_p - R_b)
    # For individual assets, selection is the excess return weighted by benchmark
    selection_effect = (benchmark_weights * asset_returns).sum() - benchmark_return
    
    # Interaction effect: (w_p - w_b) * (R_p - R_b)
    # Simplified: total excess return minus allocation and selection
    interaction_effect = portfolio_return - benchmark_return - allocation_effect - selection_effect
    
    metrics = {
        'Portfolio_Return': portfolio_return,
        'Benchmark_Return': benchmark_return,
        'Excess_Return': portfolio_return - benchmark_return,
        'Allocation_Effect': allocation_effect,
        'Selection_Effect': selection_effect,
        'Interaction_Effect': interaction_effect
    }
    
    return pd.Series(metrics)


def calculate_attribution_over_time(
    portfolio_weights_history: pd.DataFrame,
    returns_df: pd.DataFrame,
    benchmark_weights: Optional[pd.Series] = None,
    period: str = 'M'
) -> pd.DataFrame:
    """
    Calculate performance attribution over time periods.
    
    Parameters:
    -----------
    portfolio_weights_history : pd.DataFrame
        DataFrame with 'date' and 'weights' columns
    returns_df : pd.DataFrame
        Asset returns DataFrame
    benchmark_weights : pd.Series, optional
        Benchmark weights
    period : str
        Period grouping: 'M', 'Q', 'Y'
    
    Returns:
    --------
    pd.DataFrame
        Attribution metrics over time
    """
    attribution_results = []
    
    if 'date' in portfolio_weights_history.columns:
        dates = portfolio_weights_history['date']
        weights_list = portfolio_weights_history['weights']
    else:
        dates = portfolio_weights_history.index
        weights_list = portfolio_weights_history['weights']
    
    # Group returns by period
    period_returns = returns_df.resample(period).apply(lambda x: (1 + x).prod() - 1)
    
    for i, (date, weights) in enumerate(zip(dates, weights_list)):
        if isinstance(weights, pd.Series):
            period_end = date if i + 1 >= len(dates) else dates.iloc[i + 1]
            
            # Get returns for this period
            period_asset_returns = period_returns.loc[
                period_returns.index >= date
            ].iloc[0] if len(period_returns[period_returns.index >= date]) > 0 else None
            
            if period_asset_returns is not None:
                attribution = performance_attribution(
                    weights,
                    period_asset_returns,
                    benchmark_weights
                )
                attribution['date'] = date
                attribution_results.append(attribution)
    
    if attribution_results:
        return pd.DataFrame(attribution_results).set_index('date')
    else:
        return pd.DataFrame()


def conditional_correlation_stress(
    returns_df: pd.DataFrame,
    stress_threshold: float = -0.02,
    method: str = 'percentile'
) -> Dict[str, pd.DataFrame]:
    """
    Calculate conditional correlation during stress periods.
    
    Measures how correlations change during market stress, which is critical
    for understanding diversification breakdown during crises.
    
    Parameters:
    -----------
    returns_df : pd.DataFrame
        Asset returns DataFrame (columns are assets, index is dates)
    stress_threshold : float
        Threshold for defining stress period:
        - If method='percentile': percentile of market returns (e.g., -0.02 for bottom 2%)
        - If method='absolute': absolute return threshold (e.g., -0.02 for -2% daily return)
        - If method='market': threshold relative to market index (requires market returns)
    method : str
        Method for defining stress: 'percentile', 'absolute', or 'market' (default: 'percentile')
    
    Returns:
    --------
    Dict[str, pd.DataFrame]
        Dictionary containing:
        - 'normal_correlation': correlation matrix during normal periods
        - 'stress_correlation': correlation matrix during stress periods
        - 'difference': difference between stress and normal correlations
        - 'stress_periods': boolean Series indicating stress periods
        - 'stress_ratio': fraction of periods identified as stress
    """
    # Calculate portfolio or market return as proxy for market conditions
    if method == 'market':
        # Use equal-weighted portfolio as market proxy if no market index provided
        market_returns = returns_df.mean(axis=1)
    else:
        market_returns = returns_df.mean(axis=1)
    
    # Identify stress periods
    if method == 'percentile':
        # Stress defined as returns below percentile threshold
        threshold = market_returns.quantile(stress_threshold) if stress_threshold < 0 else market_returns.quantile(1 - stress_threshold)
        if stress_threshold > 0:
            stress_periods = market_returns >= threshold  # Top percentile (e.g., extreme negative)
        else:
            stress_periods = market_returns <= threshold  # Bottom percentile
    elif method == 'absolute':
        # Stress defined as absolute return threshold
        stress_periods = market_returns <= stress_threshold
    elif method == 'market':
        # Stress relative to market (already handled above)
        threshold = market_returns.quantile(0.05)  # Bottom 5%
        stress_periods = market_returns <= threshold
    else:
        raise ValueError(f"Unknown method: {method}. Use 'percentile', 'absolute', or 'market'")
    
    # Separate returns into stress and normal periods
    stress_returns = returns_df[stress_periods]
    normal_returns = returns_df[~stress_periods]
    
    # Calculate correlation matrices
    normal_corr = normal_returns.corr()
    stress_corr = stress_returns.corr()
    
    # Calculate difference
    corr_difference = stress_corr - normal_corr
    
    # Calculate stress ratio
    stress_ratio = stress_periods.sum() / len(stress_periods)
    
    return {
        'normal_correlation': normal_corr,
        'stress_correlation': stress_corr,
        'difference': corr_difference,
        'stress_periods': stress_periods,
        'stress_ratio': stress_ratio,
        'n_stress_periods': stress_periods.sum(),
        'n_normal_periods': (~stress_periods).sum()
    }


def fama_french_decomposition(
    portfolio_returns: pd.Series,
    market_returns: pd.Series,
    factor_data: Optional[pd.DataFrame] = None,
    model: str = '3factor',
    risk_free_rate: Optional[pd.Series] = None
) -> Dict[str, float]:
    """
    Decompose portfolio returns using Fama-French factor models.
    
    Estimates factor loadings (betas) and decomposes returns into:
    - Market factor (beta)
    - Size factor (SMB - Small Minus Big)
    - Value factor (HML - High Minus Low)
    - Momentum factor (MOM - Momentum) for 4-factor model
    - Profitability (RMW) and Investment (CMA) for 5-factor model
    
    Parameters:
    -----------
    portfolio_returns : pd.Series
        Portfolio excess returns (portfolio return - risk-free rate)
    market_returns : pd.Series
        Market excess returns (market return - risk-free rate)
    factor_data : pd.DataFrame, optional
        DataFrame with factor returns as columns:
        - For 3-factor: ['SMB', 'HML']
        - For 4-factor: ['SMB', 'HML', 'MOM']
        - For 5-factor: ['SMB', 'HML', 'RMW', 'CMA']
        If None, will use market returns to estimate simplified model
    model : str
        Model type: '3factor', '4factor', or '5factor' (default: '3factor')
    risk_free_rate : pd.Series, optional
        Risk-free rate series (if returns are not already excess returns)
    
    Returns:
    --------
    Dict[str, float]
        Dictionary containing:
        - 'alpha': intercept (abnormal return)
        - 'market_beta': market factor loading
        - 'smb_beta': size factor loading (if applicable)
        - 'hml_beta': value factor loading (if applicable)
        - 'mom_beta': momentum factor loading (if 4-factor or 5-factor)
        - 'rmw_beta': profitability factor loading (if 5-factor)
        - 'cma_beta': investment factor loading (if 5-factor)
        - 'r_squared': R² of the regression
        - 'adjusted_r_squared': Adjusted R²
        - 'model_fit': full regression results (if using statsmodels)
    """
    from scipy import stats
    
    # Align data
    aligned_data = pd.DataFrame({
        'portfolio': portfolio_returns,
        'market': market_returns
    }).dropna()
    
    if risk_free_rate is not None:
        # Convert to excess returns if not already
        aligned_data = aligned_data.join(risk_free_rate, how='left')
        aligned_data['portfolio'] = aligned_data['portfolio'] - aligned_data.iloc[:, -1]
        aligned_data['market'] = aligned_data['market'] - aligned_data.iloc[:, -1]
        aligned_data = aligned_data[['portfolio', 'market']].dropna()
    
    if len(aligned_data) < 10:
        raise ValueError("Insufficient data for Fama-French regression")
    
    portfolio = aligned_data['portfolio'].values
    market = aligned_data['market'].values
    
    # Prepare factor matrix
    factors = [market]
    factor_names = ['market']
    
    # Add additional factors if provided
    if factor_data is not None:
        # Align factor data
        factor_aligned = factor_data.reindex(aligned_data.index).dropna()
        aligned_data = aligned_data.reindex(factor_aligned.index)
        portfolio = aligned_data['portfolio'].values
        market = aligned_data['market'].values
        
        if model in ['3factor', '4factor', '5factor']:
            if 'SMB' in factor_data.columns:
                factors.append(factor_aligned['SMB'].values)
                factor_names.append('smb')
            if 'HML' in factor_data.columns:
                factors.append(factor_aligned['HML'].values)
                factor_names.append('hml')
        
        if model in ['4factor', '5factor']:
            if 'MOM' in factor_data.columns:
                factors.append(factor_aligned['MOM'].values)
                factor_names.append('mom')
        
        if model == '5factor':
            if 'RMW' in factor_data.columns:
                factors.append(factor_aligned['RMW'].values)
                factor_names.append('rmw')
            if 'CMA' in factor_data.columns:
                factors.append(factor_aligned['CMA'].values)
                factor_names.append('cma')
    
    # Stack factors into design matrix (add intercept column)
    X = np.column_stack([np.ones(len(portfolio))] + factors)
    
    # Fit linear regression using least squares
    # y = X @ beta, solve for beta: beta = (X^T X)^(-1) X^T y
    beta_hat = np.linalg.lstsq(X, portfolio, rcond=None)[0]
    
    # Extract intercept and coefficients
    intercept = beta_hat[0]
    coefficients = beta_hat[1:]
    
    # Calculate R-squared
    y_pred = X @ beta_hat
    ss_res = np.sum((portfolio - y_pred) ** 2)
    ss_tot = np.sum((portfolio - np.mean(portfolio)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    # Adjusted R-squared
    n = len(portfolio)
    k = len(factor_names)
    adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k - 1)
    
    # Build results dictionary
    results = {
        'alpha': intercept,
        'r_squared': r_squared,
        'adjusted_r_squared': adjusted_r_squared,
        'n_observations': n
    }
    
    # Add factor betas
    for i, factor_name in enumerate(factor_names):
        results[f'{factor_name}_beta'] = coefficients[i]
    
    # Calculate t-statistics and p-values using scipy
    try:
        # Standard errors
        mse = ss_res / (n - k - 1)
        var_coef = mse * np.linalg.inv(X.T @ X)
        se_coef = np.sqrt(np.diag(var_coef))
        
        # t-statistics
        t_stats = np.append(intercept / (se_coef[0] if len(se_coef) > 0 else 1.0),
                           coefficients / se_coef[1:] if len(se_coef) > 1 else [])
        
        # p-values (two-tailed)
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - k - 1))
        
        results['alpha_tstat'] = t_stats[0] if len(t_stats) > 0 else np.nan
        results['alpha_pvalue'] = p_values[0] if len(p_values) > 0 else np.nan
        
        for i, factor_name in enumerate(factor_names):
            idx = i + 1
            if idx < len(t_stats):
                results[f'{factor_name}_tstat'] = t_stats[idx]
                results[f'{factor_name}_pvalue'] = p_values[idx]
    except:
        # If calculation fails, skip statistical tests
        pass
    
    return results


def rolling_fama_french_decomposition(
    portfolio_returns: pd.Series,
    market_returns: pd.Series,
    factor_data: Optional[pd.DataFrame] = None,
    window: int = 252,
    model: str = '3factor'
) -> pd.DataFrame:
    """
    Calculate rolling Fama-French factor loadings over time.
    
    Parameters:
    -----------
    portfolio_returns : pd.Series
        Portfolio excess returns
    market_returns : pd.Series
        Market excess returns
    factor_data : pd.DataFrame, optional
        Factor returns DataFrame
    window : int
        Rolling window size (default: 252 for daily annual window)
    model : str
        Model type: '3factor', '4factor', or '5factor'
    
    Returns:
    --------
    pd.DataFrame
        Time series of factor loadings with columns for each beta
    """
    results_list = []
    
    for i in range(window, len(portfolio_returns)):
        window_start = i - window
        portfolio_window = portfolio_returns.iloc[window_start:i]
        market_window = market_returns.iloc[window_start:i]
        
        factor_window = None
        if factor_data is not None:
            factor_window = factor_data.iloc[window_start:i]
        
        try:
            ff_results = fama_french_decomposition(
                portfolio_window,
                market_window,
                factor_window,
                model=model
            )
            ff_results['date'] = portfolio_returns.index[i]
            results_list.append(ff_results)
        except:
            # Skip if regression fails
            continue
    
    if results_list:
        return pd.DataFrame(results_list).set_index('date')
    else:
        return pd.DataFrame()
