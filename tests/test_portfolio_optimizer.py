"""
Unit tests for portfolio_optimizer module.
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.portfolio_optimizer import PortfolioOptimizer


@pytest.fixture
def sample_returns():
    """Create sample returns DataFrame for testing."""
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    np.random.seed(42)
    returns = pd.DataFrame(
        np.random.randn(100, 3) * 0.01,
        index=dates,
        columns=['A', 'B', 'C']
    )
    return returns


def test_portfolio_return(sample_returns):
    """Test portfolio return calculation."""
    weights = np.array([0.5, 0.3, 0.2])
    mean_returns = sample_returns.mean().values
    portfolio_ret = PortfolioOptimizer.portfolio_return(weights, mean_returns)
    
    expected = np.dot(weights, mean_returns)
    assert np.isclose(portfolio_ret, expected)


def test_portfolio_volatility(sample_returns):
    """Test portfolio volatility calculation."""
    weights = np.array([0.5, 0.3, 0.2])
    cov_matrix = sample_returns.cov().values
    portfolio_vol = PortfolioOptimizer.portfolio_volatility(weights, cov_matrix)
    
    expected = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    assert np.isclose(portfolio_vol, expected)


def test_sharpe_ratio(sample_returns):
    """Test Sharpe ratio calculation."""
    weights = np.array([0.5, 0.3, 0.2])
    mean_returns = sample_returns.mean().values
    cov_matrix = sample_returns.cov().values
    risk_free_rate = 0.0
    
    sharpe = PortfolioOptimizer.sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate)
    
    portfolio_ret = PortfolioOptimizer.portfolio_return(weights, mean_returns)
    portfolio_vol = PortfolioOptimizer.portfolio_volatility(weights, cov_matrix)
    expected = portfolio_ret / portfolio_vol if portfolio_vol > 0 else 0
    
    assert np.isclose(sharpe, expected)


def test_minimum_variance_portfolio(sample_returns):
    """Test minimum variance portfolio optimization."""
    optimizer = PortfolioOptimizer(sample_returns)
    weights = optimizer.minimum_variance_portfolio()
    
    # Check weights sum to 1
    assert np.isclose(weights.sum(), 1.0)
    
    # Check all weights are non-negative
    assert (weights >= 0).all()
    
    # Check weights are between 0 and 1
    assert (weights <= 1).all()


def test_maximum_sharpe_ratio_portfolio(sample_returns):
    """Test maximum Sharpe ratio portfolio optimization."""
    optimizer = PortfolioOptimizer(sample_returns)
    weights = optimizer.maximum_sharpe_ratio_portfolio(risk_free_rate=0.0)
    
    # Check weights sum to 1
    assert np.isclose(weights.sum(), 1.0)
    
    # Check all weights are non-negative
    assert (weights >= 0).all()
    
    # Check weights are between 0 and 1
    assert (weights <= 1).all()


def test_expected_returns(sample_returns):
    """Test expected returns calculation."""
    optimizer = PortfolioOptimizer(sample_returns)
    mean_returns = optimizer.calculate_expected_returns(sample_returns)
    
    expected = sample_returns.mean().values
    np.testing.assert_array_almost_equal(mean_returns, expected)


def test_covariance_matrix(sample_returns):
    """Test covariance matrix calculation."""
    optimizer = PortfolioOptimizer(sample_returns)
    cov_matrix = optimizer.calculate_covariance_matrix(sample_returns)
    
    expected = sample_returns.cov().values
    np.testing.assert_array_almost_equal(cov_matrix, expected)


if __name__ == '__main__':
    pytest.main([__file__])
