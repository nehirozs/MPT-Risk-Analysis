"""
Unit tests for risk_metrics module.
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.risk_metrics import (
    historical_var,
    parametric_var,
    conditional_var,
    beta,
    downside_deviation,
    sortino_ratio,
    maximum_drawdown
)


@pytest.fixture
def sample_returns():
    """Create sample returns for testing."""
    np.random.seed(42)
    returns = pd.Series(np.random.randn(100) * 0.01)
    return returns


@pytest.fixture
def market_returns():
    """Create sample market returns for testing."""
    np.random.seed(43)
    returns = pd.Series(np.random.randn(100) * 0.015)
    return returns


def test_historical_var(sample_returns):
    """Test historical VaR calculation."""
    var_95 = historical_var(sample_returns, confidence_level=0.95)
    var_99 = historical_var(sample_returns, confidence_level=0.99)
    
    # VaR should be negative (loss)
    assert var_99 <= var_95 <= 0


def test_parametric_var(sample_returns):
    """Test parametric VaR calculation."""
    var_95 = parametric_var(sample_returns, confidence_level=0.95)
    var_99 = parametric_var(sample_returns, confidence_level=0.99)
    
    # VaR should be negative (loss)
    assert var_99 <= var_95


def test_conditional_var(sample_returns):
    """Test conditional VaR calculation."""
    cvar_95 = conditional_var(sample_returns, confidence_level=0.95)
    cvar_99 = conditional_var(sample_returns, confidence_level=0.99)
    
    # CVaR should be <= VaR (more negative)
    var_95 = historical_var(sample_returns, confidence_level=0.95)
    assert cvar_95 <= var_95


def test_beta(sample_returns, market_returns):
    """Test beta calculation."""
    beta_value = beta(sample_returns, market_returns)
    
    # Beta should be a number
    assert not np.isnan(beta_value)


def test_downside_deviation(sample_returns):
    """Test downside deviation calculation."""
    dd = downside_deviation(sample_returns, target_return=0.0)
    
    # Downside deviation should be non-negative
    assert dd >= 0


def test_maximum_drawdown(sample_returns):
    """Test maximum drawdown calculation."""
    max_dd = maximum_drawdown(sample_returns)
    
    # Maximum drawdown should be negative (loss)
    assert max_dd <= 0


def test_sortino_ratio(sample_returns):
    """Test Sortino ratio calculation."""
    sortino = sortino_ratio(sample_returns, annualized=False)
    
    # Sortino ratio should be a finite number
    assert np.isfinite(sortino)


if __name__ == '__main__':
    pytest.main([__file__])
