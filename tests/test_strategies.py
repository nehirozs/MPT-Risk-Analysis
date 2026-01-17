"""
Unit tests for strategies module.
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.strategies import BuyAndHold, Momentum, MeanReversion, MinimumVariance


@pytest.fixture
def sample_returns():
    """Create sample returns DataFrame for testing."""
    dates = pd.date_range('2020-01-01', periods=252, freq='D')
    np.random.seed(42)
    returns = pd.DataFrame(
        np.random.randn(252, 5) * 0.01,
        index=dates,
        columns=['A', 'B', 'C', 'D', 'E']
    )
    return returns


def test_buy_and_hold(sample_returns):
    """Test Buy & Hold strategy."""
    strategy = BuyAndHold()
    weights = strategy.get_weights(sample_returns)
    
    # Check weights sum to 1
    assert np.isclose(weights.sum(), 1.0)
    
    # Check all weights are equal
    expected_weight = 1.0 / len(sample_returns.columns)
    assert all(np.isclose(w, expected_weight) for w in weights.values)


def test_momentum(sample_returns):
    """Test Momentum strategy."""
    strategy = Momentum(top_n=3, lookback_months=3)
    current_date = sample_returns.index[100]
    weights = strategy.get_weights(sample_returns, current_date)
    
    # Check weights sum to 1
    assert np.isclose(weights.sum(), 1.0)
    
    # Check all weights are non-negative
    assert (weights >= 0).all()


def test_mean_reversion(sample_returns):
    """Test Mean Reversion strategy."""
    strategy = MeanReversion(lookback_months=12)
    current_date = sample_returns.index[100]
    weights = strategy.get_weights(sample_returns, current_date)
    
    # Check weights sum to 1
    assert np.isclose(weights.sum(), 1.0)
    
    # Check all weights are non-negative
    assert (weights >= 0).all()


def test_minimum_variance(sample_returns):
    """Test Minimum Variance strategy."""
    strategy = MinimumVariance(lookback_months=12)
    current_date = sample_returns.index[100]
    weights = strategy.get_weights(sample_returns, current_date)
    
    # Check weights sum to 1
    assert np.isclose(weights.sum(), 1.0)
    
    # Check all weights are non-negative
    assert (weights >= 0).all()


if __name__ == '__main__':
    pytest.main([__file__])
