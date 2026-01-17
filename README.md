# Portfolio Optimization & Backtesting Framework

Link to demo: https://github.com/nehirozs/MPT-Risk-Analysis/blob/main/notebooks/demo.ipynb 

A production-ready Python framework implementing Modern Portfolio Theory (MPT) for systematic portfolio construction, backtesting, and risk analysis. Built for quantitative analysis with mathematical foundations and institutional-grade features.

## Overview

This framework provides end-to-end portfolio optimization capabilities from data acquisition to performance attribution, incorporating real-world constraints, transaction costs, and comprehensive risk metrics. It demonstrates proficiency in quantitative finance, optimization theory, and software engineering practices.

**Key Capabilities:**
- Portfolio optimization algorithms (Minimum Variance, Maximum Sharpe Ratio, Efficient Frontier)
- Multi-strategy backtesting with transaction cost modeling
- Comprehensive risk analytics (VaR, CVaR, Beta, Drawdown analysis)
- Performance attribution (allocation, selection, interaction effects)
- Factor exposure analysis and benchmark comparison
- Realistic constraint modeling (position limits, sector concentration, turnover)

## Technical Highlights

### Portfolio Optimization

Implements constrained optimization using quadratic programming (SLSQP) to solve:

**Minimum Variance Portfolio:**
\[\min \sigma_p^2 = \mathbf{w}^T \mathbf{\Sigma} \mathbf{w}\]
subject to: \(\sum w_i = 1\), \(w_i \geq 0\), position/sector constraints

**Maximum Sharpe Ratio Portfolio:**
\[\max \frac{\mathbf{w}^T \mathbf{\mu} - R_f}{\sqrt{\mathbf{w}^T \mathbf{\Sigma} \mathbf{w}}}\]
subject to: \(\sum w_i = 1\), \(w_i \geq 0\), realistic constraints

**Features:**
- Efficient frontier generation for risk-return analysis
- Position limits (e.g., max 20% per asset)
- Sector concentration constraints
- Turnover constraints for transaction cost management

### Trading Strategies

1. **Buy & Hold**: Equal-weighted baseline with periodic rebalancing
2. **Momentum**: Top N selection based on K-month rolling returns
3. **Mean Reversion**: Inverse volatility weighting
4. **Minimum Variance**: Monthly reoptimization with transaction cost awareness

### Risk Metrics

- **Value at Risk (VaR)**: Historical and parametric methods (95%, 99% confidence levels)
- **Conditional VaR (CVaR)**: Expected shortfall beyond VaR threshold
- **Beta Analysis**: Market sensitivity relative to benchmark (S&P 500)
- **Drawdown Metrics**: Maximum drawdown, drawdown duration, recovery time
- **Rolling Metrics**: Time-varying Sharpe/Sortino ratios (252-day windows)
- **Downside Deviation**: Lower partial moments for downside risk

### Backtesting Engine

- **Walk-forward simulation** with realistic rebalancing frequencies (daily, weekly, monthly, quarterly)
- **Transaction cost modeling**: 10 bps per trade impact analysis
- **Performance attribution**: Decomposes returns into:
  - Allocation effect (weighting decisions)
  - Selection effect (asset-specific returns)
  - Interaction effect (combined impact)
- **Benchmark comparison**: Alpha/beta calculation vs S&P 500
- **Stress testing**: COVID crash and period-specific analysis

### Advanced Analytics

- **Factor Exposure**: Market beta and sector tilt analysis
- **Rolling Window Analysis**: Dynamic risk-return metrics over time
- **Correlation Breakdown**: Period-by-period correlation matrices
- **Rebalancing Analysis**: Cost vs drift trade-off optimization

## Project Structure

```
portfolio-optimizer/
├── src/
│   ├── data_fetcher.py          # Historical data acquisition (yfinance)
│   ├── portfolio_optimizer.py   # MPT optimization algorithms
│   ├── strategies.py            # Trading strategy implementations
│   ├── backtester.py            # Backtesting simulation engine
│   ├── risk_metrics.py          # Risk analytics (VaR, CVaR, etc.)
│   └── visualizations.py        # Interactive charts (Plotly)
├── notebooks/
│   └── demo.ipynb               # Complete workflow demonstration
├── tests/                       # Unit tests (pytest)
└── data/                        # Cached historical data
```

## Installation

**Requirements:**
- Python 3.8+
- NumPy, Pandas, SciPy (optimization)
- yfinance (data)
- Plotly (visualizations)

**Setup:**
```bash
git clone <repository-url>
cd portfolio-optimizer
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Quick Start

```python
from src.data_fetcher import DataFetcher
from src.portfolio_optimizer import PortfolioOptimizer
from src.strategies import Momentum, BuyAndHold
from src.backtester import Backtester

# Fetch historical data
fetcher = DataFetcher()
tickers = ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'JNJ']
data = fetcher.fetch_data(tickers, start_date='2020-01-01', end_date='2023-12-31')

# Optimize portfolio
optimizer = PortfolioOptimizer(data['returns'])
max_sharpe_weights = optimizer.maximum_sharpe_ratio_portfolio(risk_free_rate=0.02)
efficient_frontier = optimizer.efficient_frontier(num_portfolios=50)

# Backtest strategy
strategy = Momentum(top_n=3, lookback_months=6)
backtester = Backtester(strategy, data['returns'], 
                       initial_capital=100000,
                       rebalance_frequency='M',
                       transaction_cost=0.001)  # 10 bps

results = backtester.run()
metrics = backtester.calculate_performance_metrics()
```

**Interactive Demo:**
```bash
jupyter notebook notebooks/demo.ipynb
```

## Mathematical Foundations

### Modern Portfolio Theory

Developed by Harry Markowitz (1952), MPT quantifies the risk-return trade-off through diversification.

**Portfolio Return:**
\[R_p = \sum_{i=1}^{n} w_i R_i\]

**Portfolio Variance:**
\[\sigma_p^2 = \sum_{i=1}^{n} \sum_{j=1}^{n} w_i w_j \sigma_{ij} = \mathbf{w}^T \mathbf{\Sigma} \mathbf{w}\]

**Sharpe Ratio:**
\[SR = \frac{R_p - R_f}{\sigma_p}\]

Where:
- \(w_i\) = weight of asset \(i\)
- \(\sigma_{ij}\) = covariance between assets \(i\) and \(j\)
- \(\mathbf{\Sigma}\) = covariance matrix
- \(R_f\) = risk-free rate

### Optimization Methodology

The framework uses **Sequential Least Squares Programming (SLSQP)** for constrained nonlinear optimization:

- **Objective functions**: Portfolio variance (minimization) or Sharpe ratio (maximization)
- **Constraints**: Budget constraint (\(\sum w_i = 1\)), non-negativity (\(w_i \geq 0\)), position limits (\(w_i \leq w_{max}\)), sector constraints (\(\sum_{i \in S} w_i \leq s_{max}\))
- **Solution stability**: Covariance shrinkage and regularization techniques for estimation error mitigation

## Key Features

### Real-World Constraints

- **Position Limits**: Enforce maximum position sizes (e.g., 20% per stock)
- **Sector Concentration**: Limit sector exposure (e.g., max 40% technology)
- **Turnover Constraints**: Control rebalancing frequency to manage transaction costs

### Transaction Cost Modeling

Realistic simulation of trading costs (default: 10 bps per trade) with impact analysis:
- Cost vs drift trade-off
- Optimal rebalancing frequency
- Impact on Sharpe ratio and total returns

### Performance Attribution

Decomposes portfolio returns into:
- **Allocation Effect**: \(\sum_i (w_{p,i} - w_{b,i}) \times R_{b,i}\)
- **Selection Effect**: \(\sum_i w_{b,i} \times (R_{p,i} - R_{b,i})\)
- **Interaction Effect**: \(\sum_i (w_{p,i} - w_{b,i}) \times (R_{p,i} - R_{b,i})\)

Where \(w_p, R_p\) are portfolio weights/returns and \(w_b, R_b\) are benchmark weights/returns.

## Code Quality

- **Modular Architecture**: Clean separation of concerns (data, optimization, strategies, analytics)
- **Type Hints**: Full type annotation for maintainability
- **Documentation**: Comprehensive docstrings with mathematical notation
- **Testing**: Unit tests for core functionality (pytest)
- **Error Handling**: Robust validation and exception handling

## Technologies & Libraries

- **NumPy/SciPy**: Numerical computing and optimization
- **Pandas**: Data manipulation and time series analysis
- **yfinance**: Market data acquisition
- **Plotly**: Interactive visualizations
- **Jupyter**: Reproducible research workflow

## Future Enhancements

- **Monte Carlo Simulation**: Portfolio forecasting with stochastic processes
- **Black-Litterman Model**: Bayesian integration of market views
- **Risk Parity**: Equal risk contribution optimization
- **Fama-French Factors**: Multi-factor risk decomposition
- **Machine Learning**: LSTM/Transformer-based return prediction
- **Alternative Data**: Sentiment analysis, satellite data integration

## Acknowledgments

- Modern Portfolio Theory (Harry Markowitz, 1952)
- yfinance library for financial data
- Scientific Python stack (NumPy, SciPy, Pandas)
