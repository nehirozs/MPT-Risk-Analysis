"""
Data fetching and preprocessing module for portfolio optimization.

This module handles downloading historical stock data, calculating returns,
and managing data caching.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import os
import pickle
from datetime import datetime
from typing import List, Dict, Optional


class DataFetcher:
    """Fetches and processes historical stock data."""
    
    def __init__(self, data_dir: str = 'data'):
        """
        Initialize DataFetcher.
        
        Parameters:
        -----------
        data_dir : str
            Directory to store cached data files
        """
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
    
    def fetch_data(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        cache: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical stock data for given tickers.
        
        Parameters:
        -----------
        tickers : List[str]
            List of stock ticker symbols
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str
            End date in 'YYYY-MM-DD' format
        cache : bool
            Whether to cache data to CSV files
        
        Returns:
        --------
        Dict[str, pd.DataFrame]
            Dictionary containing 'prices' and 'returns' DataFrames
        """
        # Check cache first
        cache_file = os.path.join(self.data_dir, f"data_{'_'.join(sorted(tickers))}_{start_date}_{end_date}.pkl")
        
        if cache and os.path.exists(cache_file):
            print(f"Loading cached data from {cache_file}")
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                # Handle backward compatibility: if old cache was just a DataFrame
                if isinstance(cached_data, pd.DataFrame):
                    print("⚠ Old cache format detected. Regenerating with new format...")
                    # Delete old cache and continue to fetch fresh data
                    os.remove(cache_file)
                else:
                    return cached_data
            except Exception as e:
                print(f"⚠ Error loading cache: {e}. Fetching fresh data...")
                if os.path.exists(cache_file):
                    os.remove(cache_file)
        
        print(f"Fetching data for {len(tickers)} tickers from {start_date} to {end_date}...")
        
        # Fetch data using yfinance
        prices_dict = {}
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                data = stock.history(start=start_date, end=end_date)
                if not data.empty:
                    prices_dict[ticker] = data['Close']
                    print(f"✓ Fetched {ticker}: {len(data)} days")
                else:
                    print(f"⚠ No data for {ticker}")
            except Exception as e:
                print(f"✗ Error fetching {ticker}: {e}")
        
        if not prices_dict:
            raise ValueError("No data was successfully fetched for any ticker")
        
        # Create prices DataFrame with aligned dates
        prices_df = pd.DataFrame(prices_dict)
        prices_df.index = pd.to_datetime(prices_df.index)
        
        # Handle missing data - forward fill and then backward fill
        prices_df = prices_df.ffill().bfill()
        
        # Drop any remaining rows with NaN values
        prices_df = prices_df.dropna()
        
        # Calculate daily returns
        returns_df = prices_df.pct_change().dropna()
        
        # Calculate monthly returns (resample to month-end)
        monthly_returns = prices_df.resample('M').last().pct_change().dropna()
        
        result = {
            'prices': prices_df,
            'returns': returns_df,
            'monthly_returns': monthly_returns
        }
        
        # Cache the data (save entire result dict, not just prices)
        if cache:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
            print(f"Data cached to {cache_file}")
        
        return result
    
    def get_market_data(
        self,
        ticker: str = '^GSPC',  # S&P 500
        start_date: str = None,
        end_date: str = None
    ) -> pd.DataFrame:
        """
        Fetch market benchmark data (default: S&P 500).
        
        Parameters:
        -----------
        ticker : str
            Benchmark ticker symbol (default: '^GSPC' for S&P 500)
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str
            End date in 'YYYY-MM-DD' format
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with 'Close' prices and 'Returns'
        """
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(start=start_date, end=end_date)
            data['Returns'] = data['Close'].pct_change()
            return data[['Close', 'Returns']].dropna()
        except Exception as e:
            print(f"Error fetching benchmark data for {ticker}: {e}")
            return pd.DataFrame()


def calculate_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate daily returns from price data.
    
    Parameters:
    -----------
    prices : pd.DataFrame
        DataFrame with stock prices
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with daily returns
    """
    return prices.pct_change().dropna()


def calculate_monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate monthly returns from price data.
    
    Parameters:
    -----------
    prices : pd.DataFrame
        DataFrame with stock prices
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with monthly returns
    """
    monthly_prices = prices.resample('M').last()
    return monthly_prices.pct_change().dropna()


def align_dates(*dataframes: pd.DataFrame) -> List[pd.DataFrame]:
    """
    Align multiple DataFrames to common date index.
    
    Parameters:
    -----------
    *dataframes : pd.DataFrame
        Variable number of DataFrames to align
    
    Returns:
    --------
    List[pd.DataFrame]
        List of aligned DataFrames
    """
    # Find common index
    common_index = dataframes[0].index
    for df in dataframes[1:]:
        common_index = common_index.intersection(df.index)
    
    # Reindex all DataFrames
    aligned = [df.reindex(common_index) for df in dataframes]
    return aligned
