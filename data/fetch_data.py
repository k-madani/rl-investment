import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class MarketDataFetcher:
    def __init__(self, tickers, start_date, end_date):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        
    def fetch_data(self):
        """Fetch historical price data"""
        # Download all tickers at once
        data = yf.download(self.tickers, start=self.start_date, end=self.end_date, 
                          progress=False, auto_adjust=True)
        
        # Extract Close prices
        if len(self.tickers) > 1:
            prices = data['Close']
        else:
            prices = data[['Close']]
            prices.columns = self.tickers
        
        # Handle missing data
        prices = prices.ffill().dropna()
        
        return prices
    
    def calculate_returns(self, prices):
        """Calculate daily returns"""
        returns = prices.pct_change().dropna()
        return returns

# Test it
if __name__ == "__main__":
    tickers = ['NVDA', 'GOOGL', 'MSFT', 'AMD', 'META']
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*2)  # 2 years of data
    
    fetcher = MarketDataFetcher(tickers, start_date, end_date)
    prices = fetcher.fetch_data()
    returns = fetcher.calculate_returns(prices)
    
    print("Data shape:", prices.shape)
    print("\nFirst few rows:")
    print(prices.head())
    print("\nLast few rows:")
    print(prices.tail())
    print("\nSummary stats:")
    print(prices.describe())
    
    # Save to CSV for later use
    prices.to_csv('data/stock_prices.csv')
    print("\n✓ Data saved to data/stock_prices.csv")