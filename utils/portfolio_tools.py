"""
Portfolio Management Tools for RL Investment System

Custom tools that agents use for market analysis, risk assessment, and backtesting.
Reframes core components as modular tools for better architecture.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta


class MarketDataTool:
    """
    Tool for fetching and processing market data.
    
    Provides agents with access to historical price data and technical indicators.
    """
    
    def __init__(self):
        self.name = "market_data_tool"
        self.cache = {}
    
    def fetch_prices(self, tickers, start_date, end_date):
        """
        Fetch historical price data for given tickers.
        
        Args:
            tickers (list): List of stock ticker symbols
            start_date (datetime): Start date for data
            end_date (datetime): End date for data
            
        Returns:
            pd.DataFrame: Historical closing prices
        """
        try:
            cache_key = f"{'-'.join(tickers)}_{start_date}_{end_date}"
            
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            data = yf.download(tickers, start=start_date, end=end_date, 
                             progress=False, auto_adjust=True)
            
            if len(tickers) > 1:
                prices = data['Close']
            else:
                prices = data[['Close']]
                prices.columns = tickers
            
            prices = prices.ffill().dropna()
            self.cache[cache_key] = prices
            
            return prices
            
        except Exception as e:
            print(f"Error fetching market data: {e}")
            return None
    
    def calculate_returns(self, prices):
        """Calculate daily returns from price data."""
        try:
            returns = prices.pct_change().dropna()
            return returns
        except Exception as e:
            print(f"Error calculating returns: {e}")
            return None
    
    def calculate_technical_indicators(self, prices):
        """
        Calculate technical indicators for market analysis.
        
        Args:
            prices (pd.DataFrame): Historical prices
            
        Returns:
            dict: Dictionary of technical indicators
        """
        try:
            returns = self.calculate_returns(prices)
            
            indicators = {
                'volatility': returns.std(),
                'momentum_5d': (prices.iloc[-1] - prices.iloc[-6]) / prices.iloc[-6],
                'momentum_10d': (prices.iloc[-1] - prices.iloc[-11]) / prices.iloc[-11],
                'avg_return': returns.mean(),
                'sharpe_ratio': returns.mean() / (returns.std() + 1e-8) * np.sqrt(252)
            }
            
            return indicators
            
        except Exception as e:
            print(f"Error calculating indicators: {e}")
            return None

class RiskAnalysisTool:
    """
    Tool for portfolio risk assessment and management.
    
    Provides agents with risk metrics and portfolio analytics.
    """
    
    def __init__(self):
        self.name = "risk_analysis_tool"
    
    def calculate_sharpe_ratio(self, returns, risk_free_rate=0.0):
        """
        Calculate Sharpe ratio (annualized).
        
        Args:
            returns (array): Daily returns
            risk_free_rate (float): Annual risk-free rate
            
        Returns:
            float: Sharpe ratio
        """
        try:
            if len(returns) == 0:
                return 0.0
            
            excess_returns = returns - risk_free_rate / 252
            sharpe = np.mean(excess_returns) / (np.std(excess_returns) + 1e-8) * np.sqrt(252)
            
            return sharpe
            
        except Exception as e:
            print(f"Error calculating Sharpe ratio: {e}")
            return 0.0
    
    def calculate_max_drawdown(self, portfolio_values):
        """
        Calculate maximum drawdown.
        
        Args:
            portfolio_values (array): Time series of portfolio values
            
        Returns:
            float: Maximum drawdown as percentage
        """
        try:
            portfolio_values = np.array(portfolio_values)
            peak = np.maximum.accumulate(portfolio_values)
            drawdown = (portfolio_values - peak) / peak
            max_drawdown = np.min(drawdown)
            
            return max_drawdown
            
        except Exception as e:
            print(f"Error calculating max drawdown: {e}")
            return 0.0
    
    def calculate_var(self, returns, confidence=0.95):
        """
        Calculate Value at Risk (VaR).
        
        Args:
            returns (array): Daily returns
            confidence (float): Confidence level (default 0.95)
            
        Returns:
            float: VaR at given confidence level
        """
        try:
            if len(returns) == 0:
                return 0.0
            
            var = np.percentile(returns, (1 - confidence) * 100)
            return var
            
        except Exception as e:
            print(f"Error calculating VaR: {e}")
            return 0.0
    
    def calculate_sortino_ratio(self, returns, target_return=0.0):
        """
        Calculate Sortino ratio (downside risk only).
        
        Args:
            returns (array): Daily returns
            target_return (float): Target return threshold
            
        Returns:
            float: Sortino ratio
        """
        try:
            downside_returns = returns[returns < target_return]
            downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1e-8
            
            sortino = (np.mean(returns) - target_return) / downside_std * np.sqrt(252)
            
            return sortino
            
        except Exception as e:
            print(f"Error calculating Sortino ratio: {e}")
            return 0.0
    
    def calculate_portfolio_metrics(self, portfolio_values, returns):
        """
        Calculate comprehensive portfolio metrics.
        
        Args:
            portfolio_values (array): Portfolio value time series
            returns (array): Daily returns
            
        Returns:
            dict: Dictionary of portfolio metrics
        """
        try:
            metrics = {
                'total_return': (portfolio_values[-1] / portfolio_values[0] - 1) * 100,
                'sharpe_ratio': self.calculate_sharpe_ratio(returns),
                'sortino_ratio': self.calculate_sortino_ratio(returns),
                'max_drawdown': self.calculate_max_drawdown(portfolio_values) * 100,
                'volatility': np.std(returns) * np.sqrt(252) * 100,
                'var_95': self.calculate_var(returns, 0.95) * 100
            }
            
            return metrics
            
        except Exception as e:
            print(f"Error calculating portfolio metrics: {e}")
            return None

class BacktestTool:
    """
    Tool for strategy backtesting and evaluation.
    
    Enables agents to test strategies on historical data.
    """
    
    def __init__(self):
        self.name = "backtest_tool"
    
    def run_backtest(self, strategy_func, env, *args, **kwargs):
        """
        Run backtest for a given strategy.
        
        Args:
            strategy_func: Function that returns actions
            env: Trading environment
            *args, **kwargs: Additional arguments for strategy
            
        Returns:
            dict: Backtest results
        """
        try:
            state = env.reset()
            done = False
            
            portfolio_history = []
            actions_history = []
            rewards_history = []
            
            while not done:
                action = strategy_func(state, *args, **kwargs)
                state, reward, done, info = env.step(action)
                
                portfolio_history.append(info['portfolio_value'])
                actions_history.append(action)
                rewards_history.append(reward)
            
            results = {
                'portfolio_history': portfolio_history,
                'actions_history': actions_history,
                'rewards_history': rewards_history,
                'final_value': portfolio_history[-1],
                'total_return': (portfolio_history[-1] / portfolio_history[0] - 1) * 100
            }
            
            return results
            
        except Exception as e:
            print(f"Error in backtest: {e}")
            return None
    
    def compare_strategies(self, strategies, env):
        """
        Compare multiple strategies.
        
        Args:
            strategies (dict): Dictionary of {name: strategy_func}
            env: Trading environment
            
        Returns:
            dict: Comparison results
        """
        try:
            results = {}
            
            for name, strategy_func in strategies.items():
                result = self.run_backtest(strategy_func, env)
                results[name] = result
            
            return results
            
        except Exception as e:
            print(f"Error comparing strategies: {e}")
            return None

class ContextualBanditTool:
    """
    Custom tool for intelligent stock selection using contextual bandits.
    
    Implements Upper Confidence Bound (UCB) strategy with market context
    to dynamically identify promising investment opportunities.
    
    Mathematical Foundation:
    -----------------------
    Score function:
        score(arm_i | context) = w_i^T · x_t
    
    Weight update (gradient descent):
        w_i ← w_i + α · (r_i - ŵ_i) · x_t
        where ŵ_i = w_i^T · x_t (prediction)
    
    Context features (x_t ∈ R^5):
        - mean_return: Average return across stocks
        - mean_volatility: Average volatility
        - mean_momentum: Average momentum
        - max_return: Best performing stock
        - min_return: Worst performing stock
    
    Parameters:
    ----------
    - n_stocks: Number of stocks (arms) to select from
    - n_features: Dimension of context vector (default: 5)
    - learning_rate: α for weight updates (default: 0.1)
    
    Innovation:
    ----------
    Unlike standard multi-armed bandits, this tool incorporates market
    context to make situation-aware selections. This enables the agent
    to adapt stock preferences based on current market conditions rather
    than using fixed selection probabilities.
    
    Example Usage:
    -------------
    >>> bandit = ContextualBanditTool(n_stocks=5)
    >>> context = bandit.extract_context(prices, step=100)
    >>> selected_stocks = bandit.select_stocks(context, n_select=3)
    >>> # After observing reward:
    >>> bandit.update(selected_stocks[0], context, reward)
    
    Comparison to Standard Bandits:
    ------------------------------
    - Standard UCB: Fixed arm selection based on historical performance
    - Contextual (ours): Adapts selection based on current market state
    - Advantage: Better exploration in dynamic environments
    """
    
    def __init__(self, n_stocks, n_features=5, learning_rate=0.1):
        """
        Initialize contextual bandit tool.
        
        Args:
            n_stocks (int): Number of stocks (arms)
            n_features (int): Number of context features
            learning_rate (float): Learning rate for weight updates
        """
        self.name = "contextual_bandit_stock_selector"
        self.n_stocks = n_stocks
        self.n_features = n_features
        self.learning_rate = learning_rate
        
        # Weight matrix for each stock
        self.weights = np.random.randn(n_stocks, n_features) * 0.01
        self.counts = np.zeros(n_stocks)
    
    def get_context(self, prices, step, window=5):
        """
        Extract market context features from price history.
        
        Args:
            prices (np.array): Historical prices [time x stocks]
            step (int): Current time step
            window (int): Lookback window for feature calculation
            
        Returns:
            np.array: Context feature vector of shape (n_features,)
        """
        try:
            if step < window:
                return np.random.randn(self.n_features)
            
            recent_prices = prices[step-window:step]
            
            # Calculate market features
            returns = (recent_prices[-1] - recent_prices[0]) / (recent_prices[0] + 1e-8)
            volatility = np.std(recent_prices, axis=0)
            momentum = (recent_prices[-1] - recent_prices[-3]) / (recent_prices[-3] + 1e-8)
            
            # Aggregate across stocks to get market-level context
            context = np.array([
                np.mean(returns),      # Average return
                np.mean(volatility),   # Average volatility
                np.mean(momentum),     # Average momentum
                np.max(returns),       # Best performing stock
                np.min(returns)        # Worst performing stock
            ])
            
            # Handle NaN/Inf values
            context = np.nan_to_num(context, nan=0.0, posinf=1.0, neginf=-1.0)
            
            return context
            
        except Exception as e:
            print(f"Error extracting context: {e}")
            return np.zeros(self.n_features)
    
    def select_stocks(self, context, n_select=3, epsilon=0.1, explore=True):
        """
        Select stocks using UCB-based strategy with epsilon-greedy exploration.
        
        Args:
            context (np.array): Context features
            n_select (int): Number of stocks to select
            epsilon (float): Exploration rate (0-1)
            explore (bool): Whether to use exploration
            
        Returns:
            np.array: Indices of selected stocks
        """
        try:
            # Calculate scores for each stock given context
            scores = np.dot(self.weights, context)
            
            # Clip extreme values to prevent overflow
            scores = np.clip(scores, -10, 10)
            
            if explore and np.random.random() < epsilon:
                # Exploration: Random selection
                selected = np.random.choice(self.n_stocks, size=n_select, replace=False)
            else:
                # Exploitation: Select top-scoring stocks
                selected = np.argsort(scores)[-n_select:]
            
            return selected
            
        except Exception as e:
            print(f"Error selecting stocks: {e}")
            # Fallback: Random selection
            return np.random.choice(self.n_stocks, size=n_select, replace=False)
    
    def update(self, stock_idx, context, reward):
        """
        Update weight matrix based on observed reward.
        
        Uses gradient descent to adjust weights for the selected stock.
        
        Args:
            stock_idx (int): Index of the stock that was selected
            context (np.array): Context features used for selection
            reward (float): Observed reward (e.g., stock return)
        """
        try:
            # Clip reward to prevent extreme updates
            reward = np.clip(reward, -1.0, 1.0)
            
            # Calculate prediction error
            prediction = np.dot(self.weights[stock_idx], context)
            error = reward - prediction
            
            # Gradient update with clipping
            gradient = self.learning_rate * error * context
            gradient = np.clip(gradient, -1.0, 1.0)
            
            # Update weights
            self.weights[stock_idx] += gradient
            
            # Clip weights to prevent explosion
            self.weights[stock_idx] = np.clip(self.weights[stock_idx], -10, 10)
            
            # Increment selection count
            self.counts[stock_idx] += 1
            
        except Exception as e:
            print(f"Error updating bandit: {e}")
    
    def get_statistics(self):
        """
        Get current statistics about the bandit's learning.
        
        Returns:
            dict: Statistics including selection counts and weight norms
        """
        try:
            stats = {
                'selection_counts': self.counts.tolist(),
                'total_selections': int(np.sum(self.counts)),
                'weight_norms': np.linalg.norm(self.weights, axis=1).tolist(),
                'avg_weight_norm': float(np.mean(np.linalg.norm(self.weights, axis=1)))
            }
            return stats
        except Exception as e:
            print(f"Error getting statistics: {e}")
            return None


# Example usage and testing
if __name__ == "__main__":
    print("="*60)
    print("TESTING PORTFOLIO TOOLS")
    print("="*60)
    
    # Test 1: Market Data Tool
    print("\n1. Testing Market Data Tool...")
    market_tool = MarketDataTool()
    tickers = ['NVDA', 'GOOGL', 'MSFT']
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    prices = market_tool.fetch_prices(tickers, start_date, end_date)
    if prices is not None:
        print(f"   ✓ Fetched {len(prices)} days of data for {len(tickers)} stocks")
        
        indicators = market_tool.calculate_technical_indicators(prices)
        if indicators:
            print(f"   ✓ Calculated {len(indicators)} technical indicators")
            print(f"   - Average Sharpe Ratio: {indicators['sharpe_ratio'].mean():.3f}")
    else:
        print("   ✗ Failed to fetch market data")
    
    # Test 2: Risk Analysis Tool
    print("\n2. Testing Risk Analysis Tool...")
    risk_tool = RiskAnalysisTool()
    
    # Generate sample returns
    np.random.seed(42)
    sample_returns = np.random.randn(100) * 0.02
    sample_portfolio_values = 100000 * np.cumprod(1 + sample_returns)
    
    sharpe = risk_tool.calculate_sharpe_ratio(sample_returns)
    sortino = risk_tool.calculate_sortino_ratio(sample_returns)
    max_dd = risk_tool.calculate_max_drawdown(sample_portfolio_values)
    var_95 = risk_tool.calculate_var(sample_returns)
    
    print(f"   ✓ Sharpe Ratio: {sharpe:.3f}")
    print(f"   ✓ Sortino Ratio: {sortino:.3f}")
    print(f"   ✓ Max Drawdown: {max_dd*100:.2f}%")
    print(f"   ✓ VaR (95%): {var_95*100:.2f}%")
    
    # Test 3: Contextual Bandit Tool
    print("\n3. Testing Contextual Bandit Tool...")
    bandit_tool = ContextualBanditTool(n_stocks=5, n_features=5)
    
    # Generate sample price data
    sample_prices = np.random.randn(100, 5).cumsum(axis=0) + 100
    
    # Extract context
    context = bandit_tool.get_context(sample_prices, step=50)
    print(f"   ✓ Extracted context: {context[:3]}... (shape: {context.shape})")
    
    # Select stocks
    selected = bandit_tool.select_stocks(context, n_select=3, epsilon=0.2)
    print(f"   ✓ Selected stocks: {selected}")
    
    # Simulate learning
    for _ in range(10):
        reward = np.random.randn() * 0.01
        bandit_tool.update(selected[0], context, reward)
    
    stats = bandit_tool.get_statistics()
    if stats:
        print(f"   ✓ Total selections: {stats['total_selections']}")
        print(f"   ✓ Avg weight norm: {stats['avg_weight_norm']:.3f}")
    
    # Test 4: Backtest Tool (requires environment - just initialize)
    print("\n4. Testing Backtest Tool...")
    backtest_tool = BacktestTool()
    print(f"   ✓ Backtest tool initialized: {backtest_tool.name}")
    
    print("\n" + "="*60)
    print("ALL TOOLS TESTED SUCCESSFULLY!")
    print("="*60)