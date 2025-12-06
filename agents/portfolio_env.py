import numpy as np
import pandas as pd
import gym
from gym import spaces

class PortfolioEnv(gym.Env):
    """
    Portfolio management environment for RL
    State: [price history (10 days x 5 stocks), current allocation (5), portfolio value (1)]
    Action: New portfolio allocation [stock1_weight, stock2_weight, ..., stock5_weight]
    Reward: Daily return
    """
    
    def __init__(self, prices_df, initial_balance=100000, window_size=10, transaction_cost=0.001):
        super(PortfolioEnv, self).__init__()
        
        self.prices = prices_df.values
        self.tickers = prices_df.columns.tolist()
        self.n_stocks = len(self.tickers)
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        
        self.current_step = window_size
        self.max_steps = len(self.prices) - 1
        
        # Action space: portfolio weights [0-1] for each stock, must sum to 1
        self.action_space = spaces.Box(
            low=0, high=1, shape=(self.n_stocks,), dtype=np.float32
        )
        
        # State space: price history + current allocation + portfolio value
        state_size = (window_size * self.n_stocks) + self.n_stocks + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_size,), dtype=np.float32
        )
        
        self.reset()
    
    def reset(self):
        """Reset environment to initial state"""
        self.current_step = self.window_size
        self.portfolio_value = self.initial_balance
        self.holdings = np.zeros(self.n_stocks)  # Number of shares
        self.weights = np.ones(self.n_stocks) / self.n_stocks  # Equal weight initially
        self.portfolio_history = [self.initial_balance]
        
        return self._get_state()
    
    def _get_state(self):
        """Get current state representation"""
        # Price history (normalized)
        price_window = self.prices[self.current_step - self.window_size:self.current_step]
        price_returns = (price_window[1:] - price_window[:-1]) / (price_window[:-1] + 1e-8)
        price_features = price_returns.flatten()
        
        # Current allocation
        allocation = self.weights
        
        # Portfolio value (normalized by initial)
        portfolio_value_norm = np.array([self.portfolio_value / self.initial_balance])
        
        # Combine all features
        state = np.concatenate([price_features, allocation, portfolio_value_norm])
        
        return state.astype(np.float32)
    
    def step(self, action):
        """Execute one time step"""
        # Normalize action to ensure weights sum to 1
        action = np.clip(action, 0, 1)
        new_weights = action / (action.sum() + 1e-8)
        
        # Calculate transaction costs
        weight_change = np.abs(new_weights - self.weights).sum()
        transaction_cost = weight_change * self.transaction_cost * self.portfolio_value
        
        # Update portfolio
        self.weights = new_weights
        
        # Move to next time step
        self.current_step += 1
        
        # Calculate returns
        if self.current_step < len(self.prices):
            price_change = (self.prices[self.current_step] - self.prices[self.current_step - 1]) / \
                          (self.prices[self.current_step - 1] + 1e-8)
            
            portfolio_return = np.dot(self.weights, price_change)
            self.portfolio_value *= (1 + portfolio_return)
            self.portfolio_value -= transaction_cost
            
            reward = portfolio_return - (transaction_cost / self.portfolio_value)
        else:
            reward = 0
        
        self.portfolio_history.append(self.portfolio_value)
        
        # Check if episode is done
        done = self.current_step >= self.max_steps - 1
        
        # Get new state
        state = self._get_state()
        
        info = {
            'portfolio_value': self.portfolio_value,
            'weights': self.weights.copy(),
            'transaction_cost': transaction_cost
        }
        
        return state, reward, done, info
    
    def render(self):
        """Print current portfolio state"""
        print(f"Step: {self.current_step}/{self.max_steps}")
        print(f"Portfolio Value: ${self.portfolio_value:,.2f}")
        print(f"Weights: {dict(zip(self.tickers, self.weights.round(3)))}")


# Test the environment
if __name__ == "__main__":
    import pandas as pd
    
    prices = pd.read_csv('data/stock_prices.csv', index_col=0, parse_dates=True)
    
    env = PortfolioEnv(prices)
    state = env.reset()
    
    print("Environment initialized!")
    print(f"State shape: {state.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Number of stocks: {env.n_stocks}")
    print(f"Tickers: {env.tickers}")
    
    # Test random actions
    print("\n--- Testing 5 random steps ---")
    for i in range(5):
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        print(f"\nStep {i+1}: Reward={reward:.4f}, Portfolio=${info['portfolio_value']:,.2f}")
        if done:
            break
    
    print("\n✓ Environment working!")