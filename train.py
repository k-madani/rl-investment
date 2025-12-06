import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from agents.portfolio_env import PortfolioEnv
from agents.dqn_agent import DQNAgent
import torch
import os

class ContextualBandit:
    """
    Contextual Bandit for stock selection/exploration
    Helps decide which stocks to focus on based on market conditions
    """
    def __init__(self, n_stocks, n_features=5):
        self.n_stocks = n_stocks
        self.n_features = n_features
        # Weight matrix for each stock
        self.weights = np.random.randn(n_stocks, n_features) * 0.01
        self.counts = np.zeros(n_stocks)
        self.alpha = 0.1  # Learning rate
        
    def get_context(self, prices, step):
        """Extract market context features"""
        if step < 5:
            return np.random.randn(self.n_features)
        
        recent_prices = prices[step-5:step]
        
        # Simple features: recent returns, volatility, momentum
        returns = (recent_prices[-1] - recent_prices[0]) / (recent_prices[0] + 1e-8)
        volatility = np.std(recent_prices, axis=0)
        momentum = (recent_prices[-1] - recent_prices[-3]) / (recent_prices[-3] + 1e-8)
        
        # Average features across stocks for context
        context = np.array([
            np.mean(returns),
            np.mean(volatility),
            np.mean(momentum),
            np.max(returns),
            np.min(returns)
        ])
        
        return context
    
    def select_stocks(self, context, explore=True, epsilon=0.1):
        """Select which stocks to favor using UCB"""
        scores = np.dot(self.weights, context)
        
        if explore and np.random.random() < epsilon:
            # Exploration: random selection
            selected = np.random.choice(self.n_stocks, size=3, replace=False)
        else:
            # Exploitation: top stocks by score
            selected = np.argsort(scores)[-3:]  # Top 3 stocks
        
        return selected
    
    def update(self, stock_idx, context, reward):
        """Update weights based on reward"""
        prediction = np.dot(self.weights[stock_idx], context)
        error = reward - prediction
        self.weights[stock_idx] += self.alpha * error * context
        self.counts[stock_idx] += 1


def train_dqn(env, agent, bandit, n_episodes=100, target_update_freq=10):
    """Train DQN agent with Contextual Bandit exploration"""
    
    episode_rewards = []
    portfolio_values = []
    losses = []
    
    print("Starting training...")
    print(f"Episodes: {n_episodes}")
    print(f"Initial epsilon: {agent.epsilon:.3f}\n")
    
    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        episode_loss = []
        done = False
        
        while not done:
            # Get market context for bandit
            context = bandit.get_context(env.prices, env.current_step)
            
            # Bandit selects favorable stocks
            if episode < n_episodes * 0.3:  # Use bandit more in early episodes
                favored_stocks = bandit.select_stocks(context, explore=True, epsilon=0.3)
            else:
                favored_stocks = bandit.select_stocks(context, explore=False)
            
            # DQN selects action (portfolio weights)
            action = agent.select_action(state, training=True)
            
            # Boost weights for favored stocks (bandit influence)
            if episode < n_episodes * 0.5:
                action_boosted = action.copy()
                action_boosted[favored_stocks] *= 1.5
                action_boosted = action_boosted / action_boosted.sum()
            else:
                action_boosted = action
            
            # Execute action
            next_state, reward, done, info = env.step(action_boosted)
            
            # Store in replay buffer
            agent.memory.push(state, action_boosted, reward, next_state, done)
            
            # Train agent
            loss = agent.train_step()
            if loss is not None:
                episode_loss.append(loss)
            
            # Update bandit for favored stocks
            for stock_idx in favored_stocks:
                stock_return = (env.prices[env.current_step, stock_idx] - 
                               env.prices[env.current_step-1, stock_idx]) / \
                               (env.prices[env.current_step-1, stock_idx] + 1e-8)
                bandit.update(stock_idx, context, stock_return)
            
            episode_reward += reward
            state = next_state
        
        # Update target network
        if episode % target_update_freq == 0:
            agent.update_target_network()
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Record metrics
        episode_rewards.append(episode_reward)
        portfolio_values.append(env.portfolio_history[-1])
        if episode_loss:
            losses.append(np.mean(episode_loss))
        
        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_value = np.mean(portfolio_values[-10:])
            print(f"Episode {episode+1}/{n_episodes} | "
                  f"Avg Reward: {avg_reward:.4f} | "
                  f"Avg Portfolio: ${avg_value:,.2f} | "
                  f"Epsilon: {agent.epsilon:.3f}")
    
    print("\n✓ Training complete!")
    return episode_rewards, portfolio_values, losses


def plot_results(episode_rewards, portfolio_values, losses):
    """Plot training results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Episode rewards
    axes[0, 0].plot(episode_rewards)
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].grid(True)
    
    # Portfolio values
    axes[0, 1].plot(portfolio_values)
    axes[0, 1].axhline(y=100000, color='r', linestyle='--', label='Initial Value')
    axes[0, 1].set_title('Portfolio Value Over Episodes')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Portfolio Value ($)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Training loss
    if losses:
        axes[1, 0].plot(losses)
        axes[1, 0].set_title('Training Loss')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True)
    
    # Moving average of portfolio value
    window = 10
    if len(portfolio_values) >= window:
        ma = pd.Series(portfolio_values).rolling(window=window).mean()
        axes[1, 1].plot(portfolio_values, alpha=0.3, label='Raw')
        axes[1, 1].plot(ma, label=f'{window}-Episode MA')
        axes[1, 1].axhline(y=100000, color='r', linestyle='--', label='Initial')
        axes[1, 1].set_title('Portfolio Value (Smoothed)')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Portfolio Value ($)')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
    print("✓ Results saved to training_results.png")
    plt.close()


if __name__ == "__main__":
    # Load data
    print("Loading data...")
    prices = pd.read_csv('data/stock_prices.csv', index_col=0, parse_dates=True)
    
    # Create environment
    env = PortfolioEnv(prices, initial_balance=100000)
    
    # Create agents
    state_size = env.observation_space.shape[0]
    action_size = env.n_stocks

    print(f"Creating DQN with state_size={state_size}, action_size={action_size}")
    dqn_agent = DQNAgent(state_size, action_size, learning_rate=0.001)
    bandit = ContextualBandit(n_stocks=action_size)
    
    print(f"\nEnvironment: {env.n_stocks} stocks, {len(prices)} days")
    print(f"DQN State size: {state_size}, Action size: {action_size}")
    print(f"Contextual Bandit: {action_size} arms\n")
    
    # Train
    episode_rewards, portfolio_values, losses = train_dqn(
        env, dqn_agent, bandit, n_episodes=100
    )
    
    # Save model
    os.makedirs('models', exist_ok=True)
    dqn_agent.save('models/dqn_portfolio.pth')
    print("✓ Model saved to models/dqn_portfolio.pth")
    
    # Plot results
    plot_results(episode_rewards, portfolio_values, losses)
    
    # Print final stats
    print(f"\nFinal Results:")
    print(f"Initial Portfolio Value: $100,000")
    print(f"Final Portfolio Value: ${portfolio_values[-1]:,.2f}")
    print(f"Total Return: {((portfolio_values[-1] / 100000 - 1) * 100):.2f}%")
    print(f"Best Portfolio Value: ${max(portfolio_values):,.2f}")