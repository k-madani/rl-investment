import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from agents.portfolio_env import PortfolioEnv
from agents.dqn_agent import DQNAgent
import os
from utils.paths import get_figure_path, get_model_path

from utils.error_handler import (
    ActionValidator, StateValidator, FallbackPolicy, 
    CircuitBreaker, safe_execute, logger
)

# Initialize error handling components
action_validator = None  # Will be initialized after env creation
fallback_policy = None
circuit_breaker = CircuitBreaker(failure_threshold=5, reset_time=50)

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
    """Train DQN agent with Contextual Bandit exploration and error handling"""
    
    # Initialize error handling
    action_validator = ActionValidator(n_stocks=env.n_stocks)
    fallback_policy = FallbackPolicy(n_stocks=env.n_stocks)
    circuit_breaker = CircuitBreaker(failure_threshold=5, reset_time=50)
    
    episode_rewards = []
    portfolio_values = []
    losses = []
    
    print("Starting training with error handling...")
    print(f"Episodes: {n_episodes}")
    print(f"Initial epsilon: {agent.epsilon:.3f}\n")
    
    for episode in range(n_episodes):
        try:
            state = env.reset()
            episode_reward = 0
            episode_loss = []
            done = False
            step = 0
            
            while not done:
                try:
                    # Check circuit breaker
                    if circuit_breaker.should_use_fallback(step):
                        logger.warning(f"Using fallback policy at step {step}")
                        action = fallback_policy.get_action('equal_weight', state)
                        favored_stocks = np.array([0, 1, 2])
                    else:
                        # Get context for bandit
                        context = bandit.get_context(env.prices, env.current_step)
                        
                        # Bandit selects stocks
                        if episode < n_episodes * 0.3:
                            favored_stocks = bandit.select_stocks(context, epsilon=0.3, explore=True)
                        else:
                            favored_stocks = bandit.select_stocks(context, explore=False)
                                                
                        # DQN selects action
                        action = agent.select_action(state, training=True)
                        
                        # Validate and sanitize action
                        is_valid, error_msg = action_validator.validate(action)
                        if not is_valid:
                            logger.warning(f"Invalid action: {error_msg}")
                            action = action_validator.sanitize(action, strategy='clip')
                            circuit_breaker.record_failure(step)
                        else:
                            circuit_breaker.record_success()
                        
                        # Coordination: boost favored stocks
                        if episode < n_episodes * 0.5:
                            action_boosted = action.copy()
                            action_boosted[favored_stocks] *= 1.5
                            action_boosted = action_boosted / (action_boosted.sum() + 1e-8)
                            
                            # Re-validate boosted action
                            is_valid, _ = action_validator.validate(action_boosted)
                            if is_valid:
                                action = action_boosted
                        
                    # Execute action
                    next_state, reward, done, info = env.step(action)
                    
                    # Store in replay buffer
                    agent.memory.push(state, action, reward, next_state, done)
                    
                    # Train agent
                    loss = agent.train_step()
                    if loss is not None and not np.isnan(loss):
                        episode_loss.append(loss)
                    
                    # Update bandit
                    for stock_idx in favored_stocks:
                        try:
                            stock_return = (env.prices[env.current_step, stock_idx] - 
                                          env.prices[env.current_step-1, stock_idx]) / \
                                          (env.prices[env.current_step-1, stock_idx] + 1e-8)
                            bandit.update(stock_idx, context, stock_return)
                        except Exception as e:
                            logger.debug(f"Bandit update failed: {e}")
                            continue
                    
                    episode_reward += reward
                    state = next_state
                    step += 1
                    
                except Exception as e:
                    logger.error(f"Error in training step {step}: {e}")
                    circuit_breaker.record_failure(step)
                    break
            
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
                      f"Epsilon: {agent.epsilon:.3f} | "
                      f"Failures: {circuit_breaker.failure_count}")
        
        except Exception as e:
            logger.error(f"Error in episode {episode}: {e}")
            continue
    
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
    plt.savefig(get_figure_path('training_results.png'), dpi=300, bbox_inches='tight')
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
    dqn_agent.save(get_model_path('dqn_portfolio.pth'))
    print("✓ Model saved to models/dqn_portfolio.pth")
    
    # Plot results
    plot_results(episode_rewards, portfolio_values, losses)
    
    # Print final stats
    print(f"\nFinal Results:")
    print(f"Initial Portfolio Value: $100,000")
    print(f"Final Portfolio Value: ${portfolio_values[-1]:,.2f}")
    print(f"Total Return: {((portfolio_values[-1] / 100000 - 1) * 100):.2f}%")
    print(f"Best Portfolio Value: ${max(portfolio_values):,.2f}")