import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from agents.portfolio_env import PortfolioEnv
from agents.dqn_agent import DQNAgent
import torch
from utils.paths import get_figure_path, get_model_path

def baseline_equal_weight(env):
    """Baseline: Equal weight portfolio (20% each stock)"""
    state = env.reset()
    done = False
    action = np.ones(env.n_stocks) / env.n_stocks  # Equal weights
    
    while not done:
        state, reward, done, info = env.step(action)
    
    return env.portfolio_history

def baseline_buy_and_hold(env, stock_idx=4):
    """Baseline: Buy and hold single stock (default: NVDA)"""
    state = env.reset()
    done = False
    action = np.zeros(env.n_stocks)
    action[stock_idx] = 1.0  # 100% in one stock
    
    while not done:
        state, reward, done, info = env.step(action)
    
    return env.portfolio_history

def evaluate_dqn(env, agent):
    """Evaluate trained DQN agent"""
    agent.epsilon = 0.0  # No exploration during evaluation
    state = env.reset()
    done = False
    
    while not done:
        action = agent.select_action(state, training=False)
        state, reward, done, info = env.step(action)
    
    return env.portfolio_history

def plot_comparison(dqn_history, equal_history, buyhold_history, tickers):
    """Plot comparison of strategies"""
    plt.figure(figsize=(12, 6))
    
    days = range(len(dqn_history))
    
    plt.plot(days, dqn_history, label='DQN Agent (Ours)', linewidth=2, color='blue')
    plt.plot(days, equal_history, label='Equal Weight', linewidth=2, color='green', linestyle='--')
    plt.plot(days, buyhold_history, label='Buy & Hold NVDA', linewidth=2, color='orange', linestyle=':')
    
    plt.axhline(y=100000, color='red', linestyle='--', alpha=0.5, label='Initial Investment')
    
    plt.title('Portfolio Strategy Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Days')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add final values as text
    plt.text(len(days)*0.7, max(dqn_history)*0.95, 
             f'DQN Final: ${dqn_history[-1]:,.0f}\n'
             f'Equal Weight: ${equal_history[-1]:,.0f}\n'
             f'Buy&Hold: ${buyhold_history[-1]:,.0f}',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(get_figure_path('strategy_comparison.png'), dpi=300)
    print("✓ Comparison saved to strategy_comparison.png")
    plt.close()

def calculate_metrics(history, initial_value=100000):
    """Calculate performance metrics"""
    final_value = history[-1]
    total_return = (final_value / initial_value - 1) * 100
    
    # Daily returns
    returns = np.diff(history) / history[:-1]
    
    # Sharpe ratio (annualized, assuming 252 trading days)
    sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
    
    # Max drawdown
    peak = np.maximum.accumulate(history)
    drawdown = (history - peak) / peak
    max_drawdown = np.min(drawdown) * 100
    
    return {
        'final_value': final_value,
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown
    }

if __name__ == "__main__":
    print("Loading trained model...")
    
    # Load data
    prices = pd.read_csv('data/stock_prices.csv', index_col=0, parse_dates=True)
    
    # Create environment
    env = PortfolioEnv(prices, initial_balance=100000)
    
    # Load trained DQN agent
    state_size = env.observation_space.shape[0]
    action_size = env.n_stocks
    dqn_agent = DQNAgent(state_size, action_size)
    dqn_agent.load(get_model_path('dqn_portfolio.pth'))
    
    print("\nEvaluating strategies...")
    
    # Run evaluations
    print("  - DQN Agent...")
    dqn_history = evaluate_dqn(env, dqn_agent)
    
    print("  - Equal Weight Baseline...")
    equal_history = baseline_equal_weight(env)
    
    print("  - Buy & Hold NVDA...")
    buyhold_history = baseline_buy_and_hold(env, stock_idx=4)  # NVDA
    
    # Calculate metrics
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)
    
    strategies = {
        'DQN Agent (Ours)': dqn_history,
        'Equal Weight': equal_history,
        'Buy & Hold NVDA': buyhold_history
    }
    
    results = {}
    for name, history in strategies.items():
        metrics = calculate_metrics(history)
        results[name] = metrics
        
        print(f"\n{name}:")
        print(f"  Final Value:    ${metrics['final_value']:,.2f}")
        print(f"  Total Return:   {metrics['total_return']:.2f}%")
        print(f"  Sharpe Ratio:   {metrics['sharpe_ratio']:.3f}")
        print(f"  Max Drawdown:   {metrics['max_drawdown']:.2f}%")
    
    # Determine winner
    print("\n" + "="*60)
    best_return = max(results.items(), key=lambda x: x[1]['total_return'])
    best_sharpe = max(results.items(), key=lambda x: x[1]['sharpe_ratio'])
    
    print(f" Best Return: {best_return[0]} ({best_return[1]['total_return']:.2f}%)")
    print(f" Best Risk-Adjusted: {best_sharpe[0]} (Sharpe: {best_sharpe[1]['sharpe_ratio']:.3f})")
    print("="*60)
    
    # Plot comparison
    plot_comparison(dqn_history, equal_history, buyhold_history, env.tickers)
    
    print("\n✓ Evaluation complete!")