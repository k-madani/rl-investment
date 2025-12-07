import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from agents.portfolio_env import PortfolioEnv
from agents.dqn_agent import DQNAgent
from utils.paths import get_figure_path, get_model_path

def random_agent(env, n_runs=10):
    """Baseline: Random portfolio allocation"""
    all_histories = []
    
    for run in range(n_runs):
        state = env.reset()
        done = False
        
        while not done:
            action = np.random.dirichlet(np.ones(env.n_stocks))
            state, reward, done, info = env.step(action)
        
        all_histories.append(env.portfolio_history)
    
    min_length = min(len(h) for h in all_histories)
    truncated = [h[:min_length] for h in all_histories]
    avg_history = np.mean(truncated, axis=0)
    
    return avg_history, all_histories

def untrained_dqn_agent(env, state_size, action_size, n_runs=10):
    """Baseline: Untrained DQN"""
    all_histories = []
    
    for run in range(n_runs):
        untrained = DQNAgent(state_size, action_size)
        untrained.epsilon = 0.0
        
        state = env.reset()
        done = False
        
        while not done:
            action = untrained.select_action(state, training=False)
            state, reward, done, info = env.step(action)
        
        all_histories.append(env.portfolio_history)
    
    min_length = min(len(h) for h in all_histories)
    truncated = [h[:min_length] for h in all_histories]
    avg_history = np.mean(truncated, axis=0)
    
    return avg_history, all_histories

def plot_learning_comparison(trained_history, random_history, untrained_history, 
                            random_runs, untrained_runs):
    """Compare trained vs untrained vs random agents"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Convert to numpy arrays
    trained_history = np.array(trained_history)
    random_history = np.array(random_history)
    untrained_history = np.array(untrained_history)
    
    days = range(len(trained_history))
    
    # Plot 1: Main comparison with confidence bands
    ax1.plot(days, trained_history, label='Trained DQN', linewidth=3, color='blue', zorder=3)
    ax1.plot(days, random_history, label='Random Agent (avg)', linewidth=2, 
             color='red', linestyle='--', alpha=0.7)
    ax1.plot(days, untrained_history, label='Untrained DQN (avg)', linewidth=2,
             color='orange', linestyle=':', alpha=0.7)
    
    # Add confidence bands for random agent
    random_array = np.array([h[:len(days)] for h in random_runs])
    random_std = np.std(random_array, axis=0)
    ax1.fill_between(days, 
                     random_history - random_std,
                     random_history + random_std,
                     alpha=0.2, color='red', label='Random ±1σ')
    
    ax1.axhline(y=100000, color='gray', linestyle='--', alpha=0.5, label='Initial Value')
    
    ax1.set_xlabel('Trading Days', fontsize=12)
    ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax1.set_title('Learning Impact: Trained vs Untrained vs Random', 
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    
    # Plot 2: Cumulative returns comparison
    trained_returns = (trained_history / trained_history[0] - 1) * 100
    random_returns = (random_history / random_history[0] - 1) * 100
    untrained_returns = (untrained_history / untrained_history[0] - 1) * 100
    
    ax2.plot(days, trained_returns, label='Trained DQN', linewidth=3, color='blue')
    ax2.plot(days, random_returns, label='Random Agent', linewidth=2, 
             color='red', linestyle='--', alpha=0.7)
    ax2.plot(days, untrained_returns, label='Untrained DQN', linewidth=2,
             color='orange', linestyle=':', alpha=0.7)
    
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Trading Days', fontsize=12)
    ax2.set_ylabel('Cumulative Return (%)', fontsize=12)
    ax2.set_title('Cumulative Returns Comparison', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(get_figure_path('learning_impact_comparison.png'), dpi=300, bbox_inches='tight')
    print("✓ Saved learning_impact_comparison.png")
    plt.close()

def calculate_improvement_metrics(trained_history, random_history, untrained_history):
    """Calculate how much learning improved performance"""
    
    # Convert to numpy arrays
    trained_history = np.array(trained_history)
    random_history = np.array(random_history)
    untrained_history = np.array(untrained_history)
    
    print("\n" + "="*60)
    print("LEARNING IMPACT ANALYSIS")
    print("="*60)
    
    trained_final = trained_history[-1]
    random_final = random_history[-1]
    untrained_final = untrained_history[-1]
    
    trained_return = (trained_final / 100000 - 1) * 100
    random_return = (random_final / 100000 - 1) * 100
    untrained_return = (untrained_final / 100000 - 1) * 100
    
    print(f"\n📊 Final Portfolio Values:")
    print(f"  Trained DQN:    ${trained_final:,.2f} ({trained_return:+.2f}%)")
    print(f"  Random Agent:   ${random_final:,.2f} ({random_return:+.2f}%)")
    print(f"  Untrained DQN:  ${untrained_final:,.2f} ({untrained_return:+.2f}%)")
    
    print(f"\n🚀 Improvement from Learning:")
    improvement_vs_random = ((trained_final / random_final) - 1) * 100
    improvement_vs_untrained = ((trained_final / untrained_final) - 1) * 100
    
    print(f"  vs Random:      +{improvement_vs_random:.2f}%")
    print(f"  vs Untrained:   +{improvement_vs_untrained:.2f}%")
    
    # Calculate Sharpe ratios
    trained_returns = np.diff(trained_history) / (trained_history[:-1] + 1e-8)
    random_returns = np.diff(random_history) / (random_history[:-1] + 1e-8)
    
    trained_sharpe = np.mean(trained_returns) / (np.std(trained_returns) + 1e-8) * np.sqrt(252)
    random_sharpe = np.mean(random_returns) / (np.std(random_returns) + 1e-8) * np.sqrt(252)
    
    print(f"\n📈 Risk-Adjusted Performance (Sharpe Ratio):")
    print(f"  Trained DQN:    {trained_sharpe:.3f}")
    print(f"  Random Agent:   {random_sharpe:.3f}")
    print(f"  Improvement:    +{((trained_sharpe / random_sharpe) - 1) * 100:.1f}%")
    
    print("="*60 + "\n")

if __name__ == "__main__":
    print("Comparing trained agent vs baselines...\n")
    
    # Load data and trained model
    prices = pd.read_csv('data/stock_prices.csv', index_col=0, parse_dates=True)
    env = PortfolioEnv(prices, initial_balance=100000)
    
    state_size = env.observation_space.shape[0]
    action_size = env.n_stocks
    
    # Load trained agent
    print("Loading trained DQN agent...")
    trained_agent = DQNAgent(state_size, action_size)
    trained_agent.load(get_model_path('dqn_portfolio.pth'))
    trained_agent.epsilon = 0.0
    
    # Evaluate trained agent
    print("Evaluating trained agent...")
    state = env.reset()
    done = False
    while not done:
        action = trained_agent.select_action(state, training=False)
        state, reward, done, info = env.step(action)
    trained_history = env.portfolio_history.copy()
    
    # Evaluate random agent
    print("Evaluating random agent (10 runs)...")
    random_history, random_runs = random_agent(env, n_runs=10)
    
    # Evaluate untrained DQN
    print("Evaluating untrained DQN (10 runs)...")
    untrained_history, untrained_runs = untrained_dqn_agent(env, state_size, action_size, n_runs=10)
    
    # Plot comparison
    plot_learning_comparison(trained_history, random_history, untrained_history,
                           random_runs, untrained_runs)
    
    # Calculate metrics
    calculate_improvement_metrics(trained_history, random_history, untrained_history)
    
    print("✓ Comparison complete! Check results/figures/")