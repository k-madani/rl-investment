import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from agents.portfolio_env import PortfolioEnv
from agents.dqn_agent import DQNAgent
from utils.paths import get_figure_path, get_model_path

def analyze_portfolio_decisions(env, agent, save_prefix='analysis'):
    """Analyze how agent allocates portfolio over time"""
    agent.epsilon = 0.0  # No exploration
    state = env.reset()
    done = False
    
    # Track decisions
    allocations = []
    portfolio_values = []
    rewards = []
    
    step = 0
    while not done:
        action = agent.select_action(state, training=False)
        allocations.append(action.copy())
        
        state, reward, done, info = env.step(action)
        portfolio_values.append(info['portfolio_value'])
        rewards.append(reward)
        step += 1
    
    allocations = np.array(allocations)
    
    # Plot 1: Portfolio allocation over time
    fig, ax = plt.subplots(figsize=(14, 6))
    
    days = range(len(allocations))
    bottom = np.zeros(len(allocations))
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
    
    for i, ticker in enumerate(env.tickers):
        ax.fill_between(days, bottom, bottom + allocations[:, i], 
                        label=ticker, alpha=0.8, color=colors[i])
        bottom += allocations[:, i]
    
    ax.set_xlabel('Trading Days', fontsize=12)
    ax.set_ylabel('Portfolio Allocation', fontsize=12)
    ax.set_title('DQN Agent Portfolio Allocation Over Time', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(get_figure_path(f'{save_prefix}_allocation_over_time.png'), dpi=300)
    print(f"✓ Saved {save_prefix}_allocation_over_time.png")
    plt.close()
    
    # Plot 2: Average allocation per stock
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    avg_allocations = allocations.mean(axis=0)
    ax1.bar(env.tickers, avg_allocations, color=colors)
    ax1.set_ylabel('Average Allocation', fontsize=12)
    ax1.set_title('Average Portfolio Weights', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add percentage labels
    for i, v in enumerate(avg_allocations):
        ax1.text(i, v + 0.01, f'{v*100:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Allocation volatility
    allocation_changes = np.abs(np.diff(allocations, axis=0)).sum(axis=1)
    ax2.plot(allocation_changes, color='purple', linewidth=1.5)
    ax2.set_xlabel('Trading Days', fontsize=12)
    ax2.set_ylabel('Total Weight Change', fontsize=12)
    ax2.set_title('Portfolio Rebalancing Activity', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(get_figure_path(f'{save_prefix}_allocation_stats.png'), dpi=300)
    print(f"✓ Saved {save_prefix}_allocation_stats.png")
    plt.close()
    
    # Plot 4: Correlation between allocations and returns
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Calculate stock returns
    returns = np.diff(env.prices[env.window_size:env.window_size+len(allocations)], axis=0) / \
              (env.prices[env.window_size:env.window_size+len(allocations)-1] + 1e-8)
    
    # Correlation matrix
    correlation_data = np.column_stack([allocations[:-1], returns])
    df_corr = pd.DataFrame(
        correlation_data,
        columns=[f'{t}_alloc' for t in env.tickers] + [f'{t}_return' for t in env.tickers]
    )
    
    # Show correlation between allocations and next-day returns
    corr_matrix = np.zeros((len(env.tickers), len(env.tickers)))
    for i, ticker in enumerate(env.tickers):
        for j, ticker2 in enumerate(env.tickers):
            corr_matrix[i, j] = df_corr[f'{ticker}_alloc'].corr(df_corr[f'{ticker2}_return'])
    
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
                xticklabels=env.tickers, yticklabels=env.tickers,
                cbar_kws={'label': 'Correlation'}, ax=ax)
    ax.set_title('Allocation vs Next-Day Returns Correlation', fontsize=14, fontweight='bold')
    ax.set_xlabel('Stock Returns', fontsize=12)
    ax.set_ylabel('Allocation Decision', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(get_figure_path(f'{save_prefix}_allocation_return_correlation.png'), dpi=300)
    print(f"✓ Saved {save_prefix}_allocation_return_correlation.png")
    plt.close()
    
    return allocations, portfolio_values, avg_allocations

def compare_early_vs_late_learning(env, agent):
    """Compare agent behavior early vs late in training"""
    agent.epsilon = 0.0
    state = env.reset()
    done = False
    
    early_allocations = []
    early_values = []
    
    # Simulate first 50 days
    step = 0
    while not done and step < 50:
        action = agent.select_action(state, training=False)
        early_allocations.append(action.copy())
        state, reward, done, info = env.step(action)
        early_values.append(info['portfolio_value'])
        step += 1
    
    # Reset and simulate last 50 days
    state = env.reset()
    done = False
    step = 0
    
    # Fast forward to near end
    while step < len(env.prices) - 60 and not done:
        action = agent.select_action(state, training=False)
        state, reward, done, info = env.step(action)
        step += 1
    
    late_allocations = []
    late_values = []
    
    while not done:
        action = agent.select_action(state, training=False)
        late_allocations.append(action.copy())
        state, reward, done, info = env.step(action)
        late_values.append(info['portfolio_value'])
        step += 1
    
    early_allocations = np.array(early_allocations)
    late_allocations = np.array(late_allocations)
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
    
    # Early period allocation
    ax = axes[0, 0]
    bottom = np.zeros(len(early_allocations))
    for i, ticker in enumerate(env.tickers):
        ax.fill_between(range(len(early_allocations)), bottom, 
                        bottom + early_allocations[:, i],
                        label=ticker, alpha=0.8, color=colors[i])
        bottom += early_allocations[:, i]
    ax.set_title('Portfolio Allocation - First 50 Days', fontweight='bold')
    ax.set_ylabel('Allocation')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Late period allocation
    ax = axes[0, 1]
    bottom = np.zeros(len(late_allocations))
    for i, ticker in enumerate(env.tickers):
        ax.fill_between(range(len(late_allocations)), bottom,
                        bottom + late_allocations[:, i],
                        label=ticker, alpha=0.8, color=colors[i])
        bottom += late_allocations[:, i]
    ax.set_title('Portfolio Allocation - Last 50 Days', fontweight='bold')
    ax.set_ylabel('Allocation')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Average allocations comparison
    ax = axes[1, 0]
    early_avg = early_allocations.mean(axis=0)
    late_avg = late_allocations.mean(axis=0)
    
    x = np.arange(len(env.tickers))
    width = 0.35
    
    ax.bar(x - width/2, early_avg, width, label='First 50 Days', color='lightblue')
    ax.bar(x + width/2, late_avg, width, label='Last 50 Days', color='darkblue')
    ax.set_ylabel('Average Allocation')
    ax.set_title('Average Allocation Comparison', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(env.tickers)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Volatility comparison
    ax = axes[1, 1]
    early_volatility = np.std(early_allocations, axis=0)
    late_volatility = np.std(late_allocations, axis=0)
    
    ax.bar(x - width/2, early_volatility, width, label='First 50 Days', color='lightcoral')
    ax.bar(x + width/2, late_volatility, width, label='Last 50 Days', color='darkred')
    ax.set_ylabel('Allocation Std Dev')
    ax.set_title('Allocation Stability', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(env.tickers)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(get_figure_path('analysis_early_vs_late.png'), dpi=300)
    print("✓ Saved analysis_early_vs_late.png")
    plt.close()

def print_agent_insights(allocations, tickers):
    """Print key insights about agent's learned strategy"""
    print("\n" + "="*60)
    print("AGENT LEARNING INSIGHTS")
    print("="*60)
    
    avg_alloc = allocations.mean(axis=0)
    std_alloc = allocations.std(axis=0)
    
    print("\n📊 Average Portfolio Composition:")
    for i, ticker in enumerate(tickers):
        print(f"  {ticker}: {avg_alloc[i]*100:.1f}% (±{std_alloc[i]*100:.1f}%)")
    
    # Find preferred stock
    preferred_idx = np.argmax(avg_alloc)
    print(f"\n⭐ Most Favored Stock: {tickers[preferred_idx]} ({avg_alloc[preferred_idx]*100:.1f}%)")
    
    # Find most stable allocation
    stable_idx = np.argmin(std_alloc)
    print(f"🎯 Most Stable Allocation: {tickers[stable_idx]} (σ={std_alloc[stable_idx]*100:.1f}%)")
    
    # Calculate rebalancing frequency
    changes = np.abs(np.diff(allocations, axis=0)).sum(axis=1)
    avg_change = changes.mean()
    print(f"\n🔄 Average Daily Rebalancing: {avg_change*100:.2f}% of portfolio")
    
    # Diversification score
    herfindahl = (avg_alloc ** 2).sum()
    diversification = 1 / herfindahl
    print(f"📈 Diversification Score: {diversification:.2f} (max={len(tickers)})")
    
    print("="*60 + "\n")

if __name__ == "__main__":
    print("Loading model and analyzing agent behavior...\n")
    
    # Load data and model
    prices = pd.read_csv('data/stock_prices.csv', index_col=0, parse_dates=True)
    env = PortfolioEnv(prices, initial_balance=100000)
    
    state_size = env.observation_space.shape[0]
    action_size = env.n_stocks
    agent = DQNAgent(state_size, action_size)
    agent.load(get_model_path('dqn_portfolio.pth'))
    
    # Analyze portfolio decisions
    allocations, values, avg_alloc = analyze_portfolio_decisions(env, agent)
    
    # Print insights
    print_agent_insights(allocations, env.tickers)
    
    # Compare early vs late learning
    compare_early_vs_late_learning(env, agent)
    
    print("✓ Analysis complete! Check results/figures/ for generated images.")