"""
Run comprehensive test scenarios.

Tests agent robustness across multiple conditions.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from run_experiment import ExperimentConfig, run_experiment


def create_scenario_suite():
    """Create comprehensive test scenario suite"""
    
    scenarios = []
    
    # Scenario Group 1: Different capital amounts
    for capital in [10000, 25000, 50000, 100000, 250000, 500000]:
        scenarios.append(ExperimentConfig(
            experiment_name=f"capital_{capital//1000}k",
            description=f"Testing with ${capital:,} initial capital",
            initial_balance=capital,
            tickers=['NVDA', 'GOOGL', 'META', 'MSFT', 'AMD'],
            n_episodes=100
        ))
    
    # Scenario Group 2: Different stock universes
    universes = {
        'ai_stocks': ['NVDA', 'GOOGL', 'META', 'MSFT', 'AMD'],
        'tech_giants': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'],
        'semiconductors': ['NVDA', 'AMD', 'INTC', 'QCOM', 'TSM'],
        'cloud': ['MSFT', 'GOOGL', 'AMZN', 'CRM', 'SNOW'],
        'fintech': ['SQ', 'PYPL', 'V', 'MA', 'COIN']
    }
    
    for name, tickers in universes.items():
        scenarios.append(ExperimentConfig(
            experiment_name=f"universe_{name}",
            description=f"{name.replace('_', ' ').title()} sector portfolio",
            initial_balance=100000,
            tickers=tickers,
            n_episodes=100
        ))
    
    # Scenario Group 3: Different portfolio sizes
    for n_stocks in [3, 5, 7, 10]:
        base_tickers = ['NVDA', 'GOOGL', 'META', 'MSFT', 'AMD', 
                       'AAPL', 'AMZN', 'TSLA', 'CRM', 'ORCL']
        scenarios.append(ExperimentConfig(
            experiment_name=f"size_{n_stocks}_stocks",
            description=f"Portfolio with {n_stocks} stocks",
            initial_balance=100000,
            tickers=base_tickers[:n_stocks],
            n_episodes=100
        ))
    
    # Scenario Group 4: Different training intensities
    for episodes in [50, 100, 150, 200]:
        scenarios.append(ExperimentConfig(
            experiment_name=f"training_{episodes}_episodes",
            description=f"Training with {episodes} episodes",
            initial_balance=100000,
            tickers=['NVDA', 'GOOGL', 'META', 'MSFT', 'AMD'],
            n_episodes=episodes
        ))
    
    # Scenario Group 5: Different hyperparameters
    for lr in [0.0005, 0.001, 0.002]:
        scenarios.append(ExperimentConfig(
            experiment_name=f"lr_{str(lr).replace('.', '_')}",
            description=f"Learning rate = {lr}",
            initial_balance=100000,
            tickers=['NVDA', 'GOOGL', 'META', 'MSFT', 'AMD'],
            n_episodes=100,
            learning_rate=lr
        ))
    
    return scenarios


def run_scenario_suite(scenarios, output_dir='results/comprehensive_tests'):
    """
    Run full scenario suite and generate comparison report.
    
    Args:
        scenarios: List of ExperimentConfig objects
        output_dir: Directory to save results
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("COMPREHENSIVE SCENARIO TESTING")
    print("="*70)
    print(f"Total scenarios: {len(scenarios)}")
    print(f"Output directory: {output_path}")
    
    results = []
    
    for i, config in enumerate(scenarios, 1):
        print(f"\n{'='*70}")
        print(f"SCENARIO {i}/{len(scenarios)}: {config.experiment_name}")
        print(f"{'='*70}")
        
        try:
            result = run_experiment(config)
            
            if result:
                results.append({
                    'scenario': config.experiment_name,
                    'description': config.description,
                    'initial_balance': config.initial_balance,
                    'n_stocks': len(config.tickers),
                    'tickers': ', '.join(config.tickers),
                    'n_episodes': config.n_episodes,
                    'final_value': result['metrics']['final_value'],
                    'total_return': result['metrics']['total_return'],
                    'sharpe_ratio': result['metrics']['sharpe_ratio'],
                    'max_drawdown': result['metrics']['max_drawdown'],
                    'status': 'success'
                })
            
        except Exception as e:
            print(f"❌ Scenario failed: {e}")
            results.append({
                'scenario': config.experiment_name,
                'description': config.description,
                'status': 'failed',
                'error': str(e)
            })
    
    # Generate comprehensive report
    generate_comprehensive_report(results, output_path)
    
    return results


def generate_comprehensive_report(results, output_dir):
    """Generate comprehensive analysis report"""
    
    df = pd.DataFrame(results)
    successful = df[df['status'] == 'success'].copy()
    
    # Save raw results
    df.to_csv(output_dir / 'all_scenarios.csv', index=False)
    successful.to_csv(output_dir / 'successful_scenarios.csv', index=False)
    
    print("\n" + "="*70)
    print("COMPREHENSIVE TEST RESULTS")
    print("="*70)
    
    print(f"\nTotal Scenarios: {len(df)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(df) - len(successful)}")
    
    if len(successful) > 0:
        # Summary statistics
        print("\n📊 OVERALL PERFORMANCE STATISTICS")
        print("-" * 70)
        
        metrics = ['total_return', 'sharpe_ratio', 'max_drawdown']
        for metric in metrics:
            if metric in successful.columns:
                values = successful[metric]
                print(f"\n{metric.replace('_', ' ').title()}:")
                print(f"  Mean: {values.mean():.2f}")
                print(f"  Std:  {values.std():.2f}")
                print(f"  Min:  {values.min():.2f}")
                print(f"  Max:  {values.max():.2f}")
        
        # Top performers
        print("\n🏆 TOP 5 PERFORMERS (by return)")
        print("-" * 70)
        top5 = successful.nlargest(5, 'total_return')
        for idx, row in top5.iterrows():
            print(f"{row['scenario']:<30} {row['total_return']:>8.2f}%  "
                  f"Sharpe: {row['sharpe_ratio']:>6.3f}")
        
        # Analysis by category
        print("\n📈 ANALYSIS BY CATEGORY")
        print("-" * 70)
        
        # Capital analysis
        capital_scenarios = successful[successful['scenario'].str.startswith('capital_')]
        if len(capital_scenarios) > 0:
            print("\nBy Initial Capital:")
            for _, row in capital_scenarios.iterrows():
                print(f"  ${row['initial_balance']:>10,}: {row['total_return']:>7.2f}% return")
        
        # Universe analysis
        universe_scenarios = successful[successful['scenario'].str.startswith('universe_')]
        if len(universe_scenarios) > 0:
            print("\nBy Stock Universe:")
            for _, row in universe_scenarios.iterrows():
                print(f"  {row['scenario'][9:]:<15}: {row['total_return']:>7.2f}% return, "
                      f"Sharpe: {row['sharpe_ratio']:.3f}")
        
        # Generate visualizations
        generate_visualizations(successful, output_dir)
    
    print("\n✓ Report saved to:", output_dir)


def generate_visualizations(df, output_dir):
    """Generate visualization plots"""
    
    # Set style
    sns.set_style("whitegrid")
    
    # Plot 1: Returns by category
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Capital analysis
    ax = axes[0, 0]
    capital_df = df[df['scenario'].str.startswith('capital_')].copy()
    if len(capital_df) > 0:
        ax.bar(range(len(capital_df)), capital_df['total_return'])
        ax.set_xticks(range(len(capital_df)))
        ax.set_xticklabels([f"${int(x/1000)}K" for x in capital_df['initial_balance']], 
                          rotation=45)
        ax.set_title('Returns by Initial Capital', fontweight='bold')
        ax.set_ylabel('Total Return (%)')
        ax.grid(True, alpha=0.3)
    
    # Universe analysis
    ax = axes[0, 1]
    universe_df = df[df['scenario'].str.startswith('universe_')].copy()
    if len(universe_df) > 0:
        universe_df['universe'] = universe_df['scenario'].str.replace('universe_', '')
        ax.barh(universe_df['universe'], universe_df['total_return'])
        ax.set_title('Returns by Stock Universe', fontweight='bold')
        ax.set_xlabel('Total Return (%)')
        ax.grid(True, alpha=0.3, axis='x')
    
    # Portfolio size analysis
    ax = axes[1, 0]
    size_df = df[df['scenario'].str.startswith('size_')].copy()
    if len(size_df) > 0:
        ax.scatter(size_df['n_stocks'], size_df['total_return'], s=100, alpha=0.6)
        ax.set_title('Returns vs Portfolio Size', fontweight='bold')
        ax.set_xlabel('Number of Stocks')
        ax.set_ylabel('Total Return (%)')
        ax.grid(True, alpha=0.3)
    
    # Risk-return scatter
    ax = axes[1, 1]
    ax.scatter(df['max_drawdown'].abs(), df['total_return'], 
              s=100, alpha=0.6, c=df['sharpe_ratio'], cmap='viridis')
    ax.set_title('Risk-Return Profile', fontweight='bold')
    ax.set_xlabel('Max Drawdown (%)')
    ax.set_ylabel('Total Return (%)')
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label('Sharpe Ratio')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Visualizations saved to comprehensive_analysis.png")
    plt.close()


if __name__ == "__main__":
    print("Creating comprehensive scenario suite...")
    scenarios = create_scenario_suite()
    
    print(f"\nTotal scenarios to run: {len(scenarios)}")
    print("This will take approximately 3-4 hours...")
    
    confirm = input("\nProceed? (y/N): ").strip().lower()
    
    if confirm == 'y':
        results = run_scenario_suite(scenarios)
        print("\n✓ All scenarios complete!")
    else:
        print("Cancelled.")