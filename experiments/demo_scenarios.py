"""
Quick demo of different investment scenarios.

Runs 4-5 key scenarios to demonstrate system versatility.
"""

from run_experiment import ExperimentConfig, run_experiment


def run_demo():
    """Run demo scenarios"""
    
    print("="*70)
    print("RL-INVESTMENT DEMO SCENARIOS")
    print("="*70)
    print("\nThis demo runs 5 different investment scenarios:")
    print("  1. Small investor ($10K)")
    print("  2. Large investor ($500K)")
    print("  3. Tech giants portfolio")
    print("  4. Semiconductor focus")
    print("  5. Small portfolio (3 stocks)")
    print("\nEstimated time: ~45 minutes")
    
    confirm = input("\nProceed? (Y/n): ").strip().lower()
    if confirm == 'n':
        return
    
    scenarios = [
        ExperimentConfig(
            experiment_name="demo_small_investor",
            description="Small investor with $10,000",
            initial_balance=10000,
            tickers=['NVDA', 'GOOGL', 'META', 'MSFT', 'AMD'],
            n_episodes=80
        ),
        ExperimentConfig(
            experiment_name="demo_large_investor",
            description="Large investor with $500,000",
            initial_balance=500000,
            tickers=['NVDA', 'GOOGL', 'META', 'MSFT', 'AMD'],
            n_episodes=80
        ),
        ExperimentConfig(
            experiment_name="demo_tech_giants",
            description="Tech giants portfolio",
            initial_balance=100000,
            tickers=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'],
            n_episodes=80
        ),
        ExperimentConfig(
            experiment_name="demo_semiconductors",
            description="Semiconductor companies",
            initial_balance=100000,
            tickers=['NVDA', 'AMD', 'INTC', 'QCOM', 'TSM'],
            n_episodes=80
        ),
        ExperimentConfig(
            experiment_name="demo_small_portfolio",
            description="Small portfolio (3 stocks)",
            initial_balance=100000,
            tickers=['NVDA', 'GOOGL', 'MSFT'],
            n_episodes=80
        )
    ]
    
    results = []
    for i, config in enumerate(scenarios, 1):
        print(f"\n{'='*70}")
        print(f"DEMO SCENARIO {i}/{len(scenarios)}")
        print(f"{'='*70}")
        result = run_experiment(config)
        if result:
            results.append(result)
    
    # Summary
    print("\n" + "="*70)
    print("DEMO RESULTS SUMMARY")
    print("="*70)
    
    for i, result in enumerate(results, 1):
        config = result['config']
        metrics = result['metrics']
        print(f"\n{i}. {config['experiment_name']}")
        print(f"   Capital: ${config['initial_balance']:,}")
        print(f"   Stocks: {', '.join(config['tickers'])}")
        print(f"   Return: {metrics['total_return']:.2f}%")
        print(f"   Sharpe: {metrics['sharpe_ratio']:.3f}")
    
    print("\n✓ Demo complete! Check results/experiments/ for details")


if __name__ == "__main__":
    run_demo()