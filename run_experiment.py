"""
Interactive experiment runner for RL-Investment system.

Allows users to configure and run custom investment scenarios.
"""

import argparse
import yaml
from datetime import datetime, timedelta
from pathlib import Path
import sys

from data.fetch_data import MarketDataFetcher
from agents.portfolio_env import PortfolioEnv
from agents.dqn_agent import DQNAgent
from utils.portfolio_tools import ContextualBanditTool
from utils.paths import get_model_path, get_figure_path
from train import train_dqn, plot_results
from evaluate import evaluate_dqn, calculate_metrics


class ExperimentConfig:
    """Configuration for a single experiment"""
    
    def __init__(self, **kwargs):
        # Environment settings
        self.tickers = kwargs.get('tickers', ['NVDA', 'GOOGL', 'META', 'MSFT', 'AMD'])
        self.initial_balance = kwargs.get('initial_balance', 100000)
        self.transaction_cost = kwargs.get('transaction_cost', 0.001)
        self.window_size = kwargs.get('window_size', 10)
        
        # Training settings
        self.n_episodes = kwargs.get('n_episodes', 100)
        self.learning_rate = kwargs.get('learning_rate', 0.001)
        self.gamma = kwargs.get('gamma', 0.95)
        self.epsilon_start = kwargs.get('epsilon_start', 1.0)
        self.epsilon_end = kwargs.get('epsilon_end', 0.01)
        self.epsilon_decay = kwargs.get('epsilon_decay', 0.995)
        
        # Data settings
        self.start_date = kwargs.get('start_date', None)
        self.end_date = kwargs.get('end_date', None)
        
        # Experiment metadata
        self.experiment_name = kwargs.get('experiment_name', 'default')
        self.description = kwargs.get('description', '')
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'tickers': self.tickers,
            'initial_balance': self.initial_balance,
            'transaction_cost': self.transaction_cost,
            'window_size': self.window_size,
            'n_episodes': self.n_episodes,
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'epsilon_start': self.epsilon_start,
            'epsilon_end': self.epsilon_end,
            'epsilon_decay': self.epsilon_decay,
            'start_date': str(self.start_date) if self.start_date else None,
            'end_date': str(self.end_date) if self.end_date else None,
            'experiment_name': self.experiment_name,
            'description': self.description
        }
    
    @classmethod
    def from_dict(cls, config_dict):
        """Create from dictionary"""
        return cls(**config_dict)
    
    def save(self, filepath):
        """Save configuration to YAML"""
        with open(filepath, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    @classmethod
    def load(cls, filepath):
        """Load configuration from YAML"""
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)


def interactive_setup():
    """Interactive CLI for experiment setup"""
    
    print("="*70)
    print("RL-INVESTMENT EXPERIMENT CONFIGURATOR")
    print("="*70)
    
    config = {}
    
    # Experiment name
    print("\n📝 EXPERIMENT SETUP")
    config['experiment_name'] = input("Experiment name (default: 'experiment'): ").strip() or 'experiment'
    config['description'] = input("Description (optional): ").strip()
    
    # Initial capital
    print("\n💰 INITIAL CAPITAL")
    print("Choose initial investment amount:")
    print("  1. $10,000 (Small investor)")
    print("  2. $50,000 (Medium investor)")
    print("  3. $100,000 (Large investor)")
    print("  4. $500,000 (High net worth)")
    print("  5. Custom amount")
    
    capital_choice = input("Select option (1-5) [default: 3]: ").strip() or '3'
    
    capital_map = {
        '1': 10000,
        '2': 50000,
        '3': 100000,
        '4': 500000
    }
    
    if capital_choice in capital_map:
        config['initial_balance'] = capital_map[capital_choice]
    else:
        custom = input("Enter custom amount: $").strip()
        config['initial_balance'] = int(custom) if custom else 100000
    
    print(f"✓ Initial balance set to: ${config['initial_balance']:,}")
    
    # Stock selection
    print("\n📊 STOCK UNIVERSE")
    print("Choose stock portfolio:")
    print("  1. Original AI stocks (NVDA, GOOGL, META, MSFT, AMD)")
    print("  2. Tech giants (AAPL, MSFT, GOOGL, AMZN, META)")
    print("  3. Semiconductors (NVDA, AMD, INTC, QCOM, TSM)")
    print("  4. Cloud companies (MSFT, GOOGL, AMZN, CRM, SNOW)")
    print("  5. Custom tickers")
    
    stock_choice = input("Select option (1-5) [default: 1]: ").strip() or '1'
    
    stock_map = {
        '1': ['NVDA', 'GOOGL', 'META', 'MSFT', 'AMD'],
        '2': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'],
        '3': ['NVDA', 'AMD', 'INTC', 'QCOM', 'TSM'],
        '4': ['MSFT', 'GOOGL', 'AMZN', 'CRM', 'SNOW']
    }
    
    if stock_choice in stock_map:
        config['tickers'] = stock_map[stock_choice]
    else:
        custom_tickers = input("Enter tickers (comma-separated): ").strip().upper()
        config['tickers'] = [t.strip() for t in custom_tickers.split(',')]
    
    print(f"✓ Portfolio: {', '.join(config['tickers'])}")
    
    # Time period
    print("\n📅 TIME PERIOD")
    print("Choose data period:")
    print("  1. Last 6 months")
    print("  2. Last 1 year")
    print("  3. Last 2 years (default)")
    print("  4. Custom dates")
    
    time_choice = input("Select option (1-4) [default: 3]: ").strip() or '3'
    
    end_date = datetime.now()
    
    time_map = {
        '1': end_date - timedelta(days=180),
        '2': end_date - timedelta(days=365),
        '3': end_date - timedelta(days=730)
    }
    
    if time_choice in time_map:
        config['start_date'] = time_map[time_choice]
        config['end_date'] = end_date
    else:
        start_str = input("Start date (YYYY-MM-DD): ").strip()
        end_str = input("End date (YYYY-MM-DD, or blank for today): ").strip()
        config['start_date'] = datetime.strptime(start_str, '%Y-%m-%d') if start_str else time_map['3']
        config['end_date'] = datetime.strptime(end_str, '%Y-%m-%d') if end_str else end_date
    
    print(f"✓ Period: {config['start_date'].strftime('%Y-%m-%d')} to {config['end_date'].strftime('%Y-%m-%d')}")
    
    # Training parameters
    print("\n🎓 TRAINING SETTINGS")
    print("Choose training intensity:")
    print("  1. Quick test (50 episodes, ~5 min)")
    print("  2. Standard (100 episodes, ~10 min)")
    print("  3. Thorough (200 episodes, ~20 min)")
    print("  4. Custom")
    
    training_choice = input("Select option (1-4) [default: 2]: ").strip() or '2'
    
    training_map = {
        '1': 50,
        '2': 100,
        '3': 200
    }
    
    if training_choice in training_map:
        config['n_episodes'] = training_map[training_choice]
    else:
        custom_episodes = input("Enter number of episodes: ").strip()
        config['n_episodes'] = int(custom_episodes) if custom_episodes else 100
    
    print(f"✓ Training episodes: {config['n_episodes']}")
    
    # Advanced settings
    advanced = input("\n⚙️  Configure advanced settings? (y/N): ").strip().lower()
    
    if advanced == 'y':
        print("\nAdvanced Settings:")
        config['learning_rate'] = float(input(f"Learning rate [0.001]: ").strip() or 0.001)
        config['gamma'] = float(input(f"Discount factor (gamma) [0.95]: ").strip() or 0.95)
        config['transaction_cost'] = float(input(f"Transaction cost [0.001]: ").strip() or 0.001)
        config['window_size'] = int(input(f"Window size [10]: ").strip() or 10)
    else:
        # Use defaults
        config['learning_rate'] = 0.001
        config['gamma'] = 0.95
        config['transaction_cost'] = 0.001
        config['window_size'] = 10
    
    # Summary
    print("\n" + "="*70)
    print("EXPERIMENT CONFIGURATION SUMMARY")
    print("="*70)
    print(f"Name: {config['experiment_name']}")
    if config['description']:
        print(f"Description: {config['description']}")
    print(f"Initial Capital: ${config['initial_balance']:,}")
    print(f"Stocks: {', '.join(config['tickers'])}")
    print(f"Period: {config['start_date'].strftime('%Y-%m-%d')} to {config['end_date'].strftime('%Y-%m-%d')}")
    print(f"Episodes: {config['n_episodes']}")
    print(f"Learning Rate: {config['learning_rate']}")
    print(f"Gamma: {config['gamma']}")
    print("="*70)
    
    confirm = input("\nProceed with this configuration? (Y/n): ").strip().lower()
    
    if confirm == 'n':
        print("Experiment cancelled.")
        sys.exit(0)
    
    return ExperimentConfig(**config)


def run_experiment(config):
    """
    Run a complete experiment with given configuration.
    
    Args:
        config: ExperimentConfig object
    """
    
    print("\n" + "="*70)
    print(f"STARTING EXPERIMENT: {config.experiment_name}")
    print("="*70)
    
    # Create experiment directory
    exp_dir = Path(f"results/experiments/{config.experiment_name}")
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config.save(exp_dir / "config.yaml")
    print(f"✓ Configuration saved to {exp_dir / 'config.yaml'}")
    
    # Step 1: Fetch data
    print("\n📥 STEP 1: Fetching market data...")
    fetcher = MarketDataFetcher(
        config.tickers,
        config.start_date,
        config.end_date
    )
    prices = fetcher.fetch_data()
    
    if prices is None or len(prices) < config.window_size:
        print("❌ Error: Insufficient data. Try a longer time period.")
        return None
    
    print(f"✓ Fetched {len(prices)} days of data")
    
    # Save data
    prices.to_csv(exp_dir / "prices.csv")
    
    # Step 2: Create environment
    print("\n🎮 STEP 2: Creating environment...")
    env = PortfolioEnv(
        prices,
        initial_balance=config.initial_balance,
        window_size=config.window_size,
        transaction_cost=config.transaction_cost
    )
    print(f"✓ Environment created: {env.n_stocks} stocks, {len(prices)} days")
    
    # Step 3: Create agents
    print("\n🤖 STEP 3: Initializing agents...")
    state_size = env.observation_space.shape[0]
    action_size = env.n_stocks
    
    dqn_agent = DQNAgent(
        state_size,
        action_size,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        epsilon_start=config.epsilon_start,
        epsilon_end=config.epsilon_end,
        epsilon_decay=config.epsilon_decay
    )
    
    bandit = ContextualBanditTool(n_stocks=action_size)
    
    print(f"✓ DQN Agent: state_size={state_size}, action_size={action_size}")
    print(f"✓ Contextual Bandit: {action_size} arms")
    
    # Step 4: Train
    print("\n🎓 STEP 4: Training agents...")
    print(f"Episodes: {config.n_episodes}")
    print("This may take several minutes...\n")
    
    episode_rewards, portfolio_values, losses = train_dqn(
        env, dqn_agent, bandit, n_episodes=config.n_episodes
    )
    
    # Save model
    model_path = exp_dir / "dqn_model.pth"
    dqn_agent.save(str(model_path))
    print(f"✓ Model saved to {model_path}")
    
    # Save training history
    import pandas as pd
    history_df = pd.DataFrame({
        'episode': range(len(episode_rewards)),
        'reward': episode_rewards,
        'portfolio_value': portfolio_values,
        'loss': losses if len(losses) == len(episode_rewards) else [None] * len(episode_rewards)
    })
    history_df.to_csv(exp_dir / "training_history.csv", index=False)
    
    # Step 5: Evaluate
    print("\n📊 STEP 5: Evaluating performance...")
    
    # Reset environment for evaluation
    env = PortfolioEnv(prices, initial_balance=config.initial_balance)
    dqn_agent.epsilon = 0.0
    
    eval_history = evaluate_dqn(env, dqn_agent)
    metrics = calculate_metrics(eval_history, config.initial_balance)
    
    # Step 6: Generate report
    print("\n" + "="*70)
    print("EXPERIMENT RESULTS")
    print("="*70)
    print(f"\nInitial Balance: ${config.initial_balance:,}")
    print(f"Final Value: ${metrics['final_value']:,.2f}")
    print(f"Total Return: {metrics['total_return']:.2f}%")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
    print("="*70)
    
    # Save results
    results = {
        'config': config.to_dict(),
        'metrics': metrics,
        'final_portfolio_value': metrics['final_value'],
        'total_return_pct': metrics['total_return']
    }
    
    with open(exp_dir / "results.yaml", 'w') as f:
        yaml.dump(results, f, default_flow_style=False)
    
    print(f"\n✓ Complete results saved to {exp_dir}")
    print(f"✓ View detailed results: {exp_dir / 'results.yaml'}")
    
    return results


def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(
        description='RL-Investment Experiment Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python run_experiment.py
  
  # Run with config file
  python run_experiment.py --config experiments/my_config.yaml
  
  # Quick test
  python run_experiment.py --quick
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration YAML file'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick test with defaults'
    )
    
    parser.add_argument(
        '--batch',
        type=str,
        help='Run batch of experiments from directory'
    )
    
    args = parser.parse_args()
    
    # Determine mode
    if args.config:
        # Load from config file
        print(f"Loading configuration from {args.config}...")
        config = ExperimentConfig.load(args.config)
    
    elif args.quick:
        # Quick test mode
        print("Quick test mode - using defaults")
        config = ExperimentConfig(
            experiment_name='quick_test',
            n_episodes=50
        )
    
    elif args.batch:
        # Batch mode
        print(f"Batch mode: {args.batch}")
        batch_dir = Path(args.batch)
        config_files = list(batch_dir.glob("*.yaml"))
        
        print(f"Found {len(config_files)} configurations")
        
        for config_file in config_files:
            print(f"\n{'='*70}")
            print(f"Running: {config_file.name}")
            config = ExperimentConfig.load(config_file)
            run_experiment(config)
        
        return
    
    else:
        # Interactive mode
        config = interactive_setup()
    
    # Run single experiment
    run_experiment(config)


if __name__ == "__main__":
    main()