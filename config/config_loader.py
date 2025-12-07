"""
Configuration loader for RL Investment System.

Loads and validates configuration from YAML file.
"""

import yaml
import os
from pathlib import Path

class Config:
    """Configuration container"""
    
    def __init__(self, config_dict):
        self._config = config_dict
        
    def __getattr__(self, name):
        if name in self._config:
            value = self._config[name]
            if isinstance(value, dict):
                return Config(value)
            return value
        raise AttributeError(f"Config has no attribute '{name}'")
    
    def __getitem__(self, key):
        return self._config[key]
    
    def get(self, key, default=None):
        return self._config.get(key, default)
    
    def to_dict(self):
        return self._config


def load_config(config_path='config/config.yaml'):
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Config: Configuration object
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return Config(config_dict)


def get_default_config():
    """Get default configuration"""
    return Config({
        'environment': {
            'initial_balance': 100000,
            'window_size': 10,
            'transaction_cost': 0.001
        },
        'dqn': {
            'learning_rate': 0.001,
            'gamma': 0.95,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.995,
            'batch_size': 64,
            'replay_buffer_size': 10000,
            'hidden_size': 128
        },
        'training': {
            'n_episodes': 100,
            'target_update_freq': 10
        }
    })


if __name__ == "__main__":
    print("Testing config loader...")
    
    try:
        config = load_config()
        print(f"Initial balance: ${config.environment.initial_balance:,}")
        print(f"Learning rate: {config.dqn.learning_rate}")
        print(f"Episodes: {config.training.n_episodes}")
        print("✓ Config loaded successfully!")
    except FileNotFoundError:
        print("Using default config...")
        config = get_default_config()
        print(f"Initial balance: ${config.environment.initial_balance:,}")