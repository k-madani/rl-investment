"""
Unit tests for RL Investment System.

Tests core components for correctness and robustness.
"""
"""
Unit tests for RL Investment System.

Tests core components for correctness and robustness.
"""

import unittest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.portfolio_env import PortfolioEnv
from agents.dqn_agent import DQNAgent
from utils.error_handler import ActionValidator, FallbackPolicy
from utils.portfolio_tools import ContextualBanditTool

class TestActionValidator(unittest.TestCase):
    """Test action validation"""
    
    def setUp(self):
        self.validator = ActionValidator(n_stocks=5)
    
    def test_valid_action(self):
        action = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        is_valid, msg = self.validator.validate(action)
        self.assertTrue(is_valid)
    
    def test_nan_action(self):
        action = np.array([0.2, np.nan, 0.2, 0.2, 0.4])
        is_valid, msg = self.validator.validate(action)
        self.assertFalse(is_valid)
    
    def test_sanitize_nan(self):
        action = np.array([0.2, np.nan, 0.2, 0.2, 0.4])
        sanitized = self.validator.sanitize(action)
        is_valid, _ = self.validator.validate(sanitized)
        self.assertTrue(is_valid)

class TestDQNAgent(unittest.TestCase):
    """Test DQN agent"""
    
    def setUp(self):
        self.agent = DQNAgent(state_size=51, action_size=5)
    
    def test_action_selection(self):
        state = np.random.randn(51).astype(np.float32)
        action = self.agent.select_action(state, training=False)
        self.assertEqual(action.shape, (5,))
        self.assertAlmostEqual(action.sum(), 1.0, places=5)
    
    def test_memory_push(self):
        state = np.random.randn(51).astype(np.float32)
        action = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        reward = 0.01
        next_state = np.random.randn(51).astype(np.float32)
        done = False
        
        initial_size = len(self.agent.memory)
        self.agent.memory.push(state, action, reward, next_state, done)
        self.assertEqual(len(self.agent.memory), initial_size + 1)

class TestContextualBandit(unittest.TestCase):
    """Test contextual bandit"""
    
    def setUp(self):
        self.bandit = ContextualBanditTool(n_stocks=5)
    
    def test_context_extraction(self):
        prices = np.random.randn(100, 5).cumsum(axis=0) + 100
        context = self.bandit.extract_context(prices, step=50)
        self.assertEqual(context.shape, (5,))
        self.assertFalse(np.isnan(context).any())
    
    def test_stock_selection(self):
        context = np.random.randn(5)
        selected = self.bandit.select_stocks(context, n_select=3)
        self.assertEqual(len(selected), 3)
        self.assertTrue(all(0 <= idx < 5 for idx in selected))

if __name__ == '__main__':
    unittest.main()