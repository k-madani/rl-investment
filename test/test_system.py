"""
Comprehensive test suite for RL Investment System.

Tests:
1. Unit Tests - Individual component testing
2. Integration Tests - Full system testing  
3. Robustness Tests - Varied environment testing
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.portfolio_env import PortfolioEnv
from agents.dqn_agent import DQNAgent
from utils.error_handler import ActionValidator, FallbackPolicy, CircuitBreaker
from utils.portfolio_tools import ContextualBanditTool, MarketDataTool, RiskAnalysisTool
from utils.paths import get_model_path


# ============================================================================
# UNIT TESTS - Individual Component Testing
# ============================================================================

class TestActionValidator(unittest.TestCase):
    """Test action validation"""
    
    def setUp(self):
        self.validator = ActionValidator(n_stocks=5)
    
    def test_valid_action(self):
        """Test validation of correct action"""
        action = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        is_valid, msg = self.validator.validate(action)
        self.assertTrue(is_valid)
    
    def test_nan_action(self):
        """Test detection of NaN values"""
        action = np.array([0.2, np.nan, 0.2, 0.2, 0.4])
        is_valid, msg = self.validator.validate(action)
        self.assertFalse(is_valid)
        self.assertIn("NaN", msg)
    
    def test_sanitize_nan(self):
        """Test sanitization of invalid action"""
        action = np.array([0.2, np.nan, 0.2, 0.2, 0.4])
        sanitized = self.validator.sanitize(action)
        is_valid, _ = self.validator.validate(sanitized)
        self.assertTrue(is_valid)
    
    def test_negative_weights(self):
        """Test rejection of negative weights"""
        action = np.array([0.3, -0.1, 0.3, 0.3, 0.2])
        is_valid, msg = self.validator.validate(action)
        self.assertFalse(is_valid)
    
    def test_sum_not_one(self):
        """Test detection when weights don't sum to 1"""
        action = np.array([0.3, 0.3, 0.3, 0.3, 0.3])  # Sum = 1.5
        is_valid, msg = self.validator.validate(action)
        self.assertFalse(is_valid)


class TestDQNAgent(unittest.TestCase):
    """Test DQN agent functionality"""
    
    def setUp(self):
        self.agent = DQNAgent(state_size=51, action_size=5)
    
    def test_agent_initialization(self):
        """Test agent initializes correctly"""
        self.assertEqual(self.agent.state_size, 51)
        self.assertEqual(self.agent.action_size, 5)
        self.assertIsNotNone(self.agent.policy_net)
        self.assertIsNotNone(self.agent.target_net)
    
    def test_action_selection(self):
        """Test action selection produces valid portfolio weights"""
        state = np.random.randn(51).astype(np.float32)
        action = self.agent.select_action(state, training=False)
        
        self.assertEqual(action.shape, (5,))
        self.assertAlmostEqual(action.sum(), 1.0, places=5)
        self.assertTrue((action >= 0).all())
        self.assertTrue((action <= 1).all())
    
    def test_memory_operations(self):
        """Test replay buffer push/sample"""
        state = np.random.randn(51).astype(np.float32)
        action = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        reward = 0.01
        next_state = np.random.randn(51).astype(np.float32)
        done = False
        
        initial_size = len(self.agent.memory)
        self.agent.memory.push(state, action, reward, next_state, done)
        self.assertEqual(len(self.agent.memory), initial_size + 1)
    
    def test_epsilon_decay(self):
        """Test epsilon decreases correctly"""
        initial_epsilon = self.agent.epsilon
        self.agent.decay_epsilon()
        self.assertLess(self.agent.epsilon, initial_epsilon)
        self.assertGreaterEqual(self.agent.epsilon, self.agent.epsilon_end)


class TestContextualBandit(unittest.TestCase):
    """Test contextual bandit functionality"""
    
    def setUp(self):
        self.bandit = ContextualBanditTool(n_stocks=5)
    
    def test_context_extraction(self):
        """Test context features are extracted correctly"""
        prices = np.random.randn(100, 5).cumsum(axis=0) + 100
        context = self.bandit.extract_context(prices, step=50)
        
        self.assertEqual(context.shape, (5,))
        self.assertFalse(np.isnan(context).any())
        self.assertFalse(np.isinf(context).any())
    
    def test_stock_selection(self):
        """Test stock selection returns valid indices"""
        context = np.random.randn(5)
        selected = self.bandit.select_stocks(context, n_select=3)
        
        self.assertEqual(len(selected), 3)
        self.assertTrue(all(0 <= idx < 5 for idx in selected))
        self.assertEqual(len(set(selected)), 3)  # No duplicates
    
    def test_weight_update(self):
        """Test weight updates work without errors"""
        context = np.random.randn(5)
        initial_weights = self.bandit.weights.copy()
        
        self.bandit.update(0, context, 0.05)
        
        # Weights should change
        self.assertFalse(np.allclose(initial_weights[0], self.bandit.weights[0]))


class TestPortfolioTools(unittest.TestCase):
    """Test portfolio tool utilities"""
    
    def test_risk_analysis_tool(self):
        """Test risk metrics calculation"""
        tool = RiskAnalysisTool()
        
        returns = np.random.randn(100) * 0.02
        sharpe = tool.calculate_sharpe_ratio(returns)
        
        self.assertIsInstance(sharpe, float)
        self.assertFalse(np.isnan(sharpe))
    
    def test_max_drawdown_calculation(self):
        """Test drawdown calculation"""
        tool = RiskAnalysisTool()
        
        portfolio_values = [100000, 110000, 105000, 120000, 90000, 130000]
        max_dd = tool.calculate_max_drawdown(portfolio_values)
        
        self.assertIsInstance(max_dd, float)
        self.assertLess(max_dd, 0)  # Drawdown should be negative


# ============================================================================
# INTEGRATION TESTS - Full System Testing
# ============================================================================

class TestSystemIntegration(unittest.TestCase):
    """Test full system integration"""
    
    @classmethod
    def setUpClass(cls):
        """Load test data once for all tests"""
        try:
            cls.prices = pd.read_csv('data/stock_prices.csv', index_col=0, parse_dates=True)
        except:
            # Generate dummy data if file doesn't exist
            dates = pd.date_range('2023-01-01', periods=200, freq='D')
            cls.prices = pd.DataFrame(
                np.random.randn(200, 5).cumsum(axis=0) + 100,
                index=dates,
                columns=['NVDA', 'GOOGL', 'META', 'MSFT', 'AMD']
            )
    
    def test_environment_reset(self):
        """Test environment resets correctly"""
        env = PortfolioEnv(self.prices, initial_balance=100000)
        state1 = env.reset()
        state2 = env.reset()
        
        self.assertEqual(state1.shape, state2.shape)
        self.assertTrue(np.allclose(state1, state2))
    
    def test_single_episode(self):
        """Test agent can complete one episode"""
        env = PortfolioEnv(self.prices, initial_balance=100000)
        agent = DQNAgent(state_size=51, action_size=5)
        agent.epsilon = 0.0
        
        state = env.reset()
        done = False
        steps = 0
        
        while not done and steps < 1000:
            action = agent.select_action(state, training=False)
            state, reward, done, info = env.step(action)
            steps += 1
        
        self.assertTrue(done)
        self.assertGreater(steps, 0)
    
    def test_training_iteration(self):
        """Test training loop can run"""
        env = PortfolioEnv(self.prices, initial_balance=100000)
        agent = DQNAgent(state_size=51, action_size=5)
        
        # Run mini training (3 episodes)
        for episode in range(3):
            state = env.reset()
            done = False
            
            while not done:
                action = agent.select_action(state, training=True)
                next_state, reward, done, info = env.step(action)
                agent.memory.push(state, action, reward, next_state, done)
                
                if len(agent.memory) >= agent.batch_size:
                    loss = agent.train_step()
                    # Loss should be a number or None
                    if loss is not None:
                        self.assertFalse(np.isnan(loss))
                
                state = next_state
        
        # Should complete without errors
        self.assertTrue(True)


# ============================================================================
# ROBUSTNESS TESTS - Varied Environment Testing
# ============================================================================

class TestVariedEnvironments(unittest.TestCase):
    """Test system across varied conditions (REQUIRED FOR ASSIGNMENT)"""
    
    @classmethod
    def setUpClass(cls):
        """Load test data and model"""
        try:
            cls.prices = pd.read_csv('data/stock_prices.csv', index_col=0, parse_dates=True)
            cls.agent = DQNAgent(state_size=51, action_size=5)
            cls.agent.load(get_model_path('dqn_portfolio.pth'))
            cls.agent.epsilon = 0.0
            cls.has_model = True
        except:
            print("Warning: Trained model not found, using untrained agent")
            cls.has_model = False
    
    def test_different_capital_amounts(self):
        """Test: System works with different initial capital"""
        if not self.has_model:
            self.skipTest("No trained model available")
        
        print("\n  Testing varied capital amounts...")
        
        capital_amounts = [10000, 50000, 100000, 250000]
        returns = []
        
        for capital in capital_amounts:
            env = PortfolioEnv(self.prices, initial_balance=capital)
            state = env.reset()
            done = False
            
            while not done:
                action = self.agent.select_action(state, training=False)
                state, reward, done, info = env.step(action)
            
            final_value = env.portfolio_history[-1]
            total_return = (final_value / capital - 1) * 100
            returns.append(total_return)
            
            print(f"    ${capital:>8,}: {total_return:>6.2f}% return")
        
        # Check consistency (returns should be similar)
        std_return = np.std(returns)
        print(f"    Return std dev: {std_return:.2f}%")
        
        # Pass if returns are reasonably consistent (<20% std)
        self.assertLess(std_return, 20, 
                       f"Returns too inconsistent across capital amounts (std={std_return:.2f}%)")
    
    def test_different_portfolio_sizes(self):
        """Test: System works with different numbers of stocks"""
        print("\n  Testing varied portfolio sizes...")
        
        portfolios = {
            '3-stock': ['NVDA', 'GOOGL', 'MSFT'],
            '5-stock': ['NVDA', 'GOOGL', 'META', 'MSFT', 'AMD']
        }
        
        for name, tickers in portfolios.items():
            stock_prices = self.prices[tickers]
            env = PortfolioEnv(stock_prices, initial_balance=100000)
            
            # Create agent with correct size
            state_size = env.observation_space.shape[0]
            action_size = len(tickers)
            agent = DQNAgent(state_size, action_size)
            
            # Load if same size as trained model
            if action_size == 5 and self.has_model:
                agent.load(get_model_path('dqn_portfolio.pth'))
            
            agent.epsilon = 0.0
            
            state = env.reset()
            done = False
            
            while not done:
                action = agent.select_action(state, training=False)
                state, reward, done, info = env.step(action)
            
            final_value = env.portfolio_history[-1]
            total_return = (final_value / 100000 - 1) * 100
            
            print(f"    {name}: {total_return:>6.2f}% return")
            
            # Should complete successfully
            self.assertGreater(final_value, 0)
    
    def test_different_transaction_costs(self):
        """Test: System handles different transaction cost levels"""
        print("\n  Testing varied transaction costs...")
        
        if not self.has_model:
            self.skipTest("No trained model available")
        
        returns = []
        
        for cost in [0.0001, 0.001, 0.005]:  # 0.01%, 0.1%, 0.5%
            env = PortfolioEnv(self.prices, initial_balance=100000, 
                             transaction_cost=cost)
            state = env.reset()
            done = False
            
            while not done:
                action = self.agent.select_action(state, training=False)
                state, reward, done, info = env.step(action)
            
            final_value = env.portfolio_history[-1]
            total_return = (final_value / 100000 - 1) * 100
            returns.append(total_return)
            
            print(f"    Cost {cost*100:.2f}%: {total_return:>6.2f}% return")
        
        # Returns should decrease as costs increase
        self.assertGreater(returns[0], returns[2], 
                          "Higher transaction costs should reduce returns")


# ============================================================================
# PERFORMANCE TESTS - Verify Quality Metrics
# ============================================================================

class TestPerformanceMetrics(unittest.TestCase):
    """Verify system achieves expected performance levels"""
    
    @classmethod
    def setUpClass(cls):
        """Load test data"""
        try:
            cls.prices = pd.read_csv('data/stock_prices.csv', index_col=0, parse_dates=True)
            cls.env = PortfolioEnv(cls.prices, initial_balance=100000)
            cls.agent = DQNAgent(state_size=51, action_size=5)
            cls.agent.load(get_model_path('dqn_portfolio.pth'))
            cls.agent.epsilon = 0.0
            cls.has_model = True
        except:
            cls.has_model = False
    
    def test_beats_equal_weight(self):
        """Test: DQN should beat equal weight baseline"""
        if not self.has_model:
            self.skipTest("No trained model available")
        
        print("\n  Testing vs equal weight baseline...")
        
        # DQN performance
        state = self.env.reset()
        done = False
        while not done:
            action = self.agent.select_action(state, training=False)
            state, reward, done, info = self.env.step(action)
        dqn_return = (self.env.portfolio_history[-1] / 100000 - 1) * 100
        
        # Equal weight performance
        env2 = PortfolioEnv(self.prices, initial_balance=100000)
        state = env2.reset()
        done = False
        equal_action = np.ones(5) / 5
        while not done:
            state, reward, done, info = env2.step(equal_action)
        equal_return = (env2.portfolio_history[-1] / 100000 - 1) * 100
        
        print(f"    DQN: {dqn_return:.2f}%")
        print(f"    Equal Weight: {equal_return:.2f}%")
        
        # DQN should beat or match equal weight
        self.assertGreaterEqual(dqn_return, equal_return * 0.9,  # Allow 10% tolerance
                               "DQN should beat or match equal weight baseline")
    
    def test_positive_sharpe_ratio(self):
        """Test: Agent achieves positive risk-adjusted returns"""
        if not self.has_model:
            self.skipTest("No trained model available")
        
        print("\n  Testing Sharpe ratio...")
        
        state = self.env.reset()
        done = False
        while not done:
            action = self.agent.select_action(state, training=False)
            state, reward, done, info = self.env.step(action)
        
        returns = np.diff(self.env.portfolio_history) / self.env.portfolio_history[:-1]
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
        
        print(f"    Sharpe Ratio: {sharpe:.3f}")
        
        self.assertGreater(sharpe, 0, "Sharpe ratio should be positive")
        self.assertGreater(sharpe, 1.0, "Sharpe ratio should exceed 1.0 for good performance")


# ============================================================================
# EDGE CASE TESTS - Error Handling
# ============================================================================

class TestEdgeCases(unittest.TestCase):
    """Test system handles edge cases gracefully"""
    
    def test_circuit_breaker(self):
        """Test circuit breaker activates after failures"""
        breaker = CircuitBreaker(failure_threshold=3)
        
        # Trigger failures
        for i in range(5):
            breaker.record_failure(i)
        
        # Circuit should be open
        self.assertTrue(breaker.is_open)
        self.assertTrue(breaker.should_use_fallback(5))
    
    def test_fallback_policy(self):
        """Test fallback policy provides valid actions"""
        fallback = FallbackPolicy(n_stocks=5)
        action = fallback.get_action('equal_weight')
        
        self.assertEqual(action.shape, (5,))
        self.assertAlmostEqual(action.sum(), 1.0)
        self.assertTrue((action >= 0).all())
    
    def test_empty_portfolio_values(self):
        """Test risk tools handle empty inputs"""
        tool = RiskAnalysisTool()
        
        empty_returns = np.array([])
        sharpe = tool.calculate_sharpe_ratio(empty_returns)
        
        self.assertEqual(sharpe, 0.0)


# ============================================================================
# TEST SUITE RUNNER
# ============================================================================

def run_full_test_suite():
    """Run all tests with detailed output"""
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestActionValidator))
    suite.addTests(loader.loadTestsFromTestCase(TestDQNAgent))
    suite.addTests(loader.loadTestsFromTestCase(TestContextualBandit))
    suite.addTests(loader.loadTestsFromTestCase(TestPortfolioTools))
    suite.addTests(loader.loadTestsFromTestCase(TestSystemIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestVariedEnvironments))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformanceMetrics))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    
    # Run with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n ALL TESTS PASSED!")
        print("="*70)
        return True
    else:
        print("\n SOME TESTS FAILED")
        print("="*70)
        return False


if __name__ == '__main__':
    # Run full test suite with summary
    success = run_full_test_suite()
    sys.exit(0 if success else 1)