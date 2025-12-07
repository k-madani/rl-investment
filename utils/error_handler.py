"""
Error handling and fallback strategies for RL Investment System.

Provides robust error recovery mechanisms for production deployment.
"""

import numpy as np
import logging
from functools import wraps

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RLInvestmentError(Exception):
    """Base exception for RL Investment System"""
    pass


class ActionValidationError(RLInvestmentError):
    """Raised when agent action is invalid"""
    pass


class EnvironmentError(RLInvestmentError):
    """Raised when environment encounters issues"""
    pass


def safe_execute(fallback_value=None, log_error=True):
    """
    Decorator for safe function execution with fallback.
    
    Args:
        fallback_value: Value to return on error
        log_error: Whether to log the error
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_error:
                    logger.error(f"Error in {func.__name__}: {str(e)}")
                if fallback_value is not None:
                    logger.info(f"Returning fallback value: {fallback_value}")
                    return fallback_value
                raise
        return wrapper
    return decorator


class ActionValidator:
    """
    Validates and sanitizes agent actions.
    
    Ensures portfolio weights are valid before execution.
    """
    
    def __init__(self, n_stocks, tolerance=1e-6):
        self.n_stocks = n_stocks
        self.tolerance = tolerance
        self.equal_weight = np.ones(n_stocks) / n_stocks
        
    def validate(self, action):
        """
        Validate portfolio action.
        
        Args:
            action (np.array): Portfolio weights
            
        Returns:
            tuple: (is_valid, error_message)
        """
        # Check type
        if not isinstance(action, np.ndarray):
            return False, "Action must be numpy array"
        
        # Check shape
        if action.shape != (self.n_stocks,):
            return False, f"Action shape must be ({self.n_stocks},)"
        
        # Check for NaN/Inf
        if np.isnan(action).any():
            return False, "Action contains NaN values"
        
        if np.isinf(action).any():
            return False, "Action contains Inf values"
        
        # Check bounds
        if (action < 0).any():
            return False, "Action contains negative weights"
        
        if (action > 1).any():
            return False, "Action contains weights > 1"
        
        # Check sum
        if abs(action.sum() - 1.0) > self.tolerance:
            return False, f"Action weights sum to {action.sum():.4f}, not 1.0"
        
        return True, "Valid"
    
    def sanitize(self, action, strategy='equal_weight'):
        """
        Sanitize invalid action with fallback strategy.
        
        Args:
            action: Potentially invalid action
            strategy: Fallback strategy ('equal_weight', 'clip', 'normalize')
            
        Returns:
            np.array: Valid action
        """
        is_valid, error_msg = self.validate(action)
        
        if is_valid:
            return action
        
        logger.warning(f"Invalid action detected: {error_msg}. Using {strategy} fallback.")
        
        if strategy == 'equal_weight':
            return self.equal_weight.copy()
        
        elif strategy == 'clip':
            # Clip to valid range and normalize
            action_safe = np.clip(action, 0, 1)
            action_safe = np.nan_to_num(action_safe, nan=0.0, posinf=1.0, neginf=0.0)
            
            if action_safe.sum() == 0:
                return self.equal_weight.copy()
            
            return action_safe / action_safe.sum()
        
        elif strategy == 'normalize':
            # Handle NaN/Inf then normalize
            action_safe = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=0.0)
            action_safe = np.abs(action_safe)  # Make non-negative
            
            if action_safe.sum() == 0:
                return self.equal_weight.copy()
            
            return action_safe / action_safe.sum()
        
        else:
            return self.equal_weight.copy()


class StateValidator:
    """Validates environment state."""
    
    def __init__(self, expected_shape):
        self.expected_shape = expected_shape
    
    def validate(self, state):
        """
        Validate environment state.
        
        Args:
            state: Environment state
            
        Returns:
            tuple: (is_valid, error_message)
        """
        if not isinstance(state, np.ndarray):
            return False, "State must be numpy array"
        
        if state.shape != self.expected_shape:
            return False, f"State shape {state.shape} != expected {self.expected_shape}"
        
        if np.isnan(state).any():
            return False, "State contains NaN"
        
        if np.isinf(state).any():
            return False, "State contains Inf"
        
        return True, "Valid"


class FallbackPolicy:
    """
    Fallback policy for when primary agent fails.
    
    Provides safe, conservative actions when agent errors occur.
    """
    
    def __init__(self, n_stocks):
        self.n_stocks = n_stocks
        self.strategies = {
            'equal_weight': self._equal_weight,
            'momentum': self._momentum_based,
            'inverse_volatility': self._inverse_volatility
        }
    
    def _equal_weight(self, state=None):
        """Equal weight allocation (1/n each)"""
        return np.ones(self.n_stocks) / self.n_stocks
    
    def _momentum_based(self, state):
        """Weight by recent momentum"""
        # Extract returns from state (assuming state contains recent returns)
        # Simplified: use equal weight if can't extract
        return self._equal_weight()
    
    def _inverse_volatility(self, state):
        """Weight inversely by volatility"""
        # Simplified: use equal weight
        return self._equal_weight()
    
    def get_action(self, strategy='equal_weight', state=None):
        """
        Get fallback action.
        
        Args:
            strategy: Strategy name
            state: Environment state (optional)
            
        Returns:
            np.array: Safe portfolio action
        """
        if strategy not in self.strategies:
            strategy = 'equal_weight'
        
        logger.info(f"Using fallback policy: {strategy}")
        return self.strategies[strategy](state)


class CircuitBreaker:
    """
    Circuit breaker pattern for agent failures.
    
    Temporarily disables agent after repeated failures,
    falling back to safe policy.
    """
    
    def __init__(self, failure_threshold=5, reset_time=50):
        self.failure_threshold = failure_threshold
        self.reset_time = reset_time
        self.failure_count = 0
        self.last_failure_step = 0
        self.is_open = False
    
    def record_failure(self, current_step):
        """Record an agent failure"""
        self.failure_count += 1
        self.last_failure_step = current_step
        
        if self.failure_count >= self.failure_threshold:
            self.is_open = True
            logger.warning(f"Circuit breaker OPEN after {self.failure_count} failures")
    
    def record_success(self):
        """Record an agent success"""
        if not self.is_open:
            self.failure_count = max(0, self.failure_count - 1)
    
    def check_reset(self, current_step):
        """Check if circuit should reset"""
        if self.is_open and (current_step - self.last_failure_step) >= self.reset_time:
            self.is_open = False
            self.failure_count = 0
            logger.info("Circuit breaker RESET")
    
    def should_use_fallback(self, current_step):
        """Check if should use fallback instead of agent"""
        self.check_reset(current_step)
        return self.is_open


# Example usage
if __name__ == "__main__":
    print("Testing Error Handling Components...")
    
    # Test Action Validator
    print("\n1. Testing ActionValidator:")
    validator = ActionValidator(n_stocks=5)
    
    # Valid action
    valid_action = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    is_valid, msg = validator.validate(valid_action)
    print(f"   Valid action: {is_valid} - {msg}")
    
    # Invalid action (NaN)
    invalid_action = np.array([0.2, np.nan, 0.2, 0.2, 0.4])
    is_valid, msg = validator.validate(invalid_action)
    print(f"   Invalid action (NaN): {is_valid} - {msg}")
    
    # Sanitize
    sanitized = validator.sanitize(invalid_action, strategy='equal_weight')
    print(f"   Sanitized: {sanitized}")
    
    # Test Fallback Policy
    print("\n2. Testing FallbackPolicy:")
    fallback = FallbackPolicy(n_stocks=5)
    action = fallback.get_action('equal_weight')
    print(f"   Fallback action: {action}")
    
    # Test Circuit Breaker
    print("\n3. Testing CircuitBreaker:")
    breaker = CircuitBreaker(failure_threshold=3)
    
    for i in range(5):
        breaker.record_failure(i)
        print(f"   Step {i}: Should use fallback? {breaker.should_use_fallback(i)}")
    
    print("\n✓ Error handling tests complete!")