import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class DQNNetwork(nn.Module):
    """Deep Q-Network"""
    def __init__(self, state_size, action_size, hidden_size=128):
        super(DQNNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        # Use softmax to ensure outputs sum to 1 (portfolio weights)
        x = self.softmax(x)
        return x


class ReplayBuffer:
    """Experience replay buffer"""
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.FloatTensor(np.array(actions)),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones)
        )
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """DQN Agent for portfolio management"""
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.95, 
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Q-Networks
        self.policy_net = DQNNetwork(state_size, action_size)
        self.target_net = DQNNetwork(state_size, action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # Replay buffer
        self.memory = ReplayBuffer(capacity=10000)
        self.batch_size = 64
        
    def select_action(self, state, training=True):
        """Select action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            # Random action (random portfolio weights)
            action = np.random.dirichlet(np.ones(self.action_size))
            return action.astype(np.float32)
        else:
            # Greedy action from policy network
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action = self.policy_net(state_tensor).squeeze(0).numpy()
            return action
    
    def train_step(self):
        """Perform one training step"""
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Current Q values
        current_q = self.policy_net(states)
        
        # Next Q values from target network
        with torch.no_grad():
            next_q = self.target_net(next_states)
            # For portfolio, we calculate expected return
            next_values = torch.sum(next_q * actions, dim=1)
            target_q = rewards + self.gamma * next_values * (1 - dones)
        
        # Current values
        current_values = torch.sum(current_q * actions, dim=1)
        
        # Loss
        loss = self.criterion(current_values, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network with policy network weights"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath):
        """Save model"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
    
    def load(self, filepath):
        """Load model"""
        checkpoint = torch.load(filepath)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']


# Test the DQN agent
if __name__ == "__main__":
    state_size = 51  # From our environment
    action_size = 5  # 5 stocks
    
    agent = DQNAgent(state_size, action_size)
    
    print("DQN Agent initialized!")
    print(f"State size: {state_size}")
    print(f"Action size: {action_size}")
    print(f"Network architecture:")
    print(agent.policy_net)
    
    # Test action selection
    dummy_state = np.random.randn(state_size).astype(np.float32)
    action = agent.select_action(dummy_state)
    print(f"\nTest action (portfolio weights): {action}")
    print(f"Sum of weights: {action.sum():.4f}")
    
    print("\n✓ DQN Agent working!")