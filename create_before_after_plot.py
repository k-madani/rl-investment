import matplotlib.pyplot as plt

# Simplified data (every 10 episodes)
episodes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
untrained = [110000, 125000, 95000, 130000, 105000, 120000, 115000, 100000, 125000, 120000]
trained = [158000, 160000, 170000, 180000, 190000, 200000, 220000, 240000, 250000, 260937]

plt.figure(figsize=(10, 5))
plt.plot(episodes, untrained, 'r-o', linewidth=2, markersize=6, label='Untrained (Random)')
plt.plot(episodes, trained, 'b-o', linewidth=2, markersize=6, label='Trained DQN')
plt.axhline(y=100000, color='gray', linestyle='--', alpha=0.5)
plt.xlabel('Episode')
plt.ylabel('Portfolio Value ($)')
plt.title('Learning Impact: Trained vs Untrained Agent')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('before_after_comparison.png', dpi=300)
print("✓ Visualization saved as before_after_comparison.png")