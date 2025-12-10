# Multi-Agent RL Portfolio Optimization

A reinforcement learning system for automated portfolio management that learns optimal stock allocation strategies through experience.

## Overview

A multi-agent reinforcement learning system for adaptive portfolio management. Combines Deep Q-Networks (DQN) with Contextual Bandits to learn optimal allocation strategies across stocks, dynamically adapting to market conditions.

**Problem Solved:** Traditional portfolio management uses static rules (equal-weight, 60/40) that fail during market regime changes. This system learns adaptive strategies through experience.

**Use Cases:**

- Retail investors ($50k-$500k portfolios)
- Robo-advisory platforms requiring dynamic rebalancing
- Quantitative hedge funds seeking adaptive strategies
- Academic research in financial RL

## Installation

### Prerequisites

- Python 3.10 or higher
- pip 21.0 or higher
- 16GB RAM (8GB minimum)
- No GPU required

### Setup

```bash
git clone https://github.com/k-madani/rl-investment.git
cd rl-investment
pip install -r requirements.txt
```

**Train the agent:**

```bash
python train.py
```

**Evaluate against baselines:**

```bash
python evaluate.py
```

**Run robustness experiments:**

```bash
python experiment.py
```

**Analyze agent behavior:**

```bash
python analyze_agent.py
```

## Features

### Core Capabilities

- **Multi-Agent Architecture**: Coordinated stock selection (Contextual Bandits) and allocation (DQN)
- **Superior Returns**: 210% vs 116% equal-weight baseline (+94pp outperformance)
- **Best Risk-Adjusted**: 1.931 Sharpe ratio, p < 0.001 statistical significance
- **Crisis-Tested**: +12% during COVID crash, +237% in bull markets
- **Production-Ready**: Circuit breakers, error handling, <10ms inference

### Technical Features

- **Zero-Failure Training**: Automatic fallback mechanisms
- **Fast Inference**: <10ms per decision, suitable for real-time trading
- **Comprehensive Logging**: Audit trails for all decisions
- **Error Handling**: Circuit breakers with graceful degradation
- **Reproducible**: Fixed random seeds, documented experiments
- **Visualization**: Real-time training curves and performance metrics

### Innovation

- **Novel Approach**: First documented application of Contextual Bandits to portfolio stock filtering
- **Academically Sound**: Implements state-of-the-art RL algorithms with rigorous validation
- **Open Source**: MIT licensed, full code transparency

## Performance

### Benchmark Results

| Metric | RL System | Equal Weight | Buy & Hold NVDA |
|--------|-----------|--------------|-----------------|
| **Total Return** | **210.06%** | 116.30% | 276.14% |
| **Sharpe Ratio** | **1.931** ⭐ | 1.464 | 1.586 |
| **Max Drawdown** | **-26.10%** | -28.05% | -36.88% |
| **Significance** | **Baseline** | p < 0.001 | p = 0.043 |

### Robustness Testing

| Scenario | Return | Sharpe | Status |
|----------|--------|--------|--------|
| Baseline (2Y) | +210% | 1.931 | ✅ Passed |
| COVID Crash | +12% | 0.730 | ✅ Passed |
| Bull Market | +237% | 0.872 | ✅ Passed |
| High Trans. Cost | +198% | 1.850 | ✅ Passed |

## System Architecture

The system uses three coordinated components:

1. **Contextual Bandit Agent** - Selects promising stocks based on market context
2. **DQN Agent** - Optimizes capital allocation across selected stocks  
3. **Portfolio Environment** - Simulates realistic trading with transaction costs

## Technical Details

**RL Approaches:**

- Deep Q-Network with experience replay and target network
- Contextual Bandits using UCB for stock selection
- Multi-agent coordination with shared rewards

**Dataset:**

- Training: 2-year historical data (NVDA, GOOGL, META, MSFT, AMD)
- Testing: Separate 2-year holdout period
- Validation: Statistical significance testing (p < 0.001)

## Project Structure

``
rl-investment/
├── agents/
│   ├── dqn_agent.py
│   ├── orchestrator.py
│   └── tools/portfolio_tools.py
├── environment/portfolio_env.py
├── train.py
├── evaluate.py
├── experiment.py
└── analyze_agent.py
``
