# 🐪 Mansa: A Bazaar-AI RL Agent

**Bazaar-AI** is a Python-based library built on top of [Arelai](https://github.com/chandra-gummaluru/arelai) that faithfully recreates the strategic card game Jaipur. It provides:

- 🎮 **a complete game implementation**, accurately modeling all Jaipur rules
- 🤖 **a plug-and-play agent interface**, allowing custom agents to be created by extending a single class
- 📊 **a serializable game state**, making it easy to convert states into observations for RL algorithms
- 🎯 **support for custom reward functions**, so you can define your own training signals
- 🖥️ **a visual simulator**, featuring a polished web-based UI for watching agents compete in real time

<p align="center">
  <a href="https://chandra-gummaluru.github.io/2025-12-29/bazaar-ai">
    <img src="https://github.com/chandra-gummaluru/bazaar-ai/blob/main/bazaar-demo-gif.gif?raw=true" style="width: 100%; max-width: 100%;" />
  </a>
</p>


With Bazaar-AI, students can experiment with different RL approaches—from simple heuristics to Monte Carlo Tree Search to deep reinforcement learning—all within a rich, strategic environment.

## Why Jaipur?

Jaipur is an ideal game for AI research and education because it captures many challenges found in real-world decision-making, including:

- **partial observability**, where players cannot see the deck or their opponent’s full hand, requiring decisions to be made under uncertainty.

- **delayed rewards**, in which actions taken now may only pay off several turns later, testing an agent’s ability to plan ahead.

- **strategic depth**, where a relatively small action space still produces surprising complexity and multiple viable strategies.

- **competitive dynamics**, as optimal play depends on anticipating and responding to an opponent’s strategy, introducing game-theoretic considerations.

- **fast episodes**, with games typically lasting 20 to 40 turns, enabling rapid iteration during development.

These characteristics make Jaipur an excellent stepping stone between toy problems and complex real-world applications—challenging enough to be interesting, but tractable enough for a semester project.

## How It Works

The framework is built on Arelai, providing a clean object-oriented structure for game states, actions, and observations. The core game loop handles all the mechanics automatically, so you can focus entirely on developing your agent's decision-making logic.

### Creating Custom Agents

Implementing your own agent is straightforward. Just extend the `Trader` class and override two key methods:

```python
from backend.trader import Trader, TraderAction
from backend.market import MarketObservation
from typing import Optional, Callable

class MyCustomAgent(Trader):
    def __init__(self, seed, name):
        super().__init__(seed, name)
        # Initialize any agent-specific data structures
        self.memory = []
        
    def select_action(self,
                      actions: list[TraderAction],
                      observation: MarketObservation,
                      simulate_action_fnc: Callable[[TraderAction], MarketObservation]) -> TraderAction:
        """
        Choose an action based on the current market state.
        
        Args:
            actions: List of legal actions available
            observation: Current view of the market (your hand, market cards, etc.)
            simulate_action_fnc: Function to simulate what happens if you take an action
            
        Returns:
            The action you want to take
        """
        # Implement your decision logic here
        # You can use simulate_action_fnc to look ahead!
        
        for action in actions:
            future_state = simulate_action_fnc(action)
            # Evaluate this future state...
        
        return best_action
    
    def calculate_reward(self,
                        old_observation: MarketObservation,
                        new_observation: MarketObservation,
                        has_acted: bool,
                        environment_reward: Optional[float]):
        """
        Calculate rewards and update any internal state.
        
        This is called after every turn (yours and your opponent's).
        Use it to update value estimates, store experiences, etc.
        
        Args:
            old_observation: Market state before the action
            new_observation: Market state after the action
            has_acted: True if this was your turn
            environment_reward: Optional reward from the game (e.g., points scored)
        """
        # Update your agent's learning signals
        reward = self._compute_reward(old_observation, new_observation)
        self.memory.append((old_observation, new_observation, reward))
```

The `MarketObservation` provides everything an agent needs to make decisions, including:

- **the agent’s current hand of goods**, representing available resources
- **the cards currently in the market**, defining the set of immediate trade options
- **the number of camels held**, which affects both trading capacity and scoring
- **the available bonus tokens**, indicating potential future rewards
- **the current score**, reflecting overall progress in the game
- **the opponent’s visible information**, including their score, camel count, and hand size

### The Visual Simulator

One of the best features for teaching is the real-time visualization. The included web UI lets you:

- watch agents compete step by step or at variable speeds
- see the full game state at each turn
- compare different agent strategies visually
- debug agent behavior in real time

This makes it easy to identify where an agent's strategy succeeds or fails, and helps students build intuition about what makes a good trading strategy.

## Getting Started

### 1. Installation

First, clone the repository.

```bash
git clone https://github.com/chandra-gummaluru/bazaar-ai.git
```

### 2. Try the Demo

Run a game between the built-in agents:

**For Linux-Based Systems**:

```bash
cd bazaar-ai
bash launch-bazaar-ai
```

**For Windows-Based Systems**:

```
cd bazaar-ai
.\launch-bazaar-ai.bat
```

This will start a local web server and open your browser to the UI. You can select different agents and watch them compete.

### 3. Build Your Own Agent
Within the `agents` directory, create a new file in `my_agent.py`:

```python
from backend.trader import Trader, TraderAction
from backend.market import MarketObservation

class MyAgent(Trader):
    def __init__(self, seed, name):
        super().__init__(seed, name)
        
    def select_action(self, actions, observation, simulate_action_fnc):
        # Start simple - maybe just pick randomly?
        import random
        return random.choice(actions)
    
    def calculate_reward(self, old_obs, new_obs, has_acted, env_reward):
        pass  # No learning yet
```

Then test it in the simulator by selecting it from the dropdown menu.

## Source Code

The complete project is open source and available on GitHub:

- **base library**: [arelai](https://github.com/chandra-gummaluru/arelai)
- **core library**: [bazaar-ai](https://github.com/chandra-gummaluru/bazaar_ai)

Whether you're teaching a course on AI, or just want to build a game-playing agent for fun, Bazaar-AI provides an accessible and engaging platform to get started.

Have ideas for improvements or new features? Found a bug? I'd love to hear from you—feel free to open an issue or reach out directly!
