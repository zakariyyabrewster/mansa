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