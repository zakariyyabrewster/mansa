import time
import multiprocessing
import random
from collections import defaultdict
from typing import Dict

import sys
from pathlib import Path

# Add the src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from backend.bazaar import Bazaar, BasicBazaar

# from agents.mansa2_agent import MansaAgent as Agent1
# from agents.mansa_agent import MansaAgent as Agent1
# from agents.optimal_agent import OptimalAgent as Agent1
from agents.simple_agent import SmartAgent as Agent1
from agents.random_agent import RandomAgent as Agent2

NUM_GAMES = 100    # How many games to play total
SEED_START = 100       # Starting random seed (change for different matchups)

def run_single_game(seed: int, verbose=False) -> Dict[str, int]:
    """Runs one game and returns final scores."""
    
    # Initialize Agents
    agent_1 = Agent1(seed=seed, name="Agent1")
    agent_2 = Agent2(seed=seed+1000, name="Agent2")
    
    agents = [agent_1, agent_2]
    
    # Setup game

    game = BasicBazaar(seed=seed, players=agents)
    state = game.state
    
    # Play game
    while not game.terminal(state):
        actor = state.actor
        actions = game.all_actions(actor, state)
        
        if not actions:
            break
            
        def simulate_action(observation, action):
            next_state = game.apply_action(observation, action)
            return game.observe(actor, next_state)
            
        try:
            observation = game.observe(actor, state)
            chosen_action = actor.select_action(actions, observation, simulate_action)
        except Exception:
            chosen_action = random.choice(actions)
            
        # --- CRITICAL FIX IS HERE ---
        # 1. Apply the action to get the new state
        state = game.apply_action(state, chosen_action)
        
        if verbose:
            print(f"{actor.name} takes action: {chosen_action}")

        # 2. UPDATE THE GAME'S INTERNAL STATE
        # Without this, the game engine doesn't know the turn changed,
        # and it will keep assigning the turn to Agent 2 forever.
        game.state = state 
        # ----------------------------
    
    # Get final scores
    scores = {}
    for player in agents:
        scores[player.name] = game.calculate_reward(player, state, state)
    
    if verbose:
        print(f"\nFinal Scores:")
        for name, score in scores.items():
            print(f"{name}: {score} points")
    return scores

def run_tournament():
    """Run tournament with settings from top of file."""
    
    print(f"\n{'='*60}")
    print(f"TOURNAMENT: {Agent1.__name__} vs {Agent2.__name__}")
    print(f"Games: {NUM_GAMES}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    num_workers = multiprocessing.cpu_count()
    print(f"Using {num_workers} CPU cores.\n")
    
    wins = defaultdict(int)
    total_scores = defaultdict(int)
    completed = 0
    
    print("Running games...")
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        iterator = pool.imap_unordered(run_single_game, range(SEED_START, SEED_START + NUM_GAMES), chunksize=10)
        
        for game_result in iterator:
            completed += 1
            
            # Progress updates
            milestone = max(1, NUM_GAMES // 20)
            if completed % milestone == 0 or completed == NUM_GAMES:
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                remaining = (NUM_GAMES - completed) / rate if rate > 0 else 0
                
                print(f"Progress: {completed}/{NUM_GAMES} "
                      f"({completed/NUM_GAMES*100:.0f}%) | "
                      f"{rate:.1f} games/s | "
                      f"ETA: {remaining:.0f}s")
            
            # Track results
            names = list(game_result.keys())
            p1, p2 = names[0], names[1]
            s1, s2 = game_result[p1], game_result[p2]
            
            total_scores[p1] += s1
            total_scores[p2] += s2
            
            if s1 > s2:
                wins[p1] += 1
            elif s2 > s1:
                wins[p2] += 1
            else:
                wins["Draw"] += 1

    # Final report
    duration = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Time: {duration:.2f}s ({duration/NUM_GAMES:.3f}s per game)")
    
    print(f"\n{'='*60}")
    print("WIN RATES")
    print(f"{'='*60}")
    
    for name, win_count in sorted(wins.items(), key=lambda x: x[1], reverse=True):
        pct = (win_count / NUM_GAMES) * 100
        bar = '█' * int(pct / 2)
        print(f"{name:15} : {win_count:4} wins ({pct:5.1f}%) {bar}")
    
    print(f"\n{'='*60}")
    print("AVERAGE SCORES")
    print(f"{'='*60}")
    
    for name, total in sorted(total_scores.items(), key=lambda x: x[1], reverse=True):
        avg = total / NUM_GAMES
        print(f"{name:15} : {avg:6.1f} points")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    run_tournament()
    # run_single_game(seed=SEED_START, verbose=True)

