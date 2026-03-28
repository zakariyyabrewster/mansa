"""
evolve_mansa.py
===============
Two-phase evolutionary curriculum for MansaAgent.

Phase 1 — Anchor-only training
-------------------------------
The evolving population plays exclusively against Random and Smart anchors.
No self-play occurs. Selection is driven purely by composite win-rate against
those two opponents, so the population converges toward weights that reliably
beat your actual test conditions before the self-play arms race begins.

  fitness = w_random * winrate_vs_random + w_smart * winrate_vs_smart

Phase 2 — Mixed self-play + anchor tournament (Option C)
---------------------------------------------------------
The best survivors from Phase 1 seed a standard round-robin tournament that
includes self-play AND persistent Random/Smart anchors, with OptimalAgent
injected periodically as a stress-test. Fitness is a composite score that
weights anchor win-rates more heavily than self-play Elo so the population
cannot drift away from your test conditions.

  fitness = w_random * wr_random + w_smart * wr_smart + w_self * selfplay_elo_norm

Why two phases?
  Phase 1 ensures the gene pool entering Phase 2 is already competent against
  your real opponents, so the Phase 2 arms race starts from a meaningful
  baseline rather than noise vs noise. Phase 2 then discovers strategies that
  fixed-opponent training alone cannot find (multi-turn traps, denial plays,
  adaptive tempo) because the self-play opponents actually adapt.

Usage
-----
    python evolve_mansa.py                              # full two-phase run
    python evolve_mansa.py --p1-gens 15 --p2-gens 40   # custom phase lengths
    python evolve_mansa.py --resume                     # resume from checkpoint
    python evolve_mansa.py --phase2-only                # skip Phase 1, load Phase 1 results
"""

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Optional

# ── project imports ───────────────────────────────────────────────────────────
from backend.bazaar import BasicBazaar
from backend.trader import Trader
from agents.mansa2_agent import MansaAgent, DEFAULT_WEIGHTS
from agents.random_agent import RandomAgent
from agents.simple_agent import SmartAgent
from agents.optimal_agent import OptimalAgent


# ── constants ─────────────────────────────────────────────────────────────────
NUM_WEIGHTS  = len(DEFAULT_WEIGHTS)
WEIGHT_MIN   = 0.01     # hard floor — weights must stay positive
WEIGHT_MAX   = 20.0     # hard ceiling — prevents any single weight exploding

# Elo constants (Phase 2 only — Phase 1 uses win-rate, not Elo)
ELO_K        = 32       # update step for self-play games
ELO_K_ANCHOR = 16       # smaller step for anchor games so their Elo converges
                        # slowly and acts as a stable reference, not a volatile signal
ELO_INIT     = 1200.0

# Anchor sentinel IDs — negative so they cannot collide with evolving agent IDs
ANCHOR_ID_RANDOM  = -1
ANCHOR_ID_SMART   = -2
ANCHOR_ID_OPTIMAL = -3

# How many Random and Smart instances to include in the Phase 2 tournament.
# More instances shifts the self-play/anchor game ratio toward anchors without
# changing the underlying structure of the round-robin.
NUM_RANDOM_ANCHORS = 4   # gives ~16 anchor games vs Random per evolving agent
NUM_SMART_ANCHORS  = 4   # gives ~16 anchor games vs Smart  per evolving agent

# How often (in Phase 2 generations) to inject OptimalAgent.
# Kept periodic because a dominant opponent distorts Elo rankings —
# every agent's pressure collapses into "don't get destroyed" rather than
# actually improving against Random and Smart.
OPTIMAL_STRESS_EVERY = 5

# Phase 1 composite fitness weights — must sum to 1.0
# Smart is weighted higher because it is the harder and more informative test;
# any agent that can beat Smart can almost certainly beat Random too.
P1_FITNESS_W_RANDOM = 0.40
P1_FITNESS_W_SMART  = 0.60

# Phase 2 composite fitness weights — must sum to 1.0
# Anchor win-rates dominate (0.70 total) so self-play cannot wash out the
# signal from your actual test opponents.
P2_FITNESS_W_RANDOM   = 0.35
P2_FITNESS_W_SMART    = 0.35
P2_FITNESS_W_SELFPLAY = 0.30   # normalized self-play Elo contribution

# Number of games played against each anchor type when computing Phase 1 fitness.
# Higher = more accurate win-rate estimate but slower per generation.
P1_ANCHOR_GAMES = 10   # × 2 orderings = 20 games per anchor type per agent


# ─────────────────────────────────────────────────────────────────────────────
# AgentRecord
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AgentRecord:
    """
    Pairs a weight vector with all performance history for one agent.
    Anchor agents share this dataclass but have is_anchor=True and
    anchor_type set, which blocks them from breeding and mutation.
    """
    agent_id:       int
    weights:        list            # 24-element vector; None for anchors
    elo:            float = ELO_INIT
    wins:           int   = 0       # reset each generation
    losses:         int   = 0
    draws:          int   = 0
    generation_born: int  = 0
    parent_ids:     list  = field(default_factory=list)
    is_anchor:      bool  = False
    anchor_type:    str   = ""      # "random" | "smart" | "optimal" | ""
    anchor_seed:    int   = 0       # per-instance seed so multiple Random/Smart
                                    # instances behave differently from each other
    # Phase 1 fitness fields — populated after each Phase 1 generation
    p1_wr_random:   float = 0.0     # win-rate vs Random in Phase 1
    p1_wr_smart:    float = 0.0     # win-rate vs Smart  in Phase 1
    p1_fitness:     float = 0.0     # composite Phase 1 fitness score

    def win_rate(self):
        total = self.wins + self.losses + self.draws
        return self.wins / total if total else 0.0

    def label(self):
        """Human-readable name used in all log output."""
        if self.is_anchor:
            return f"{self.anchor_type}_{abs(self.agent_id)}"
        return f"A{self.agent_id}"

    def to_dict(self):
        return {
            "agent_id":        self.agent_id,
            "weights":         self.weights,
            "elo":             round(self.elo, 2),
            "wins":            self.wins,
            "losses":          self.losses,
            "draws":           self.draws,
            "generation_born": self.generation_born,
            "parent_ids":      self.parent_ids,
            "is_anchor":       self.is_anchor,
            "anchor_type":     self.anchor_type,
            "anchor_seed":     self.anchor_seed,
            "p1_wr_random":    self.p1_wr_random,
            "p1_wr_smart":     self.p1_wr_smart,
            "p1_fitness":      self.p1_fitness,
        }

    @staticmethod
    def from_dict(d):
        return AgentRecord(
            agent_id        = d["agent_id"],
            weights         = d["weights"],
            elo             = d.get("elo", ELO_INIT),
            wins            = d.get("wins", 0),
            losses          = d.get("losses", 0),
            draws           = d.get("draws", 0),
            generation_born = d.get("generation_born", 0),
            parent_ids      = d.get("parent_ids", []),
            is_anchor       = d.get("is_anchor", False),
            anchor_type     = d.get("anchor_type", ""),
            anchor_seed     = d.get("anchor_seed", 0),
            p1_wr_random    = d.get("p1_wr_random", 0.0),
            p1_wr_smart     = d.get("p1_wr_smart", 0.0),
            p1_fitness      = d.get("p1_fitness", 0.0),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Anchor factories
# ─────────────────────────────────────────────────────────────────────────────

def make_p1_anchors() -> list[AgentRecord]:
    """
    Phase 1 anchors: one Random and one Smart instance.

    Phase 1 uses direct win-rate measurement rather than a round-robin, so
    we only need one instance of each — we control the game count explicitly.
    """
    return [
        AgentRecord(ANCHOR_ID_RANDOM, None, is_anchor=True,
                    anchor_type="random", anchor_seed=0),
        AgentRecord(ANCHOR_ID_SMART,  None, is_anchor=True,
                    anchor_type="smart",  anchor_seed=0),
    ]


def make_p2_anchors() -> list[AgentRecord]:
    """
    Phase 2 anchors: multiple Random and Smart instances + one Optimal.

    Multiple instances of each type serve two purposes:
      1. Each instance has a different seed, so the evolving agents see a
         wider distribution of Random/Smart behavior rather than one fixed
         opponent they can overfit to.
      2. More anchor instances in the round-robin increases the fraction of
         games played against real opponents vs self-play, shifting selection
         pressure toward your actual test conditions.

    IDs are assigned as consecutive negatives so they never collide with
    evolving agent IDs (which start at 0 and count up).
    """
    anchors = []
    next_anchor_id = -1

    # multiple Random instances — different seeds mean different action choices
    for i in range(NUM_RANDOM_ANCHORS):
        anchors.append(AgentRecord(
            agent_id    = next_anchor_id,
            weights     = None,
            is_anchor   = True,
            anchor_type = "random",
            anchor_seed = i * 17,   # spread seeds so instances are uncorrelated
        ))
        next_anchor_id -= 1

    # multiple Smart instances
    for i in range(NUM_SMART_ANCHORS):
        anchors.append(AgentRecord(
            agent_id    = next_anchor_id,
            weights     = None,
            is_anchor   = True,
            anchor_type = "smart",
            anchor_seed = i * 13,
        ))
        next_anchor_id -= 1

    # single Optimal — periodic stress-test only, so one instance is enough
    anchors.append(AgentRecord(
        agent_id    = ANCHOR_ID_OPTIMAL,
        weights     = None,
        is_anchor   = True,
        anchor_type = "optimal",
        anchor_seed = 0,
    ))

    return anchors


def instantiate_player(rec: AgentRecord, game_seed: int, slot: str) -> Trader:
    """
    Build a live Trader instance from an AgentRecord for one game.

    slot is "a" or "b" — offset by 1 so both players have distinct internal
    RNG states even when the same game_seed is passed. Without this, two
    MansaAgents in the same game would make identical random fallback choices.
    """
    seed_offset = 0 if slot == "a" else 1

    if rec.is_anchor:
        cls = {
            "random":  RandomAgent,
            "smart":   SmartAgent,
            "optimal": OptimalAgent,
        }[rec.anchor_type]
        # use the anchor's fixed per-instance seed (not the game seed) so each
        # anchor instance behaves consistently across all games it plays
        return cls(seed=rec.anchor_seed + seed_offset, name=rec.label())

    # evolving MansaAgent — inject the evolved weight vector
    return MansaAgent(seed=game_seed + seed_offset, name=rec.label(),
                      weights=rec.weights)


# ─────────────────────────────────────────────────────────────────────────────
# Elo helpers (Phase 2)
# ─────────────────────────────────────────────────────────────────────────────

def _expected(ra: float, rb: float) -> float:
    """Standard Elo expected score for player A. Returns probability in (0,1)."""
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))


def update_elo(winner: AgentRecord, loser: AgentRecord,
               draw: bool = False, k: float = ELO_K):
    """
    Apply one Elo update to both players.
    k can be overridden — anchor games use ELO_K_ANCHOR so anchor ratings
    converge slowly and act as stable landmarks rather than volatile signals.
    """
    ea = _expected(winner.elo, loser.elo)
    eb = _expected(loser.elo, winner.elo)
    sa, sb = (0.5, 0.5) if draw else (1.0, 0.0)
    winner.elo += k * (sa - ea)
    loser.elo  += k * (sb - eb)


# ─────────────────────────────────────────────────────────────────────────────
# Core game runner (shared by both phases)
# ─────────────────────────────────────────────────────────────────────────────

def run_game(rec_a: AgentRecord, rec_b: AgentRecord,
             seed: int) -> Optional[int]:
    """
    Run one complete Bazaar game. Returns 0 if rec_a wins, 1 if rec_b wins,
    None for a draw.
    """
    player_a = instantiate_player(rec_a, seed, slot="a")
    player_b = instantiate_player(rec_b, seed, slot="b")
    game = BasicBazaar(seed=seed, players=[player_a, player_b])
    
    state = game.state
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
        
        # 2. UPDATE THE GAME'S INTERNAL STATE
        # Without this, the game engine doesn't know the turn changed,
        # and it will keep assigning the turn to Agent 2 forever.
        game.state = state 
        # ----------------------------


    sa = game.calculate_reward(player_a, state, state)
    sb = game.calculate_reward(player_b, state, state)
    if sa > sb:  return 0
    if sb > sa:  return 1
    return None


def play_series(rec_a: AgentRecord, rec_b: AgentRecord,
                n_games: int) -> tuple[int, int, int]:
    """
    Play n_games between two agents, alternating first-mover each game to
    cancel positional advantage. Returns (wins_a, wins_b, draws).
    """
    wins_a = wins_b = draws = 0
    for i in range(n_games):
        seed = random.randint(0, 2**31 - 1)
        if i % 2 == 0:
            # rec_a goes first
            result = run_game(rec_a, rec_b, seed)
            if result == 0:   wins_a += 1
            elif result == 1: wins_b += 1
            else:             draws  += 1
        else:
            # rec_b goes first — flip which slot result maps to which agent
            result = run_game(rec_b, rec_a, seed)
            if result == 0:   wins_b += 1
            elif result == 1: wins_a += 1
            else:             draws  += 1
    return wins_a, wins_b, draws


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — anchor-only fitness evaluation
# ─────────────────────────────────────────────────────────────────────────────

def p1_evaluate_agent(rec: AgentRecord,
                      random_anchor: AgentRecord,
                      smart_anchor: AgentRecord,
                      n_games: int = P1_ANCHOR_GAMES) -> None:
    """
    Measure one evolving agent's win-rate against Random and Smart, then
    compute its composite Phase 1 fitness score. Results are written directly
    onto the AgentRecord so the caller can sort by rec.p1_fitness.

    n_games x 2 orderings = 2*n_games total games per anchor type.
    We do not use Elo here — Phase 1 selection is purely win-rate based
    so there is no shared pool to corrupt with anchor rating noise.
    """
    wa, _, da = play_series(rec, random_anchor, n_games)
    rec.p1_wr_random = (wa + 0.5 * da) / n_games   # win-rate vs Random

    wa, _, da = play_series(rec, smart_anchor, n_games)
    rec.p1_wr_smart  = (wa + 0.5 * da) / n_games   # win-rate vs Smart

    # composite fitness — weighted combination of both win-rates.
    # Smart is weighted higher (0.60) because it is a harder opponent and a
    # more informative signal; an agent that beats Smart almost certainly
    # beats Random too, but the reverse is not true.
    rec.p1_fitness = (
        P1_FITNESS_W_RANDOM * rec.p1_wr_random +
        P1_FITNESS_W_SMART  * rec.p1_wr_smart
    )


def run_phase1(
    population:     list[AgentRecord],
    n_generations:  int,
    survival_ratio: float,
    elite_count:    int,
    mutation_sigma: float,
    pop_size:       int,
    rng:            random.Random,
    next_id:        int,
    history:        list,
    checkpoint_path: str,
    verbose:        bool,
) -> tuple[list[AgentRecord], int, list]:
    """
    Phase 1 evolution loop.

    Each generation:
      1. Every evolving agent plays P1_ANCHOR_GAMES games vs Random and Smart.
      2. Agents are ranked by composite win-rate fitness (no Elo, no self-play).
      3. Bottom half is eliminated.
      4. Survivors breed the next generation via crossover + mutation.

    This directly optimizes for performance against your test opponents before
    any self-play pressure is introduced.
    """
    # one fixed anchor instance per type — sufficient since we control game count
    p1_anchors = make_p1_anchors()
    random_anchor = next(a for a in p1_anchors if a.anchor_type == "random")
    smart_anchor  = next(a for a in p1_anchors if a.anchor_type == "smart")

    survivors_n = max(elite_count + 1, int(pop_size * survival_ratio))

    for gen in range(n_generations):
        t0 = time.time()
        print(f"\n{'='*60}")
        print(f"  PHASE 1 | Generation {gen}  |  pop = {len(population)}")
        print(f"{'='*60}")

        # ── evaluate every agent against both anchors ─────────────────────────
        for i, rec in enumerate(population):
            p1_evaluate_agent(rec, random_anchor, smart_anchor)
            if verbose:
                print(f"  [{i+1}/{len(population)}] {rec.label():<8}  "
                      f"wr_random={rec.p1_wr_random:.2f}  "
                      f"wr_smart={rec.p1_wr_smart:.2f}  "
                      f"fitness={rec.p1_fitness:.3f}")

        # ── selection — rank purely by composite fitness, no Elo ─────────────
        population.sort(key=lambda r: r.p1_fitness, reverse=True)
        survivors = population[:survivors_n]

        # ── log this generation ───────────────────────────────────────────────
        best = survivors[0]
        entry = {
            "phase":          1,
            "generation":     gen,
            "best_id":        best.agent_id,
            "best_fitness":   round(best.p1_fitness, 4),
            "best_wr_random": round(best.p1_wr_random, 3),
            "best_wr_smart":  round(best.p1_wr_smart, 3),
            "mean_fitness":   round(sum(r.p1_fitness for r in survivors) / len(survivors), 4),
            "best_weights":   best.weights,
        }
        history.append(entry)
        print(f"\n  Best: {best.label()}  fitness={best.p1_fitness:.3f}  "
              f"wr_random={best.p1_wr_random:.2f}  wr_smart={best.p1_wr_smart:.2f}")

        # ── reproduction ──────────────────────────────────────────────────────
        # elites carry over unchanged — the best solution is never lost
        new_population = list(survivors[:elite_count])

        while len(new_population) < pop_size:
            # fitness-proportionate selection: softmax over p1_fitness scores.
            # Shift by min before exp() to prevent overflow on large values.
            fits    = [r.p1_fitness for r in survivors]
            min_fit = min(fits)
            sel_w   = [math.exp((f - min_fit) * 10) for f in fits]
            # ×10 sharpens the softmax — without scaling, small fitness differences
            # produce nearly uniform probabilities and selection becomes random
            total_w = sum(sel_w)
            probs   = [w / total_w for w in sel_w]

            pa = rng.choices(survivors, weights=probs)[0]
            pb = rng.choices(survivors, weights=probs)[0]
            while pb is pa and len(survivors) > 1:
                pb = rng.choices(survivors, weights=probs)[0]

            child = breed(pa, pb, next_id, gen, mutation_sigma)
            new_population.append(child)
            next_id += 1

        population = new_population
        elapsed = time.time() - t0
        print(f"  Phase 1 gen {gen} complete in {elapsed:.1f}s")

        # checkpoint after every generation so progress survives crashes
        save_checkpoint(checkpoint_path, "phase1", gen + 1, population,
                        [], history, next_id)

    return population, next_id, history


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — mixed self-play + anchor tournament
# ─────────────────────────────────────────────────────────────────────────────

def p2_composite_fitness(rec: AgentRecord,
                         all_evolving: list[AgentRecord]) -> float:
    """
    Compute Phase 2 composite fitness for one evolving agent.

    Combines:
      - win-rate vs Random anchors  (from this generation's tournament results)
      - win-rate vs Smart anchors   (from this generation's tournament results)
      - normalized self-play Elo    (relative rank among evolving agents)

    Anchor win-rates dominate (0.70 total) so self-play cannot wash out the
    test-opponent signal. Self-play Elo is normalized to [0,1] across the
    current evolving pool so it contributes a consistent fraction regardless
    of the absolute Elo values, which drift upward across generations.
    """
    # compute win-rates from this generation's accumulated wins/losses/draws.
    # We track separate counters for anchor-type opponents by scanning the
    # tournament log embedded in the record (see run_p2_tournament).
    wr_random = rec.p2_wr_random if hasattr(rec, "p2_wr_random") else 0.0
    wr_smart  = rec.p2_wr_smart  if hasattr(rec, "p2_wr_smart")  else 0.0

    # normalize self-play Elo to [0,1] across the current evolving pool
    elos    = [r.elo for r in all_evolving]
    min_elo = min(elos)
    max_elo = max(elos)
    elo_range = max_elo - min_elo
    elo_norm = (rec.elo - min_elo) / elo_range if elo_range > 0 else 0.5

    return (
        P2_FITNESS_W_RANDOM   * wr_random +
        P2_FITNESS_W_SMART    * wr_smart  +
        P2_FITNESS_W_SELFPLAY * elo_norm
    )


def run_p2_tournament(
    evolving:       list[AgentRecord],
    active_anchors: list[AgentRecord],
    games_per_pair: int,
    verbose:        bool,
) -> None:
    """
    Round-robin tournament for Phase 2. All evolving agents play each other
    and each active anchor. Anchor-vs-anchor matchups are skipped because
    they don't affect evolving agent fitness.

    Per-type win-rate tracking:
      Each evolving agent gets p2_wr_random and p2_wr_smart attributes
      updated in place so p2_composite_fitness can read them after the
      tournament without needing to re-run any games.

    Elo update rules:
      - self-play games use ELO_K
      - anchor games use ELO_K_ANCHOR so anchor ratings are stable landmarks
    """
    # initialise per-type win-rate accumulators on each evolving agent
    for rec in evolving:
        rec.p2_random_w = rec.p2_random_g = 0   # wins and total games vs Random
        rec.p2_smart_w  = rec.p2_smart_g  = 0   # wins and total games vs Smart

    all_participants = evolving + active_anchors
    n = len(all_participants)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    total = len(pairs)

    for idx, (i, j) in enumerate(pairs):
        ra, rb = all_participants[i], all_participants[j]

        # skip anchor-vs-anchor — results don't inform evolving agent selection
        if ra.is_anchor and rb.is_anchor:
            continue

        is_anchor_game = ra.is_anchor or rb.is_anchor
        k_elo = ELO_K_ANCHOR if is_anchor_game else ELO_K

        wa, wb, d = play_series(ra, rb, games_per_pair)

        # update global win/loss/draw counts (for display)
        ra.wins += wa; rb.wins += wb
        ra.losses += wb; rb.losses += wa
        ra.draws += d; rb.draws += d

        # update Elo once per individual game result
        for _ in range(wa):
            update_elo(ra, rb, k=k_elo)
        for _ in range(wb):
            update_elo(rb, ra, k=k_elo)
        for _ in range(d):
            update_elo(ra, rb, draw=True, k=k_elo)

        # update per-type win-rate accumulators for whichever participant
        # is an evolving agent (anchor agents don't need per-type tracking)
        def _update_type_stats(evolving_rec, anchor_rec, ew, d):
            """Record wins and game count for one evolving agent vs one anchor."""
            total_g = wa + wb + d
            wins_for_evolving = ew
            if anchor_rec.anchor_type == "random":
                evolving_rec.p2_random_w += wins_for_evolving + 0.5 * d
                evolving_rec.p2_random_g += total_g
            elif anchor_rec.anchor_type == "smart":
                evolving_rec.p2_smart_w  += wins_for_evolving + 0.5 * d
                evolving_rec.p2_smart_g  += total_g

        if is_anchor_game:
            if not ra.is_anchor and rb.is_anchor:
                _update_type_stats(ra, rb, wa, d)
            elif ra.is_anchor and not rb.is_anchor:
                _update_type_stats(rb, ra, wb, d)

        if verbose:
            tag = "  [anchor]" if is_anchor_game else ""
            print(f"  [{idx+1}/{total}] {ra.label():>12} vs {rb.label():<12} "
                  f"→ {wa}-{wb}-{d}  Elo: {ra.elo:.0f}/{rb.elo:.0f}{tag}")

    # convert accumulators to win-rates on each evolving agent
    for rec in evolving:
        rec.p2_wr_random = (rec.p2_random_w / rec.p2_random_g
                            if rec.p2_random_g > 0 else 0.0)
        rec.p2_wr_smart  = (rec.p2_smart_w  / rec.p2_smart_g
                            if rec.p2_smart_g  > 0 else 0.0)


def run_phase2(
    population:      list[AgentRecord],
    n_generations:   int,
    survival_ratio:  float,
    elite_count:     int,
    mutation_sigma:  float,
    pop_size:        int,
    rng:             random.Random,
    next_id:         int,
    history:         list,
    checkpoint_path: str,
    optimal_every:   int,
    games_per_pair:  int,
    verbose:         bool,
) -> tuple[list[AgentRecord], int, list]:
    """
    Phase 2 evolution loop.

    Each generation:
      1. Determine active anchors: always Random×N + Smart×N, plus Optimal
         every `optimal_every` generations.
      2. Run full round-robin (evolving + active anchors).
      3. Compute composite fitness for each evolving agent.
      4. Cull bottom half by composite fitness.
      5. Breed next generation.
    """
    # build the persistent anchor pool — multiple instances of Random and Smart
    # so the round-robin game ratio is roughly 50/50 self-play vs anchor games
    p2_anchors = make_p2_anchors()
    # separate out Optimal so we can inject it selectively
    optimal_anchor  = next(a for a in p2_anchors if a.anchor_type == "optimal")
    standing_anchors = [a for a in p2_anchors if a.anchor_type != "optimal"]

    survivors_n = max(elite_count + 1, int(pop_size * survival_ratio))

    for gen in range(n_generations):
        t0 = time.time()
        print(f"\n{'='*60}")
        print(f"  PHASE 2 | Generation {gen}  |  pop = {len(population)}")
        print(f"{'='*60}")

        # reset per-generation counters on all evolving agents
        for rec in population:
            rec.wins = rec.losses = rec.draws = 0

        # determine whether Optimal is injected this generation
        inject_optimal = (gen % optimal_every == 0)
        active_anchors = standing_anchors + ([optimal_anchor] if inject_optimal else [])
        if inject_optimal:
            print(f"  [stress-test] OptimalAgent injected this generation")
            optimal_anchor.wins = optimal_anchor.losses = optimal_anchor.draws = 0

        # ── tournament ────────────────────────────────────────────────────────
        run_p2_tournament(population, active_anchors, games_per_pair, verbose)

        # ── composite fitness and selection ───────────────────────────────────
        for rec in population:
            rec.p2_fitness = p2_composite_fitness(rec, population)

        population.sort(key=lambda r: r.p2_fitness, reverse=True)
        survivors = population[:survivors_n]

        # ── log this generation ───────────────────────────────────────────────
        best = survivors[0]
        entry = {
            "phase":          2,
            "generation":     gen,
            "best_id":        best.agent_id,
            "best_fitness":   round(best.p2_fitness, 4),
            "best_wr_random": round(best.p2_wr_random, 3),
            "best_wr_smart":  round(best.p2_wr_smart, 3),
            "best_elo":       round(best.elo, 1),
            "mean_fitness":   round(sum(r.p2_fitness for r in survivors) / len(survivors), 4),
            "anchor_elos":    {a.label(): round(a.elo, 1) for a in active_anchors},
            "best_weights":   best.weights,
            "optimal_injected": inject_optimal,
        }
        history.append(entry)

        # print leaderboard
        print(f"\n  ── Phase 2 leaderboard (top 5) ──────────────────")
        for rank, rec in enumerate(survivors[:5], 1):
            print(f"  #{rank}  {rec.label():<8}  fit={rec.p2_fitness:.3f}  "
                  f"wr_rand={rec.p2_wr_random:.2f}  wr_smart={rec.p2_wr_smart:.2f}  "
                  f"elo={rec.elo:.0f}  born={rec.generation_born}")
        print(f"\n  ── Anchor Elos ───────────────────────────────────")
        for a in active_anchors:
            print(f"       {a.label():<14}  Elo={a.elo:.1f}  "
                  f"W/L/D={a.wins}/{a.losses}/{a.draws}")
        print()

        # ── reproduction ──────────────────────────────────────────────────────
        new_population = list(survivors[:elite_count])   # elites unchanged

        while len(new_population) < pop_size:
            # softmax over composite fitness for parent selection.
            # ×10 scaling sharpens the distribution so fitter agents are
            # meaningfully more likely to be selected (not just marginally).
            fits    = [r.p2_fitness for r in survivors]
            min_fit = min(fits)
            sel_w   = [math.exp((f - min_fit) * 10) for f in fits]
            total_w = sum(sel_w)
            probs   = [w / total_w for w in sel_w]

            pa = rng.choices(survivors, weights=probs)[0]
            pb = rng.choices(survivors, weights=probs)[0]
            while pb is pa and len(survivors) > 1:
                pb = rng.choices(survivors, weights=probs)[0]

            child = breed(pa, pb, next_id, gen, mutation_sigma)
            new_population.append(child)
            next_id += 1

        population = new_population
        elapsed = time.time() - t0
        print(f"  Phase 2 gen {gen} complete in {elapsed:.1f}s")

        save_checkpoint(checkpoint_path, "phase2", gen + 1, population,
                        [a.to_dict() for a in p2_anchors], history, next_id)

    return population, next_id, history


# ─────────────────────────────────────────────────────────────────────────────
# Mutation, crossover, breeding (shared)
# ─────────────────────────────────────────────────────────────────────────────

def mutate(weights: list, sigma: float = 0.15,
           mutation_rate: float = 0.4) -> list:
    """
    Apply Gaussian noise to a weight vector. Each weight is independently
    mutated with probability mutation_rate. Noise magnitude scales with
    the weight's current value so large and small weights shift proportionally.
    Returns a new list — the input is never modified.
    """
    new_w = []
    for w in weights:
        if random.random() < mutation_rate:
            delta = random.gauss(0, sigma * (abs(w) + 0.1))
            w += delta
        new_w.append(round(max(WEIGHT_MIN, min(WEIGHT_MAX, w)), 4))
    return new_w


def crossover(w_a: list, w_b: list) -> list:
    """
    Uniform crossover: each gene independently inherited from parent A or B
    with 50/50 probability. More diverse than single-point crossover because
    genes can mix from anywhere in the vector, not just two halves.
    """
    return [wa if random.random() < 0.5 else wb for wa, wb in zip(w_a, w_b)]


def breed(parent_a: AgentRecord, parent_b: AgentRecord,
          child_id: int, generation: int, sigma: float) -> AgentRecord:
    """Crossover then mutate — order matters (combine traits, then add novelty)."""
    w = crossover(parent_a.weights, parent_b.weights)
    w = mutate(w, sigma=sigma)
    return AgentRecord(
        agent_id        = child_id,
        weights         = w,
        generation_born = generation,
        parent_ids      = [parent_a.agent_id, parent_b.agent_id],
    )


def perturb_default(agent_id: int, generation: int,
                    sigma: float = 0.2) -> AgentRecord:
    """
    Seed the initial population by perturbing DEFAULT_WEIGHTS.
    mutation_rate=1.0 perturbs every weight (not just 40%) for maximum
    initial diversity. Higher sigma (0.2 vs 0.15) broadens the starting
    gene pool before competition pressure narrows it.
    """
    w = mutate(list(DEFAULT_WEIGHTS), sigma=sigma, mutation_rate=1.0)
    return AgentRecord(agent_id=agent_id, weights=w, generation_born=generation)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint I/O
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(path: str, phase: str, generation: int,
                    population: list[AgentRecord],
                    anchor_dicts: list,
                    history: list, next_id: int):
    """Save full state after every generation so crashes lose at most one gen."""
    data = {
        "phase":      phase,       # "phase1" or "phase2"
        "generation": generation,  # next gen to run on resume
        "next_id":    next_id,
        "population": [r.to_dict() for r in population],
        "anchors":    anchor_dicts,
        "history":    history,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  [checkpoint] {phase} gen {generation} saved → {path}")


def load_checkpoint(path: str):
    """Restore state from checkpoint. Returns (phase, generation, population, anchors, history, next_id)."""
    with open(path) as f:
        data = json.load(f)
    population = [AgentRecord.from_dict(d) for d in data["population"]]
    anchors    = [AgentRecord.from_dict(d) for d in data.get("anchors", [])]
    return (data["phase"], data["generation"], population,
            anchors, data.get("history", []), data.get("next_id", 1000))


# ─────────────────────────────────────────────────────────────────────────────
# Final assessment
# ─────────────────────────────────────────────────────────────────────────────

def final_assessment(best: AgentRecord, n_games: int = 20) -> dict:
    """
    Official post-training evaluation of the best evolved agent against
    Random and Smart only (the stated final assessment opponents).
    Optimal is excluded — it was a training stressor, not a benchmark target.
    Each opponent plays n_games seeds × 2 orderings to cancel positional bias.
    """
    results = {}
    for anchor_type, label in [("random", "Random"), ("smart", "Smart")]:
        anchor = AgentRecord(-99, None, is_anchor=True,
                             anchor_type=anchor_type, anchor_seed=0)
        wins = draws = 0
        for seed in range(n_games):
            for swap in [False, True]:
                ra, rb = (best, anchor) if not swap else (anchor, best)
                result = run_game(ra, rb, seed)
                # map result back to the evolved agent's perspective
                evolved_slot = 1 if swap else 0
                if result == evolved_slot:   wins  += 1
                elif result is None:         draws += 1
        total = n_games * 2
        wr = (wins + 0.5 * draws) / total
        results[label] = round(wr, 3)
        print(f"  vs {label:<6}: WR={wr:.3f}  "
              f"({wins}W / {total-wins-draws}L / {draws}D  of {total} games)")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Save results
# ─────────────────────────────────────────────────────────────────────────────

def save_results(path: str, best: AgentRecord,
                 history: list, assessment: dict):
    out = {
        "best_agent":       best.to_dict(),
        "default_weights":  DEFAULT_WEIGHTS,
        "weight_deltas":    [round(b - d, 4)
                             for b, d in zip(best.weights, DEFAULT_WEIGHTS)],
        "final_assessment": assessment,
        "history":          history,
    }
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Results saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Top-level entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(
    pop_size:        int   = 20,
    p1_generations:  int   = 15,    # anchor-only phase — builds baseline competence
    p2_generations:  int   = 35,    # mixed phase — refines via self-play arms race
    survival_ratio:  float = 0.5,
    elite_count:     int   = 1,
    mutation_sigma:  float = 0.15,
    games_per_pair:  int   = 2,     # Phase 2 self-play games per matchup
    optimal_every:   int   = OPTIMAL_STRESS_EVERY,
    checkpoint_path: str   = "evo_checkpoint.json",
    resume:          bool  = False,
    phase2_only:     bool  = False,  # skip Phase 1 and load its output directly
    verbose:         bool  = True,
):
    rng     = random.Random(42)   # seeded for reproducible parent selection
    history = []
    next_id = 0

    # ── initialise population ─────────────────────────────────────────────────
    if resume and os.path.exists(checkpoint_path):
        phase, start_gen, population, _, history, next_id = load_checkpoint(checkpoint_path)
        print(f"  Resumed: {phase} gen {start_gen}, pop={len(population)}")
        # adjust generation counts so we run the remaining gens, not the full count
        if phase == "phase1":
            p1_generations = max(0, p1_generations - start_gen)
        else:
            p2_generations = max(0, p2_generations - start_gen)
            phase2_only = True   # Phase 1 is already done
    else:
        population = [AgentRecord(next_id, list(DEFAULT_WEIGHTS), generation_born=0)]
        next_id += 1
        while len(population) < pop_size:
            population.append(perturb_default(next_id, 0, sigma=0.25))
            next_id += 1

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    if not phase2_only and p1_generations > 0:
        print(f"\n{'#'*60}")
        print(f"  STARTING PHASE 1 ({p1_generations} generations, anchor-only)")
        print(f"{'#'*60}")
        population, next_id, history = run_phase1(
            population, p1_generations, survival_ratio, elite_count,
            mutation_sigma, pop_size, rng, next_id, history,
            checkpoint_path, verbose,
        )
        best_p1 = max(population, key=lambda r: r.p1_fitness)
        print(f"\n  Phase 1 complete. Best: {best_p1.label()}  "
              f"fitness={best_p1.p1_fitness:.3f}  "
              f"wr_random={best_p1.p1_wr_random:.2f}  "
              f"wr_smart={best_p1.p1_wr_smart:.2f}")

    # ── Phase 2 ───────────────────────────────────────────────────────────────
    if p2_generations > 0:
        print(f"\n{'#'*60}")
        print(f"  STARTING PHASE 2 ({p2_generations} generations, mixed)")
        print(f"{'#'*60}")
        # Phase 2 Elo starts fresh — the Phase 1 population never played self-play
        # so their Elo of 1200 is a reasonable neutral starting point for Phase 2
        for rec in population:
            rec.elo = ELO_INIT
        population, next_id, history = run_phase2(
            population, p2_generations, survival_ratio, elite_count,
            mutation_sigma, pop_size, rng, next_id, history,
            checkpoint_path, optimal_every, games_per_pair, verbose,
        )

    # ── final report ──────────────────────────────────────────────────────────
    best = max(population, key=lambda r: getattr(r, "p2_fitness", r.p1_fitness))
    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE")
    print(f"  Best agent : {best.label()}  (born gen {best.generation_born})")
    print(f"\n  Evolved weights vs defaults:")
    for i, (we, wd) in enumerate(zip(best.weights, DEFAULT_WEIGHTS)):
        print(f"    w[{i:2d}] = {we:6.3f}  (default {wd:5.2f}, Δ={we-wd:+.3f})")

    print(f"\n{'='*60}")
    print(f"  FINAL ASSESSMENT")
    print(f"{'='*60}")
    assessment = final_assessment(best, n_games=20)
    save_results("evo_results.json", best, history, assessment)
    return best, history, assessment


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Two-phase MansaAgent curriculum trainer")
    p.add_argument("--pop",          type=int,   default=20,                    help="Population size")
    p.add_argument("--p1-gens",      type=int,   default=15,                    help="Phase 1 generations (anchor-only)")
    p.add_argument("--p2-gens",      type=int,   default=35,                    help="Phase 2 generations (mixed)")
    p.add_argument("--survive",      type=float, default=0.5,                   help="Survival ratio")
    p.add_argument("--sigma",        type=float, default=0.15,                  help="Mutation std-dev")
    p.add_argument("--elite",        type=int,   default=1,                     help="Elites preserved per gen")
    p.add_argument("--games",        type=int,   default=2,                     help="Phase 2 self-play games per pair")
    p.add_argument("--opt-every",    type=int,   default=OPTIMAL_STRESS_EVERY,  help="Phase 2 gens between Optimal injections")
    p.add_argument("--checkpoint",   type=str,   default="evo_checkpoint.json", help="Checkpoint file")
    p.add_argument("--resume",       action="store_true",                       help="Resume from checkpoint")
    p.add_argument("--phase2-only",  action="store_true",                       help="Skip Phase 1")
    p.add_argument("--quiet",        action="store_true",                       help="Suppress per-game output")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        pop_size       = args.pop,
        p1_generations = args.p1_gens,
        p2_generations = args.p2_gens,
        survival_ratio = args.survive,
        elite_count    = args.elite,
        mutation_sigma = args.sigma,
        games_per_pair = args.games,
        optimal_every  = args.opt_every,
        checkpoint_path= args.checkpoint,
        resume         = args.resume,
        phase2_only    = args.phase2_only,
        verbose        = not args.quiet,
    )