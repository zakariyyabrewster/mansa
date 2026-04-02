from backend.trader import Trader, TraderAction, SellAction, TakeAction, TradeAction
from backend.goods import GoodType
from backend.coins import BonusType


class MansaAgent(Trader):
    """
    Jaipur agent for ROB311 W26.

    Strategy overview
    -----------------
    Every legal action is scored by evaluating the *resulting* market
    observation (post-action, pre-opponent-turn).  The score is a linear
    combination of hand-quality, sell-urgency, market-awareness, and
    end-game-pressure features — no look-ahead by default.

    Key edges over SmartAgent
    -------------------------
    * Sell urgency fires early on premium goods when coin stacks are small,
      instead of hoarding for a bonus that yields less than selling now.
    * Trade value is judged by resulting hand composition, not a flat penalty.
    * End-game pressure continuously shifts the agent toward selling as the
      deck empties, preventing the hoard-and-lose failure mode.
    * Bonus token expected value is estimated from remaining tokens rather
      than a fixed multiplier.
    """

    # ------------------------------------------------------------------
    # Static game knowledge
    # ------------------------------------------------------------------

    # Base per-coin face values used when estimating future sell EV
    GOOD_BASE_VALUE = {
        GoodType.DIAMOND: 6.0,
        GoodType.GOLD:    5.5,
        GoodType.SILVER:  5.0,
        GoodType.FABRIC:  2.5,
        GoodType.SPICE:   2.5,
        GoodType.LEATHER: 1.5,
        GoodType.CAMEL:   0.0,
    }

    # Minimum cards required to sell each good
    MIN_SELL = {
        GoodType.DIAMOND: 2,
        GoodType.GOLD:    2,
        GoodType.SILVER:  2,
        GoodType.FABRIC:  1,
        GoodType.SPICE:   1,
        GoodType.LEATHER: 1,
        GoodType.CAMEL:   0,
    }

    # Full bonus token stacks (for EV estimation when no observation yet)
    FULL_BONUS_STACKS = {
        BonusType.THREE: [1, 1, 2, 2, 2, 3, 3],
        BonusType.FOUR:  [4, 4, 5, 5, 6, 6],
        BonusType.FIVE:  [8, 8, 9, 10, 10],
    }

    # Total non-camel goods in a standard game (for deck estimation)
    TOTAL_NON_CAMEL = 6 + 6 + 6 + 8 + 8 + 10  # diamond gold silver fabric spice leather

    def __init__(self, seed, name):
        super().__init__(seed, name)

    # ------------------------------------------------------------------
    # Required interface
    # ------------------------------------------------------------------

    def select_action(self, actions, observation, simulate_action_fnc):
        best_action = None
        best_score = float('-inf')

        for action in actions:
            try:
                next_obs = simulate_action_fnc(observation, action)
                score = self._score_action(action, observation, next_obs)
            except Exception:
                score = 0.0

            if score > best_score:
                best_score = score
                best_action = action

        return best_action if best_action is not None else actions[0]

    def calculate_reward(self, old_observation, new_observation,
                         has_acted, environment_reward):
        pass

    # ------------------------------------------------------------------
    # Master scoring function
    # ------------------------------------------------------------------

    def _score_action(self, action, obs, next_obs):
        """
        Score a (action, resulting_observation) pair.

        We decompose into:
          A) Immediate sell value (only for SellActions)
          B) Hand composition quality in the resulting state
          C) Sell urgency / opportunity cost
          D) End-game pressure adjustment
        """
        deck_size   = obs.market_reserved_goods_count
        total_goods = deck_size + obs.market_goods.count()
        phase       = self._game_phase(deck_size)   # 0.0 (early) → 1.0 (late)

        score = 0.0

        # ---- A. Immediate sell value --------------------------------
        if isinstance(action, SellAction):
            score += self._sell_score(action, obs, phase)

        # ---- B. Hand composition after action ----------------------
        score += self._hand_quality(next_obs, phase)

        # ---- C. Market opportunity / urgency signals ---------------
        score += self._market_urgency(obs, next_obs, action, phase)

        # ---- D. End-game pressure ----------------------------------
        score += self._endgame_pressure(next_obs, phase)

        return score

    # ------------------------------------------------------------------
    # Component A — sell score
    # ------------------------------------------------------------------

    def _sell_score(self, action, obs, phase):
        """
        Exact value of the coins we receive plus expected bonus token value,
        tempered by urgency: selling sooner is better when top coins are high
        and the stack is small.
        """
        good   = action._sell
        count  = action._count
        stack  = obs.market_goods_coins.get(good, [])

        if len(stack) < count:
            return -9999.0

        # Coins are sorted ascending; we pop from the top (highest first)
        coins_received = stack[-count:]
        coin_total     = sum(coins_received)

        # Expected bonus token value
        bonus_ev = 0.0
        if count >= 3:
            bonus_type = {3: BonusType.THREE, 4: BonusType.FOUR,
                          5: BonusType.FIVE}.get(min(count, 5))
            if bonus_type:
                bonus_stack = obs.market_bonus_coins_counts  # dict BonusType→count
                remaining   = bonus_stack.get(bonus_type, 0)
                full_stack  = self.FULL_BONUS_STACKS[bonus_type]
                # Average of the top `remaining` coins in the full stack
                if remaining > 0:
                    expected_bonus_coins = full_stack[-remaining:]
                    bonus_ev = sum(expected_bonus_coins) / len(expected_bonus_coins)

        # Urgency multiplier: rewards selling when stack is small
        # (we capture the high-value coins before opponent does)
        stack_size     = len(stack)
        urgency_mult   = 1.0 + phase * (1.0 / max(stack_size, 1))

        # Penalty for selling a large set when remaining stack coins are high
        # → only applies if we're NOT in end-game (we should still wait a bit)
        deferral_penalty = 0.0
        if phase < 0.5 and count < 5 and stack_size > count + 2:
            # There are still good coins left — maybe we can grow the set
            future_top = stack[-(count + 1)] if stack_size > count else 0
            deferral_penalty = -future_top * 0.3 * (1.0 - phase)

        sell_val = (coin_total + bonus_ev) * urgency_mult + deferral_penalty

        # Strong bonus for achieving count milestones
        if count >= 5:
            sell_val += 12.0
        elif count >= 4:
            sell_val += 7.0
        elif count >= 3:
            sell_val += 3.0

        return sell_val

    # ------------------------------------------------------------------
    # Component B — hand quality
    # ------------------------------------------------------------------

    def _hand_quality(self, next_obs, phase):
        """
        Assess how good our hand is *after* the action resolves.

        Factors
        -------
        * Secured coins (already banked — always good)
        * Set progress: how close each good type is to a sellable threshold,
          weighted by the expected value of that sell
        * Camel value (end-bonus potential + trade fuel)
        * Hand pressure: penalise being maxed out with no sell ready
        """
        score = 0.0

        # Secured goods coins (already earned — strongest signal)
        for good in GoodType:
            coins = next_obs.actor_goods_coins.get(good, [])
            score += sum(coins) * 1.5

        # Secured bonus coins count
        for bt in BonusType:
            score += next_obs.actor_bonus_coins_counts.get(bt, 0) * 4.0

        # Set progress for each good type
        actor_goods = next_obs.actor_goods
        for good in GoodType:
            if good == GoodType.CAMEL:
                continue
            count      = actor_goods[good]
            min_sell   = self.MIN_SELL[good]
            stack      = next_obs.market_goods_coins.get(good, [])
            stack_size = len(stack)

            if count == 0 or stack_size == 0:
                continue

            # Expected value of next sell at current count
            sell_count  = max(count, min_sell)
            sell_count  = min(sell_count, stack_size)
            if sell_count <= 0:
                continue

            expected_coins = stack[-sell_count:]
            ev_sell        = sum(expected_coins)

            # Scale by how close we are to threshold milestones
            milestone_bonus = 0.0
            if count >= 5:
                milestone_bonus = 10.0
            elif count >= 4:
                milestone_bonus = 6.0
            elif count >= 3:
                milestone_bonus = 3.0
            elif count >= min_sell:
                milestone_bonus = 1.0

            # Discount set progress when end-game pressure is high
            # (better to sell what we have than to wait)
            progress_weight = 0.6 - 0.3 * phase

            score += (ev_sell * progress_weight) + milestone_bonus

        # Camel value
        camel_count = actor_goods[GoodType.CAMEL]
        camel_val   = camel_count * (1.5 + 2.0 * phase)  # more valuable late
        score       += camel_val

        # Hand pressure penalty
        hand_used = next_obs.actor_non_camel_goods_count
        hand_max  = next_obs.max_player_goods_count
        if hand_used >= hand_max:
            score -= 8.0
        elif hand_used >= hand_max - 1:
            score -= 3.0

        return score

    # ------------------------------------------------------------------
    # Component C — market urgency
    # ------------------------------------------------------------------

    def _market_urgency(self, obs, next_obs, action, phase):
        """
        Reward actions that respond to market scarcity signals.

        * Premium goods (diamond/gold/silver) with small stacks → urgent
        * Taking a good that completes a set is high value
        * Penalise passing on a good the opponent likely wants too
        """
        score = 0.0

        # Scan current coin stacks for urgency
        for good in [GoodType.DIAMOND, GoodType.GOLD, GoodType.SILVER]:
            stack = obs.market_goods_coins.get(good, [])
            if not stack:
                continue
            top_coin   = stack[-1]
            stack_size = len(stack)
            # Scarcity signal: high-value coin + small stack = urgent
            scarcity   = top_coin / max(stack_size, 1)
            score      += scarcity * 0.4 * phase

        # Reward take/trade actions that move us closer to a sell threshold
        if isinstance(action, (TakeAction, TradeAction)):
            actor_goods  = next_obs.actor_goods
            prev_goods   = obs.actor_goods
            for good in GoodType:
                if good == GoodType.CAMEL:
                    continue
                prev_count = prev_goods[good]
                next_count = actor_goods[good]
                if next_count <= prev_count:
                    continue
                # We gained this good — did we cross a threshold?
                for threshold in (3, 4, 5):
                    if prev_count < threshold <= next_count:
                        stack    = next_obs.market_goods_coins.get(good, [])
                        top_coin = stack[-1] if stack else 0
                        score   += top_coin * 1.2

        return score

    # ------------------------------------------------------------------
    # Component D — end-game pressure
    # ------------------------------------------------------------------

    def _endgame_pressure(self, next_obs, phase):
        """
        As the deck empties, penalise holding unsellable goods and reward
        any conversion to coins.  Also flag approaching the 3-empty-stack
        terminal condition.
        """
        score = 0.0

        # Count nearly-empty coin stacks (approaching terminal condition)
        near_empty = sum(
            1 for good in GoodType
            if good != GoodType.CAMEL
            and len(next_obs.market_goods_coins.get(good, [])) <= 1
        )
        # The closer to 3 empty stacks, the more urgent everything is
        score -= near_empty * 3.0 * phase

        # Penalise holding goods whose coin stack is now empty (unsellable)
        actor_goods = next_obs.actor_goods
        for good in GoodType:
            if good == GoodType.CAMEL:
                continue
            stack = next_obs.market_goods_coins.get(good, [])
            if len(stack) == 0 and actor_goods[good] > 0:
                # Stuck holding worthless goods
                score -= actor_goods[good] * 4.0 * phase

        return score

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _game_phase(self, deck_size):
        """
        Smoothly interpolate from 0.0 (full deck) to 1.0 (empty deck).
        Uses TOTAL_NON_CAMEL as the denominator since camels don't run out.
        """
        return max(0.0, min(1.0, 1.0 - deck_size / self.TOTAL_NON_CAMEL))