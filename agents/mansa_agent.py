from backend.trader import Trader, TraderAction, SellAction, TakeAction, TradeAction
from backend.goods import GoodType


class MansaAgent(Trader):
    """
    Non-RL Jaipur/Bazaar agent:
    - 1-ply lookahead via simulate_action_fnc
    - phase-aware heuristic evaluation
    - better sell timing
    - market denial
    - camel management
    - selective trading only when it meaningfully improves hand quality
    """

    def __init__(self, seed, name):
        super().__init__(seed, name)

        # Base latent values, not coin values. Used for hand potential / trade quality.
        self.base_good_value = {
            GoodType.DIAMOND: 8.5,
            GoodType.GOLD: 7.5,
            GoodType.SILVER: 6.2,
            GoodType.FABRIC: 4.2,
            GoodType.SPICE: 4.0,
            GoodType.LEATHER: 3.2,
            GoodType.CAMEL: 1.0,
        }

        # Approximate expected bonus values.
        self.bonus_expectation = {
            3: 2.0,   # [1,1,2,2,2,3,3]
            4: 5.0,   # [4,4,5,5,6,6]
            5: 9.0,   # [8,8,9,10,10]
        }

        self.premium_goods = {
            GoodType.DIAMOND,
            GoodType.GOLD,
            GoodType.SILVER,
        }

        # Track rough opponent signal from what they just took/sold.
        self.opp_interest = {
            GoodType.DIAMOND: 0,
            GoodType.GOLD: 0,
            GoodType.SILVER: 0,
            GoodType.FABRIC: 0,
            GoodType.SPICE: 0,
            GoodType.LEATHER: 0,
            GoodType.CAMEL: 0,
        }

    def select_action(self, actions, observation, simulate_action_fnc):
        if not actions:
            return None

        # Cheap phase features
        deck_left = observation.market_reserved_goods_count
        empty_stacks = self._count_empty_coin_stacks(observation)
        endgame = (deck_left <= 10) or (empty_stacks >= 2)
        earlygame = deck_left >= 30

        best_action = None
        best_score = float("-inf")

        # Score every action by simulating the next observation
        for action in actions:
            try:
                future_obs = simulate_action_fnc(observation, action)
            except Exception:
                # Fail safe: never crash the game
                future_obs = None

            score = self._score_action(
                action=action,
                obs=observation,
                future_obs=future_obs,
                earlygame=earlygame,
                endgame=endgame
            )

            # Tiny deterministic tie-break by action type preference
            score += self._tiebreak(action, observation, earlygame, endgame)

            if score > best_score:
                best_score = score
                best_action = action

        return best_action if best_action is not None else self.rng.choice(actions)

    def calculate_reward(self, old_observation, new_observation, has_acted, environment_reward):
        # No online learning, but we can update lightweight opponent-interest memory
        try:
            act = new_observation.action
            if act is None:
                return

            if isinstance(act, TakeAction):
                self.opp_interest[act._take] += act._count

            elif isinstance(act, SellAction):
                # If opponent sells a type, they likely no longer want to hoard more of it immediately
                self.opp_interest[act._sell] = max(0, self.opp_interest[act._sell] - act._count)

            elif isinstance(act, TradeAction):
                for gt in GoodType:
                    self.opp_interest[gt] += act.requested_goods[gt]
                    self.opp_interest[gt] = max(0, self.opp_interest[gt] - act.offered_goods[gt])

            # Soft decay so this stays recent
            for gt in GoodType:
                self.opp_interest[gt] *= 0.85
        except Exception:
            pass

    # -------------------------
    # Main action scoring
    # -------------------------

    def _score_action(self, action, obs, future_obs, earlygame, endgame):
        score = 0.0

        # 1) Immediate tactical value of the move itself
        if isinstance(action, SellAction):
            score += self._score_sell(action, obs, endgame)
        elif isinstance(action, TakeAction):
            score += self._score_take(action, obs, earlygame, endgame)
        elif isinstance(action, TradeAction):
            score += self._score_trade(action, obs, earlygame, endgame)

        # 2) Lookahead state value
        if future_obs is not None:
            score += 0.85 * self._state_value(future_obs, endgame)

            # 3) Market denial / what we leave exposed
            score += self._denial_value(obs, future_obs, endgame)

            # 4) Post-action hand-shape adjustments
            score += self._post_action_shape_bonus(future_obs, endgame)

        return score

    # -------------------------
    # Sell scoring
    # -------------------------

    def _score_sell(self, action, obs, endgame):
        gt = action._sell
        count = action._count
        coins = obs.market_goods_coins[gt]

        if len(coins) < count:
            return -10**6

        immediate = sum(coins[-count:])

        # Expected bonus value
        bonus = 0.0
        if count >= 5 and obs.market_bonus_coins_counts.get(type("x", (), {"value": 5})(), 0) >= 0:
            bonus = self.bonus_expectation[5]
        elif count == 4:
            bonus = self.bonus_expectation[4]
        elif count == 3:
            bonus = self.bonus_expectation[3]

        # Preserve flexibility: dumping all medium goods too early can hurt
        remaining_after = obs.actor_goods[gt] - count

        # Premium goods should usually be sold earlier because their stacks are short/high
        premium_release = 0.0
        if gt in self.premium_goods:
            premium_release += 2.5
            premium_release += 0.6 * len(coins)

        # For common goods, reward 3+ bundle more than piecemeal selling
        bundle_bonus = 0.0
        if count >= 5:
            bundle_bonus += 8.0
        elif count == 4:
            bundle_bonus += 5.0
        elif count == 3:
            bundle_bonus += 2.5

        # Penalty if we are breaking a potentially better bundle too soon
        hoard_penalty = 0.0
        if not endgame:
            if gt not in self.premium_goods:
                if count == 1 and obs.actor_goods[gt] >= 3:
                    hoard_penalty += 5.0
                if count == 2 and obs.actor_goods[gt] >= 4:
                    hoard_penalty += 3.0
            else:
                # premium goods are less likely to improve by waiting too long
                if count == 2 and obs.actor_goods[gt] >= 4 and len(coins) >= 4:
                    hoard_penalty += 1.0

        # Endgame liquidation matters
        endgame_push = 0.0
        if endgame:
            endgame_push += 2.0 * count
            if remaining_after == 0:
                endgame_push += 2.5

        return (
            2.4 * immediate
            + 1.8 * bonus
            + premium_release
            + bundle_bonus
            + endgame_push
            - hoard_penalty
        )

    # -------------------------
    # Take scoring
    # -------------------------

    def _score_take(self, action, obs, earlygame, endgame):
        gt = action._take
        count = action._count

        if gt == GoodType.CAMEL:
            camels_taken = count
            market_non_camels_after = obs.max_market_goods_count - count

            # Camels are strongest when there are 2+ available,
            # when our hand is clogged,
            # or when they deny a juicy refill/tempo to opponent.
            score = 0.0
            score += 3.0 * camels_taken

            my_camels = obs.actor_goods[GoodType.CAMEL]
            if my_camels < 4:
                score += 2.0

            if obs.actor_non_camel_goods_count >= obs.max_player_goods_count - 1:
                score += 4.0

            if camels_taken >= 3:
                score += 4.5

            # Endgame camel-majority matters more
            if endgame:
                score += 1.5 * camels_taken

            # Early, avoid camel obsession when only 1 camel and good market exists
            if earlygame and camels_taken == 1:
                score -= 2.0

            return score

        # Non-camel single take
        hand_space = obs.max_player_goods_count - obs.actor_non_camel_goods_count
        if hand_space <= 0:
            return -10**6

        coins = obs.market_goods_coins[gt]
        top_coin = coins[-1] if coins else 0
        owned = obs.actor_goods[gt]

        score = 0.0

        # Latent value + top coin timing
        score += 1.7 * self.base_good_value[gt]
        score += 0.8 * top_coin
        score += 0.35 * len(coins)

        # Set completion incentives
        new_count = owned + 1
        if new_count == 2:
            score += 1.5 if gt in self.premium_goods else 1.0
        elif new_count == 3:
            score += 6.0
        elif new_count == 4:
            score += 7.0
        elif new_count >= 5:
            score += 7.5

        # Premium goods are strong to snipe when available
        if gt in self.premium_goods:
            score += 2.5

        # If hand is getting full, only take if it is really helping
        if hand_space == 1:
            score -= 2.5
            if new_count < 3:
                score -= 3.0

        # Endgame: prefer direct conversion, not speculative takes
        if endgame:
            if new_count >= 2:
                score += 2.0
            else:
                score -= 2.5

        # Opponent denial
        score += 0.6 * self.opp_interest.get(gt, 0)

        return score

    # -------------------------
    # Trade scoring
    # -------------------------

    def _score_trade(self, action, obs, earlygame, endgame):
        req = action.requested_goods
        off = action.offered_goods

        score = 0.0

        value_gained = 0.0
        value_lost = 0.0

        for gt in GoodType:
            if gt == GoodType.CAMEL:
                continue

            value_gained += req[gt] * self.base_good_value[gt]
            value_lost += off[gt] * self.base_good_value[gt]

        # Slightly separate camel cost from normal loss
        camel_offered = off[GoodType.CAMEL]

        score += 2.0 * (value_gained - value_lost)
        score -= 1.2 * camel_offered

        # Reward bundle formation, especially turning junk into concentrated sets
        concentration_bonus = 0.0
        dilution_penalty = 0.0

        for gt in GoodType:
            if gt == GoodType.CAMEL:
                continue

            before = obs.actor_goods[gt]
            after = before - off[gt] + req[gt]

            # Hitting these thresholds matters a lot
            if before < 3 <= after:
                concentration_bonus += 7.0
            if before < 4 <= after:
                concentration_bonus += 5.0
            if before < 5 <= after:
                concentration_bonus += 4.0

            # Trading away a completed/common bundle can be bad
            if before >= 3 and after <= 1:
                dilution_penalty += 3.0

        score += concentration_bonus - dilution_penalty

        # Reward "junk -> premium" trades
        premium_taken = req[GoodType.DIAMOND] + req[GoodType.GOLD] + req[GoodType.SILVER]
        junk_given = off[GoodType.LEATHER] + off[GoodType.SPICE] + off[GoodType.FABRIC]
        score += 2.0 * min(premium_taken, junk_given)

        # Penalize fancy trades that do not materially improve hand quality
        total_taken = req.count()
        if total_taken >= 4 and concentration_bonus < 5.0:
            score -= 3.5

        # Endgame: prefer trades only if they create near-immediate sells
        if endgame and concentration_bonus < 7.0:
            score -= 4.0

        # Opponent denial: remove goods they seem interested in
        denial = 0.0
        for gt in GoodType:
            if gt == GoodType.CAMEL:
                continue
            denial += req[gt] * self.opp_interest.get(gt, 0)
        score += 0.5 * denial

        return score

    # -------------------------
    # State evaluation
    # -------------------------

    def _state_value(self, obs, endgame):
        val = 0.0

        # Real scored money already banked
        val += self._banked_coin_total(obs)

        # Expected camel edge. We only know our count, not opponent's exact hand from obs,
        # so use a soft value rather than binary.
        my_camels = obs.actor_goods[GoodType.CAMEL]
        val += min(my_camels, 6) * 0.45
        if endgame:
            val += min(my_camels, 8) * 0.55

        # Hand potential
        for gt in GoodType:
            if gt == GoodType.CAMEL:
                continue

            n = obs.actor_goods[gt]
            if n <= 0:
                continue

            coins = obs.market_goods_coins[gt]
            next_coin = coins[-1] if len(coins) >= 1 else 0
            second_coin = coins[-2] if len(coins) >= 2 else 0

            # Convertibility: sets that are already sellable are valuable
            if gt in self.premium_goods:
                if n >= 2:
                    val += 2.2 * n + 0.8 * (next_coin + second_coin)
                else:
                    val += 0.8 * n
            else:
                if n >= 3:
                    val += 2.0 * n + self.bonus_expectation[3]
                elif n == 2:
                    val += 1.2 * n
                else:
                    val += 0.5

            # Threshold shaping
            if n == 3:
                val += 3.5
            elif n == 4:
                val += 6.0
            elif n >= 5:
                val += 8.0

        # Hand clog penalty
        non_camels = obs.actor_non_camel_goods_count
        if non_camels >= 7:
            val -= 6.0
        elif non_camels == 6:
            val -= 3.0

        # Market opportunity / danger
        for gt in self.premium_goods:
            if obs.market_goods[gt] > 0:
                val += 0.5  # good market for us if we can exploit it

        # Endgame shifts value toward immediate liquidation
        if endgame:
            for gt in GoodType:
                if gt == GoodType.CAMEL:
                    continue
                n = obs.actor_goods[gt]
                if gt in self.premium_goods and n >= 2:
                    val += 2.0
                elif gt not in self.premium_goods and n >= 1:
                    val += 0.8 * n

        return val

    def _post_action_shape_bonus(self, obs, endgame):
        score = 0.0

        # Reward hands with few distinct non-camel goods: concentrated hands sell better
        distinct = 0
        for gt in GoodType:
            if gt != GoodType.CAMEL and obs.actor_goods[gt] > 0:
                distinct += 1

        if distinct <= 2:
            score += 3.0
        elif distinct == 3:
            score += 1.0
        elif distinct >= 5:
            score -= 2.5

        # Light reward for being able to sell immediately next turn
        sellable_types = 0
        for gt in GoodType:
            if gt == GoodType.CAMEL:
                continue
            n = obs.actor_goods[gt]
            min_sell = 2 if gt in self.premium_goods else 1
            if n >= min_sell:
                sellable_types += 1
        score += 1.2 * sellable_types

        if endgame:
            score += 0.8 * sellable_types

        return score

    def _denial_value(self, old_obs, new_obs, endgame):
        score = 0.0

        # Goods removed from market that opponent might have wanted
        for gt in GoodType:
            removed = old_obs.market_goods[gt] - new_obs.market_goods[gt]
            if removed > 0:
                score += removed * 0.7 * self.opp_interest.get(gt, 0)

        # If we leave a premium-rich market untouched, that's risky
        exposed_premium = 0
        for gt in self.premium_goods:
            exposed_premium += new_obs.market_goods[gt]
        score -= 0.9 * exposed_premium

        # Leaving 3+ camels can let opponent reset market cheaply
        if new_obs.market_goods[GoodType.CAMEL] >= 3:
            score -= 2.0

        # Endgame: denial matters more because each move has larger swing
        if endgame:
            score *= 1.25

        return score

    # -------------------------
    # Utilities
    # -------------------------

    def _banked_coin_total(self, obs):
        total = 0
        for gt in GoodType:
            total += sum(obs.actor_goods_coins[gt])
        return total

    def _count_empty_coin_stacks(self, obs):
        cnt = 0
        for gt in GoodType:
            if gt == GoodType.CAMEL:
                continue
            if len(obs.market_goods_coins[gt]) == 0:
                cnt += 1
        return cnt

    def _tiebreak(self, action, obs, earlygame, endgame):
        # Small nudges only
        if isinstance(action, SellAction):
            # In ties, prefer converting points
            return 0.06 * action._count

        if isinstance(action, TakeAction):
            if action._take == GoodType.CAMEL:
                return 0.02 if earlygame else 0.04
            return 0.01 * self.base_good_value[action._take]

        if isinstance(action, TradeAction):
            # Break ties toward simpler trades
            return -0.01 * action.requested_goods.count()

        return 0.0