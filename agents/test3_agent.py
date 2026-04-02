from backend.trader import Trader, SellAction, TakeAction, TradeAction
from backend.goods import GoodType, Goods
from backend.market import MarketObservation
from backend.coins import BonusType


class AdvancedHeuristicV2Agent(Trader):
    """
    V2 non-RL Bazaar / Jaipur agent.

    Main upgrades over a strong heuristic baseline:
    1. One-ply evaluation for all legal actions
    2. Selective opponent-reply search on top-k candidate moves
    3. Opponent model built from observed actions
    4. Stronger endgame liquidation mode
    5. Much stricter trade discipline
    """

    def __init__(self, seed, name):
        super().__init__(seed, name)

        self.base_good_value = {
            GoodType.DIAMOND: 8.8,
            GoodType.GOLD: 7.8,
            GoodType.SILVER: 6.4,
            GoodType.FABRIC: 4.3,
            GoodType.SPICE: 4.1,
            GoodType.LEATHER: 3.2,
            GoodType.CAMEL: 1.0,
        }

        self.premium_goods = {
            GoodType.DIAMOND,
            GoodType.GOLD,
            GoodType.SILVER,
        }

        # Used in our own heuristic evaluator
        self.bonus_expectation = {
            3: 2.0,
            4: 5.0,
            5: 9.0,
        }

        # Tracks what opponent seems interested in
        self.opp_interest = {
            gt: 0.0 for gt in GoodType
        }

        # Estimated opponent holdings from observed actions
        # This is only an estimate / lower-bound style model
        self.opp_goods_est = {
            gt: 0 for gt in GoodType
        }

        # Since each player initially receives 5 goods in BasicBazaar,
        # but we do not observe opponent's initial hand, this acts as a
        # soft estimate of "unknown hidden cards". We don't use it directly
        # as exact holdings, but we keep it for possible extensions.
        self.opp_unknown_cards = 5

        # Search control
        self.top_k_for_reply_search = 5

    def select_action(self, actions, observation, simulate_action_fnc):
        if not actions:
            return None

        deck_left = observation.market_reserved_goods_count
        empty_stacks = self._count_empty_coin_stacks(observation)

        earlygame = deck_left >= 30
        endgame = (deck_left <= 10) or (empty_stacks >= 2)
        ultra_endgame = (deck_left <= 5) or (empty_stacks >= 2 and deck_left <= 12)

        # Phase 1: score all actions with our own evaluator
        scored = []
        for action in actions:
            try:
                future_obs = simulate_action_fnc(observation, action)
            except Exception:
                future_obs = None

            score = self._score_action(
                action=action,
                obs=observation,
                future_obs=future_obs,
                earlygame=earlygame,
                endgame=endgame,
                ultra_endgame=ultra_endgame
            )
            score += self._tiebreak(action, observation, earlygame, endgame)
            scored.append((score, action, future_obs))

        scored.sort(key=lambda x: x[0], reverse=True)

        # If only one move or we are in a trivial spot
        if len(scored) == 1:
            return scored[0][1]

        # Phase 2: selective opponent-reply search on the top-k moves only
        refined = []
        search_count = min(self.top_k_for_reply_search, len(scored))

        for i, (base_score, action, future_obs) in enumerate(scored):
            total_score = base_score

            if i < search_count and future_obs is not None:
                opp_reply_penalty = self._estimate_opponent_best_reply_value(
                    future_obs=future_obs,
                    endgame=endgame,
                    ultra_endgame=ultra_endgame
                )

                # In late game, opponent reply matters more
                if ultra_endgame:
                    total_score -= 1.15 * opp_reply_penalty
                elif endgame:
                    total_score -= 1.00 * opp_reply_penalty
                else:
                    total_score -= 0.80 * opp_reply_penalty

            refined.append((total_score, action))

        refined.sort(key=lambda x: x[0], reverse=True)
        return refined[0][1]

    def calculate_reward(self, old_observation, new_observation, has_acted, environment_reward):
        """
        No online learning.
        We only update a lightweight opponent model from observed actions.
        """
        try:
            act = new_observation.action
            if act is None:
                return

            # If this was NOT our turn, then opponent acted.
            if not has_acted:
                self._update_opponent_model_from_action(act)

            # Soft decay opponent-interest every turn
            for gt in GoodType:
                self.opp_interest[gt] *= 0.90

        except Exception:
            pass

    # ============================================================
    # Main scoring
    # ============================================================

    def _score_action(self, action, obs, future_obs, earlygame, endgame, ultra_endgame):
        score = 0.0

        if isinstance(action, SellAction):
            score += self._score_sell(action, obs, endgame, ultra_endgame)
        elif isinstance(action, TakeAction):
            score += self._score_take(action, obs, earlygame, endgame, ultra_endgame)
        elif isinstance(action, TradeAction):
            score += self._score_trade(action, obs, earlygame, endgame, ultra_endgame)

        if future_obs is not None:
            score += 0.90 * self._state_value(future_obs, endgame, ultra_endgame)
            score += self._denial_value(obs, future_obs, endgame, ultra_endgame)
            score += self._post_action_shape_bonus(future_obs, endgame, ultra_endgame)

        return score

    # ============================================================
    # Sell
    # ============================================================

    def _score_sell(self, action, obs, endgame, ultra_endgame):
        gt = action._sell
        count = action._count

        coins = obs.market_goods_coins[gt]
        if len(coins) < count:
            return -10**6

        immediate = sum(coins[-count:])
        score = 0.0

        # Immediate real points matter a lot
        score += 2.7 * immediate

        # Bonus expectation
        if count >= 5:
            score += 2.2 * self.bonus_expectation[5]
        elif count == 4:
            score += 2.0 * self.bonus_expectation[4]
        elif count == 3:
            score += 1.7 * self.bonus_expectation[3]

        owned = obs.actor_goods[gt]
        remaining_after = owned - count

        # Premium goods: sell earlier because stacks are short/high-value
        if gt in self.premium_goods:
            score += 2.5 + 0.7 * len(coins)

            # Strong endgame push to liquidate premium
            if count >= 2:
                score += 2.0

        # Common goods: prefer 3/4/5 bundles, dislike fragmenting too early
        else:
            if not endgame:
                if owned >= 4 and count <= 2:
                    score -= 5.0
                elif owned >= 3 and count == 1:
                    score -= 6.0

            if count >= 5:
                score += 8.0
            elif count == 4:
                score += 5.0
            elif count == 3:
                score += 3.0

        # Late-game liquidation mode
        if endgame:
            score += 1.8 * count
            if remaining_after == 0:
                score += 2.0

        if ultra_endgame:
            score += 2.4 * count
            # In ultra endgame, concrete conversion dominates speculative holding
            if gt not in self.premium_goods and count >= 2:
                score += 1.5
            if gt in self.premium_goods and count >= 2:
                score += 2.5

        return score

    # ============================================================
    # Take
    # ============================================================

    def _score_take(self, action, obs, earlygame, endgame, ultra_endgame):
        gt = action._take
        count = action._count

        # Camel take
        if gt == GoodType.CAMEL:
            my_camels = obs.actor_goods[GoodType.CAMEL]
            hand_fullish = obs.actor_non_camel_goods_count >= obs.max_player_goods_count - 1

            score = 0.0
            score += 3.0 * count

            if count >= 3:
                score += 5.0
            elif count == 2:
                score += 2.0

            if my_camels < 4:
                score += 1.5

            if hand_fullish:
                score += 4.0

            # Denial of opponent camel sweep
            if obs.market_goods[GoodType.CAMEL] >= 3:
                score += 2.5

            # Lone camel is often weak in early game when good market exists
            if earlygame and count == 1:
                premium_visible = sum(obs.market_goods[g] for g in self.premium_goods)
                if premium_visible >= 1:
                    score -= 2.5

            if endgame:
                score += 1.0 * count
            if ultra_endgame:
                score += 1.5 * count

            return score

        # Non-camel take
        hand_space = obs.max_player_goods_count - obs.actor_non_camel_goods_count
        if hand_space <= 0:
            return -10**6

        owned = obs.actor_goods[gt]
        new_count = owned + 1
        coins = obs.market_goods_coins[gt]
        top_coin = coins[-1] if coins else 0

        score = 0.0
        score += 1.8 * self.base_good_value[gt]
        score += 0.9 * top_coin
        score += 0.3 * len(coins)

        # Set thresholds
        if new_count == 2:
            score += 1.5 if gt in self.premium_goods else 0.8
        elif new_count == 3:
            score += 6.5
        elif new_count == 4:
            score += 7.5
        elif new_count >= 5:
            score += 8.0

        if gt in self.premium_goods:
            score += 2.8

        # Opponent denial
        score += 0.9 * self.opp_interest.get(gt, 0.0)

        # Full hand penalty
        if hand_space == 1:
            score -= 3.0
            if new_count < 3:
                score -= 3.0

        # Late game becomes much less speculative
        if endgame:
            if gt in self.premium_goods and new_count >= 2:
                score += 2.0
            elif gt not in self.premium_goods and new_count >= 2:
                score += 1.0
            else:
                score -= 2.0

        if ultra_endgame:
            if gt in self.premium_goods and new_count >= 2:
                score += 3.0
            elif gt not in self.premium_goods and new_count >= 3:
                score += 2.0
            else:
                score -= 4.0

        return score

    # ============================================================
    # Trade
    # ============================================================

    def _score_trade(self, action, obs, earlygame, endgame, ultra_endgame):
        req = action.requested_goods
        off = action.offered_goods

        total_taken = req.count()
        total_offered = off.count()

        # Safety
        if total_taken != total_offered or total_taken < 2:
            return -10**6

        score = 0.0

        value_gained = 0.0
        value_lost = 0.0

        premium_taken = 0
        junk_given = 0
        camel_offered = off[GoodType.CAMEL]

        concentration_bonus = 0.0
        dilution_penalty = 0.0
        immediate_sell_creation = 0.0

        for gt in GoodType:
            if gt == GoodType.CAMEL:
                continue

            value_gained += req[gt] * self.base_good_value[gt]
            value_lost += off[gt] * self.base_good_value[gt]

            if gt in self.premium_goods:
                premium_taken += req[gt]
            else:
                junk_given += off[gt]

            before = obs.actor_goods[gt]
            after = before - off[gt] + req[gt]

            if before < 2 <= after and gt in self.premium_goods:
                immediate_sell_creation += 5.0
            if before < 1 <= after and gt not in self.premium_goods:
                immediate_sell_creation += 1.0
            if before < 3 <= after and gt not in self.premium_goods:
                immediate_sell_creation += 8.0

            if before < 3 <= after:
                concentration_bonus += 7.0
            if before < 4 <= after:
                concentration_bonus += 5.0
            if before < 5 <= after:
                concentration_bonus += 4.0

            if before >= 3 and after <= 1:
                dilution_penalty += 4.0

        score += 2.2 * (value_gained - value_lost)
        score -= 1.4 * camel_offered
        score += concentration_bonus
        score += immediate_sell_creation
        score -= dilution_penalty

        # Reward junk -> premium swaps
        score += 2.0 * min(premium_taken, junk_given)

        # Stronger trade discipline:
        # if trade doesn't clearly improve hand concentration, punish it
        if concentration_bonus < 5.0 and immediate_sell_creation < 5.0:
            score -= 6.0

        # Very large trades are risky unless they are clearly strong
        if total_taken >= 4 and concentration_bonus < 7.0:
            score -= 4.0

        # Endgame: only allow trades that are almost immediately monetizable
        if endgame and immediate_sell_creation < 5.0:
            score -= 7.0

        if ultra_endgame and immediate_sell_creation < 7.0:
            score -= 12.0

        # Opponent denial from removing goods they seem to want
        denial = 0.0
        for gt in GoodType:
            if gt != GoodType.CAMEL:
                denial += req[gt] * self.opp_interest.get(gt, 0.0)
        score += 0.6 * denial

        return score

    # ============================================================
    # State evaluation
    # ============================================================

    def _state_value(self, obs, endgame, ultra_endgame):
        val = 0.0

        # Banked score
        val += self._banked_coin_total(obs)

        # Camel flexibility
        my_camels = obs.actor_goods[GoodType.CAMEL]
        val += 0.45 * min(my_camels, 6)
        if endgame:
            val += 0.40 * min(my_camels, 8)
        if ultra_endgame:
            val += 0.55 * min(my_camels, 8)

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

            if gt in self.premium_goods:
                if n >= 2:
                    val += 2.5 * n + 0.9 * (next_coin + second_coin)
                else:
                    val += 0.7 * n
            else:
                if n >= 3:
                    val += 2.2 * n + self.bonus_expectation[3]
                elif n == 2:
                    val += 1.0 * n
                else:
                    val += 0.4

            if n == 3:
                val += 4.0
            elif n == 4:
                val += 6.5
            elif n >= 5:
                val += 8.5

        # Hand clogging
        non_camels = obs.actor_non_camel_goods_count
        if non_camels >= 7:
            val -= 7.0
        elif non_camels == 6:
            val -= 3.5

        # Visible premium market is opportunity, but only if we can exploit it
        for gt in self.premium_goods:
            if obs.market_goods[gt] > 0:
                val += 0.5

        # Endgame shifts toward liquidation
        if endgame:
            for gt in GoodType:
                if gt == GoodType.CAMEL:
                    continue
                n = obs.actor_goods[gt]
                if gt in self.premium_goods and n >= 2:
                    val += 2.5
                elif gt not in self.premium_goods and n >= 1:
                    val += 1.0 * n

        if ultra_endgame:
            for gt in GoodType:
                if gt == GoodType.CAMEL:
                    continue
                n = obs.actor_goods[gt]
                if gt in self.premium_goods and n >= 2:
                    val += 4.0
                elif gt not in self.premium_goods and n >= 2:
                    val += 2.0
                else:
                    val -= 0.5 * n

        return val

    def _post_action_shape_bonus(self, obs, endgame, ultra_endgame):
        score = 0.0

        distinct = 0
        sellable_types = 0

        for gt in GoodType:
            if gt == GoodType.CAMEL:
                continue

            if obs.actor_goods[gt] > 0:
                distinct += 1

            n = obs.actor_goods[gt]
            min_sell = 2 if gt in self.premium_goods else 1
            if n >= min_sell:
                sellable_types += 1

        if distinct <= 2:
            score += 3.5
        elif distinct == 3:
            score += 1.2
        elif distinct >= 5:
            score -= 3.0

        score += 1.4 * sellable_types

        if endgame:
            score += 1.0 * sellable_types
        if ultra_endgame:
            score += 1.2 * sellable_types

        return score

    def _denial_value(self, old_obs, new_obs, endgame, ultra_endgame):
        score = 0.0

        for gt in GoodType:
            removed = old_obs.market_goods[gt] - new_obs.market_goods[gt]
            if removed > 0:
                score += removed * 0.9 * self.opp_interest.get(gt, 0.0)

        exposed_premium = sum(new_obs.market_goods[gt] for gt in self.premium_goods)
        score -= 1.1 * exposed_premium

        if new_obs.market_goods[GoodType.CAMEL] >= 3:
            score -= 2.5

        if endgame:
            score *= 1.20
        if ultra_endgame:
            score *= 1.30

        return score

    # ============================================================
    # Opponent reply search
    # ============================================================

    def _estimate_opponent_best_reply_value(self, future_obs, endgame, ultra_endgame):
        """
        Build a pseudo-observation for the opponent using our estimated opponent hand,
        then choose the reply that a simple greedy agent would most likely prefer.

        This is not exact hidden-state search. It is exploitation-oriented approximation.
        """
        opp_obs = self._build_estimated_opponent_observation(future_obs)
        opp_actions = self._generate_estimated_opponent_actions(opp_obs)

        if not opp_actions:
            return 0.0

        best = float("-inf")
        for act in opp_actions:
            val = self._score_action_like_simple_agent(act, opp_obs)

            # Late game: selling becomes even more important
            if ultra_endgame and isinstance(act, SellAction):
                val += 4.0
            elif endgame and isinstance(act, SellAction):
                val += 2.0

            if val > best:
                best = val

        return max(best, 0.0)

    def _build_estimated_opponent_observation(self, future_obs):
        """
        Constructs a pseudo MarketObservation using:
        - estimated opponent goods
        - current visible market from future_obs
        - current coin stacks / bonus counts from future_obs
        """
        est_goods = Goods.from_dict(self.opp_goods_est)

        empty_goods_coins = {gt: [] for gt in GoodType}
        empty_bonus_counts = {b: 0 for b in BonusType}

        opp_obs = MarketObservation(
            observer_id=future_obs.actor,
            actor_id=future_obs.actor,
            action=future_obs.action,
            actor_goods=est_goods,
            actor_goods_coins=empty_goods_coins,
            actor_bonus_coins_counts=empty_bonus_counts,
            market_goods=future_obs.market_goods,
            market_goods_coins=future_obs.market_goods_coins,
            market_bonus_coins_counts=future_obs.market_bonus_coins_counts,
            market_reserved_goods_count=future_obs.market_reserved_goods_count,
            max_player_goods_count=future_obs.max_player_goods_count,
            max_market_goods_count=future_obs.max_market_goods_count
        )
        return opp_obs

    def _generate_estimated_opponent_actions(self, opp_obs):
        """
        Uses engine action generators on our estimated opponent observation.
        """
        acts = []
        try:
            acts.extend(TradeAction.all_actions(opp_obs))
        except Exception:
            pass

        try:
            acts.extend(SellAction.all_actions(opp_obs))
        except Exception:
            pass

        try:
            acts.extend(TakeAction.all_actions(opp_obs))
        except Exception:
            pass

        return acts

    def _score_action_like_simple_agent(self, action, observation):
        """
        Approximate the provided simple-agent logic so we can predict its next move.
        """
        if isinstance(action, SellAction):
            return self._simple_eval_sell(action, observation)
        if isinstance(action, TakeAction):
            return self._simple_eval_take(action, observation)
        if isinstance(action, TradeAction):
            return self._simple_eval_trade(action, observation)
        return -10**6

    def _simple_eval_sell(self, action, observation):
        gt = action._sell
        count = action._count

        available_coins = observation.market_goods_coins.get(gt, [])
        if len(available_coins) < count:
            return -1000

        coins_to_receive = available_coins[-count:]
        coin_value = sum(coins_to_receive)

        bonus_multiplier = 1.0
        if count >= 5:
            bonus_multiplier = 3.0
        elif count >= 4:
            bonus_multiplier = 2.5
        elif count >= 3:
            bonus_multiplier = 2.0

        scarcity_bonus = 0
        if gt in self.premium_goods:
            scarcity_bonus = len(available_coins) * 2

        return (coin_value * bonus_multiplier) + scarcity_bonus

    def _simple_eval_take(self, action, observation):
        gt = action._take
        count = action._count

        if gt == GoodType.CAMEL:
            camel_value = 5 if observation.market_reserved_goods_count < 15 else 2
            return camel_value

        hand_space = observation.max_player_goods_count - observation.actor_non_camel_goods_count
        if hand_space <= 2:
            return -500

        good_value = {
            GoodType.DIAMOND: 7,
            GoodType.GOLD: 6,
            GoodType.SILVER: 5,
            GoodType.FABRIC: 4,
            GoodType.SPICE: 4,
            GoodType.LEATHER: 3,
            GoodType.CAMEL: 1
        }.get(gt, 1)

        available_coins = observation.market_goods_coins.get(gt, [])
        coins_remaining = len(available_coins)

        actor_goods = observation.actor_goods
        cards_of_type = actor_goods[gt]

        bonus_potential = 0
        if cards_of_type + count >= 5:
            bonus_potential = 30
        elif cards_of_type + count >= 4:
            bonus_potential = 20
        elif cards_of_type + count >= 3:
            bonus_potential = 15

        coin_availability_bonus = coins_remaining * 2

        return (good_value * 5) + bonus_potential + coin_availability_bonus

    def _simple_eval_trade(self, action, observation):
        requested = action.requested_goods
        offered = action.offered_goods

        simple_values = {
            GoodType.DIAMOND: 7,
            GoodType.GOLD: 6,
            GoodType.SILVER: 5,
            GoodType.FABRIC: 4,
            GoodType.SPICE: 4,
            GoodType.LEATHER: 3,
            GoodType.CAMEL: 1
        }

        value_gained = sum(
            simple_values.get(gt, 0) * requested[gt]
            for gt in GoodType
        )
        value_lost = sum(
            simple_values.get(gt, 0) * offered[gt]
            for gt in GoodType
        )

        net_value = value_gained - value_lost

        if offered[GoodType.CAMEL] > 0:
            net_value -= offered[GoodType.CAMEL] * 3

        actor_goods = observation.actor_goods
        set_bonus = 0

        for gt in GoodType:
            if gt == GoodType.CAMEL:
                continue

            new_count = actor_goods[gt] - offered[gt] + requested[gt]
            if new_count >= 5:
                set_bonus += 15
            elif new_count >= 4:
                set_bonus += 10
            elif new_count >= 3:
                set_bonus += 5

        return (net_value * 3) + set_bonus - 5

    # ============================================================
    # Opponent-model updates
    # ============================================================

    def _update_opponent_model_from_action(self, act):
        """
        Update estimated opponent holdings from observed opponent action.
        This is still approximate because opponent initial hand is hidden.
        """
        if isinstance(act, TakeAction):
            self.opp_goods_est[act._take] += act._count
            self.opp_interest[act._take] += act._count

        elif isinstance(act, SellAction):
            # If estimate is low because of hidden initial hand, clamp at zero
            self.opp_goods_est[act._sell] = max(0, self.opp_goods_est[act._sell] - act._count)
            self.opp_interest[act._sell] = max(0.0, self.opp_interest[act._sell] - 0.5 * act._count)

        elif isinstance(act, TradeAction):
            for gt in GoodType:
                got = act.requested_goods[gt]
                gave = act.offered_goods[gt]

                if got > 0:
                    self.opp_goods_est[gt] += got
                    self.opp_interest[gt] += got

                if gave > 0:
                    self.opp_goods_est[gt] = max(0, self.opp_goods_est[gt] - gave)
                    self.opp_interest[gt] = max(0.0, self.opp_interest[gt] - 0.25 * gave)

        # Prevent ridiculous estimated hand sizes
        self._normalize_opponent_estimate()

    def _normalize_opponent_estimate(self):
        total_noncamel = sum(
            self.opp_goods_est[gt]
            for gt in GoodType if gt != GoodType.CAMEL
        )

        # Hand cap from rules
        max_noncamel = 7
        if total_noncamel > max_noncamel:
            overflow = total_noncamel - max_noncamel

            # Remove overflow from least valuable common goods first
            shrink_order = [
                GoodType.LEATHER,
                GoodType.SPICE,
                GoodType.FABRIC,
                GoodType.SILVER,
                GoodType.GOLD,
                GoodType.DIAMOND,
            ]

            for gt in shrink_order:
                if overflow <= 0:
                    break
                removable = min(self.opp_goods_est[gt], overflow)
                self.opp_goods_est[gt] -= removable
                overflow -= removable

    # ============================================================
    # Utilities
    # ============================================================

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
        if isinstance(action, SellAction):
            return 0.08 * action._count

        if isinstance(action, TakeAction):
            if action._take == GoodType.CAMEL:
                return 0.02 if earlygame else 0.04
            return 0.015 * self.base_good_value[action._take]

        if isinstance(action, TradeAction):
            return -0.01 * action.requested_goods.count()

        return 0.0