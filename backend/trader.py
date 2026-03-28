from __future__ import annotations
from typing import Optional, Callable
from enum import Enum

from arelai.player import Player, Action

from .market import MarketObservation
from .goods import GoodType, Goods

class TraderActionType(Enum):
    TAKE    = "Take"
    SELL    = "Sell"
    TRADE   = "Trade"

class TraderAction(Action):
    def __init__(self,
                 trader_action_type: TraderActionType,
                 actor: Trader,
                 requested_goods: Goods,
                 offered_goods: Goods
                 ):
        self._trader_action_type = trader_action_type
        self.requested_goods = requested_goods
        self.offered_goods = offered_goods
        super().__init__(actor)

    @property
    def trader_action_type(self):
        return self._trader_action_type


class SellAction(TraderAction):
    MIN_SELL_COUNT = {
    GoodType.DIAMOND: 2,
    GoodType.GOLD: 2,
    GoodType.SILVER: 2,
    GoodType.FABRIC: 1,
    GoodType.SPICE: 1,
    GoodType.LEATHER: 1,
    GoodType.CAMEL: 0
    }
    
    def __init__(self, actor: Trader, sell: GoodType, count: int):
        
        self._sell = sell
        self._count = count

        requested_goods = Goods()
        offered_goods = Goods.from_dict({sell: count})
        
        super().__init__(
            TraderActionType.SELL,
            actor,
            requested_goods,
            offered_goods
            )
    
    def all_actions(observation: MarketObservation) -> list['SellAction']:
        actor = observation.actor
        actions = []

        market_goods = observation.market_goods
        if market_goods.count() < 5:
            return []
    
        actor_goods = observation.actor_goods
        for good_type in GoodType:
            if good_type == GoodType.CAMEL:
                continue
            for count in range(SellAction.MIN_SELL_COUNT[good_type], actor_goods[good_type]+1):
                action = SellAction(actor, good_type, count)
                actions.append(action)
        return actions
    
class TakeAction(TraderAction):
    def __init__(self, actor: Trader, take: GoodType, count: int):
        
        self._take = take
        self._count = count

        offered_goods = Goods()
        requested_goods = Goods.from_dict({take: count})

        super().__init__(
            TraderActionType.TAKE,
            actor,
            requested_goods,
            offered_goods)

    def all_actions(observation: MarketObservation) -> list['TakeAction']:
        actions = []
        
        market_goods = observation.market_goods
        actor_goods = observation.actor_goods

        actor = observation.actor

        if market_goods.count() < 5:
            return []

        # if taking camels, must take all camels
        if market_goods[GoodType.CAMEL] > 0:
            actions.append(TakeAction(actor, GoodType.CAMEL, market_goods[GoodType.CAMEL]))
        # otherwise, the actor cannot take more goods than he/she can hold
        if actor_goods.count(include_camels=False) < observation.max_player_goods_count:
            for good_type in GoodType:
                if good_type != GoodType.CAMEL:
                    if market_goods[good_type] > 0:
                        actions.append(TakeAction(actor, good_type, 1))
        return actions
    

class TradeAction(TraderAction):
    def __init__(self, actor: Trader, net: Goods):

        requested_goods = Goods()
        offered_goods = Goods()

        for good_type in GoodType:
            if net[good_type] > 0:
                for _ in range(net[good_type]):
                    requested_goods.add(good_type)
            elif net[good_type] < 0:
                for _ in range(abs(net[good_type])):
                    offered_goods.add(good_type)

        super().__init__(
            TraderActionType.TRADE,
            actor,
            requested_goods,
            offered_goods)

    def all_actions(observation: MarketObservation) -> list['TradeAction']:
        from itertools import product

        actor_goods = observation.actor_goods
        market_goods = observation.market_goods

        actor = observation.actor

        if market_goods.count() < 5:
            return []

        max_take = {gt: market_goods[gt] for gt in GoodType if gt != GoodType.CAMEL}
        max_give = {gt: actor_goods[gt] for gt in GoodType}

        # Generate all combinations of give/take from -max_give to +max_take
        ranges = {gt: range(-max_give[gt], max_take.get(gt, 0)+1) for gt in GoodType}
        all_possible = product(*[ranges[gt] for gt in GoodType])

        actions = []
        for combo in all_possible:
            action = TradeAction(actor, Goods.from_dict(dict(zip(GoodType, combo))))
            
            requested_goods = action.requested_goods
            offered_goods = action.offered_goods

            # no camels can be taken
            if requested_goods[GoodType.CAMEL] > 0:
                continue

            # the number of goods taken must equal the number of goods given
            if requested_goods.count() != offered_goods.count():
                continue

            # at least two goods must be taken
            if requested_goods.count() < 2:
                continue

            # cannot take and give cards of the same type
            same_type_trade = False
            for gt in GoodType:
                if requested_goods[gt] > 0 and offered_goods[gt] > 0:
                    same_type_trade = True
                    break
            if same_type_trade:
                continue
            
            # cannot take more than the actor can hold
            requested_non_camels_count = requested_goods.count(include_camels=False)
            if actor_goods.count(include_camels=False) + requested_non_camels_count > observation.max_player_goods_count:
                continue 

            # cannot give more than actor has or take more than market has
            violates = False
            if market_goods.count() < 5 and requested_goods.count() >= 1:
                violates = True
            for gt in GoodType:
                if requested_goods[gt] > market_goods[gt]:
                    violates = True
                if offered_goods[gt] > actor_goods[gt]:
                    violates = True
            if violates:
                continue
            actions.append(action)
        return actions


class Trader(Player):
    def __init__(self,
                 seed,
                 name):
        super().__init__(seed, name)

    def select_action(self,
                      actions: list[TraderAction],
                      observation: MarketObservation,
                      simulate_action_fnc: Callable[[TraderAction], MarketObservation]):
        return self.rng.choice(actions)
    
    def calculate_reward(self,
                         old_observation: MarketObservation,
                         new_observation: MarketObservation,
                         has_acted: bool,
                         environment_reward: Optional[float]):
        pass