from __future__ import annotations
from typing import TYPE_CHECKING

from arelai.game import State, Observation

from .goods import GoodType, Goods
from .coins import BonusType, Coins

if TYPE_CHECKING:
    from .trader import Trader, TraderAction

import random
from uuid import UUID




class Market(State):
   
    def __init__(
        self,
        seed,
        players: list[Trader],
        actor: Trader,
        action: TraderAction,
        reserved_goods: list[GoodType],
        goods_coins: dict[GoodType, list],
        bonus_coins: dict[BonusType, list],
        camel_bonus: int,
        max_goods_count: int,
        max_player_goods_count: int,
        initial_player_goods_count: int,
    ):
        
        super().__init__(
            actor=actor,
            action=action)

        # use this for random operations
        self.rng = random.Random(seed)

        self.players = players

        self.reserved_goods = reserved_goods
        self.rng.shuffle(self.reserved_goods)

        self.coins = Coins()
        for good_type in goods_coins.keys():
            for coin in goods_coins[good_type]:
                self.coins.add_goods_coin(good_type, coin)
        for bonus_type in bonus_coins.keys():
            for coin in bonus_coins[bonus_type]:
                self.coins.add_bonus_coin(bonus_type, coin)

        self.player_goods = {}
        self.player_coins = {}
        
        for player in self.players:
            self.player_coins[player] = Coins()
            self.player_goods[player] = Goods()

        self.max_player_goods_count = max_player_goods_count
        self.initial_player_goods_count = initial_player_goods_count
        
        # give each player some goods
        for _ in range(initial_player_goods_count):
            for player in self.players:
                good_type = self.reserved_goods.pop()
                self.player_goods[player].add(good_type)

        self.max_goods_count = max_goods_count
        self.goods = Goods()
        
        # Place exactly 3 camels in the initial market
        camels_to_place = 3
        i = len(self.reserved_goods) - 1
        while camels_to_place > 0 and i >= 0:
            if self.reserved_goods[i] == GoodType.CAMEL:
                self.goods.add(self.reserved_goods.pop(i))
                camels_to_place -= 1
            i -= 1
        
        # Fill the rest of the market
        self.refill_market()

        self.sold_goods = []

        self.camel_bonus = camel_bonus
        self.max_goods_count = max_goods_count
    
    def refill_market(self):
        while self.goods.count() < self.max_goods_count and self.reserved_goods:
            good_type = self.reserved_goods.pop()
            self.goods.add(good_type)

    def get_non_actor(self):
        non_actor = [player for player in self.players if player != self.actor][0]
        return non_actor


class MarketObservation(Observation):
    def __init__(self,
            observer_id: UUID,
            actor_id: UUID,
            action: TraderAction,
            actor_goods: Goods,
            actor_goods_coins: dict[GoodType, list[int]],
            actor_bonus_coins_counts: dict[BonusType, int],
            market_goods: Goods,
            market_goods_coins: dict[GoodType, list[int]],
            market_bonus_coins_counts: dict[BonusType, int],
            market_reserved_goods_count: int,
            max_player_goods_count: int,
            max_market_goods_count: int
    ):
        self.actor = actor_id
        self.action = action
        self.actor_goods = actor_goods
        self.actor_goods_coins = actor_goods_coins
        self.actor_bonus_coins_counts = actor_bonus_coins_counts
        self.market_goods = market_goods
        self.market_goods_coins = market_goods_coins
        self.market_bonus_coins_counts = market_bonus_coins_counts
        self.market_reserved_goods_count = market_reserved_goods_count
        self.max_player_goods_count = max_player_goods_count
        self.max_market_goods_count = max_market_goods_count

        self.actor_non_camel_goods_count = self.actor_goods.count(include_camels=False)

        super().__init__(observer_id)
    
