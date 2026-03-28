from enum import Enum
from .goods import GoodType

class BonusType(Enum):
    THREE = 3
    FOUR = 4
    FIVE = 5

class Coins:

    def __init__(self):
        self._goods_coins = {
            GoodType.CAMEL: [],
            GoodType.LEATHER: [],
            GoodType.SPICE: [],
            GoodType.FABRIC: [],
            GoodType.SILVER: [],
            GoodType.GOLD: [],
            GoodType.DIAMOND: []
        }
        self._bonus_coins = {
            BonusType.THREE: [],
            BonusType.FOUR: [],
            BonusType.FIVE: []
        }

    def add_goods_coin(self, good_type: GoodType, value: int):
        if value:
            self._goods_coins[good_type].append(value)
            self._goods_coins[good_type].sort()

    def pop_goods_coin(self, good_type: GoodType):
        if self._goods_coins[good_type]:
            return self._goods_coins[good_type].pop()
        return None

    def add_bonus_coin(self, bonus_type: BonusType, value: int):
        if value:
            self._bonus_coins[bonus_type].append(value)
            self._bonus_coins[bonus_type].sort()

    def pop_bonus_coin(self, bonus_type: BonusType):
        if self._bonus_coins[bonus_type]:
            return self._bonus_coins[bonus_type].pop()
        return None
    
    @property
    def goods_coins(self):
        return self._goods_coins
    
    @property
    def bonus_coins(self):
        return self._bonus_coins