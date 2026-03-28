from enum import Enum

class GoodType(Enum):
    DIAMOND = "💎"
    GOLD = "🪙"
    SILVER = "🪨"
    FABRIC = "👚"
    SPICE = "🫑"
    LEATHER = "👞"
    CAMEL = "🐪"


class Goods:

    def __init__(self):

        self._goods = {
            GoodType.CAMEL: 0,
            GoodType.LEATHER: 0,
            GoodType.SPICE: 0,
            GoodType.FABRIC: 0,
            GoodType.SILVER: 0,
            GoodType.GOLD: 0,
            GoodType.DIAMOND: 0
        }
    
    def __getitem__(self, gt: GoodType):
        return self._goods[gt]
    
    def add(self, good_type):
        self._goods[good_type] += 1

    def remove(self, good_type):
        if self._goods[good_type] > 0:
            self._goods[good_type] -= 1

    def count(self, include_camels=True):
        cnt = 0
        for good_type in GoodType:
            cnt += self._goods.get(good_type, 0)
        
        if not include_camels:
            cnt -= self._goods.get(GoodType.CAMEL, 0)

        return cnt
    
    def to_list(self) -> list[GoodType]:
        lst = []
        for good_type in GoodType:
            for _ in range(self._goods[good_type]):
                lst.append(good_type)
        return lst

    @staticmethod
    def from_list(lst: list[GoodType]) -> 'Goods':
        goods = Goods()
        for good_type in lst:
            goods.add(good_type)
        return goods
    
    @staticmethod
    def from_dict(dct: dict[GoodType, int]) -> 'Goods':
        goods = Goods()
        for good_type in GoodType:
            goods._goods[good_type] = dct.get(good_type, 0)
        return goods
