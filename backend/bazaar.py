from uuid import UUID

from arelai.game import Game

from .market import Market, MarketObservation
from .trader import (
    Trader, TraderActionType, TraderAction,
    TradeAction, TakeAction, SellAction
)
from .goods import GoodType
from .coins import BonusType


class Bazaar(Game):
    """
    A class representing a turn-based trading game based on the Jaipur mechanics.

    Inherits from:
    ------------
    Game : Base game class with player-state setup.

    Attributes
    ----------
    players : dict[UUID, Trader]
        A dictionary mapping player UUIDs to Trader instances.
    state : Market
        The current game state represented by a Market instance.
    """

    def __init__(self,
                 players: dict[UUID, Trader],
                 state: Market,
                 max_rounds: int = 500):
        """
        Initialize the Bazaar game with players and an initial state.

        Parameters
        ----------
        players : dict[UUID, Trader]
            Mapping from UUIDs to Trader objects.
        state : Market
            The initial state of the market.
        """
        super().__init__(players, state)
        self.max_rounds = max_rounds

    def terminal(self, state: Market) -> bool:
        """
        Determine whether the game has reached a terminal (end) state.

        The game ends when:
        1. Maximum rounds exceeded, OR
        2. Three or more good types have empty coin stacks, OR
        3. The deck is empty AND the current player has no valid actions

        Parameters
        ----------
        state : Market
            The current game state.

        Returns
        -------
        bool
            True if the game is over, False otherwise.
        """
        if self.round > self.max_rounds:
            return True

        # Check if 3 or more goods have empty coin stacks
        empty_goods_coin_stacks = sum(
            len(state.coins.goods_coins[good]) == 0
            for good in GoodType if good != GoodType.CAMEL
        )
        if empty_goods_coin_stacks >= 3:
            return True

        # If deck is empty, check if current player has any valid actions
        if len(state.reserved_goods) == 0:
            # Get all legal actions for the current actor
            legal_actions = self.all_actions(state.actor, state)
            # Game ends only if there are no valid actions
            return len(legal_actions) == 0

        return False

    def observe(self, observer: Trader, state: Market) -> MarketObservation:
        """
        Generate a partial observation of the market for the given trader.

        Parameters
        ----------
        observer : Trader
            The observing trader.
        state : Market
            The current state of the market.

        Returns
        -------
        MarketObservation
            The observable view of the market for the observer.
        """
        trader_goods = state.player_goods[observer]
        trader_goods_coins = state.player_coins[observer].goods_coins

        trader_bonus_counts = {
            bonus: len(state.player_coins[observer].bonus_coins[bonus])
            for bonus in BonusType
        }

        global_bonus_counts = {
            bonus: len(state.coins.bonus_coins[bonus])
            for bonus in BonusType
        }

        return MarketObservation(
            observer,
            state.actor,
            state.action,
            trader_goods,
            trader_goods_coins,
            trader_bonus_counts,
            state.goods,
            state.coins.goods_coins,
            global_bonus_counts,
            len(state.reserved_goods),
            state.max_player_goods_count,
            state.max_goods_count
        )

    def all_actions(self, actor: Trader, state: Market) -> list[TraderAction]:
        """
        Get all possible actions available to a trader.

        Parameters
        ----------
        actor : Trader
            The trader taking the action.
        state : Market
            The current market state.

        Returns
        -------
        list of TraderAction
            A list of all legal actions for the given actor.
        """
        obs = self.observe(actor, state)
        return (
            TradeAction.all_actions(obs) +
            SellAction.all_actions(obs) +
            TakeAction.all_actions(obs)
        )

    def apply_action(self, state: Market, action: TraderAction) -> Market:
        """
        Apply the selected action to the game state and return the new state.

        Parameters
        ----------
        state : Market
            The current market state.
        action: TraderAction
            The current actor's action.

        Returns
        -------
        Market
            The new market state after applying the action.
        """
        new_state = state.clone()
        actor = action.actor

        for good in action.requested_goods.to_list():
            new_state.player_goods[actor].add(good)
            new_state.goods.remove(good)

        for good in action.offered_goods.to_list():
            new_state.player_goods[actor].remove(good)
            if action.trader_action_type == TraderActionType.SELL:
                new_state.sold_goods.append(good)
            else:
                new_state.goods.add(good)

        if action.trader_action_type == TraderActionType.SELL:
            for _ in range(action._count):
                coin = new_state.coins.pop_goods_coin(action._sell)
                new_state.player_coins[actor].add_goods_coin(action._sell, coin)

            if action._count in BonusType._value2member_map_:
                bonus_type = BonusType(action._count)
                bonus_coin = new_state.coins.pop_bonus_coin(bonus_type)
                new_state.player_coins[actor].add_bonus_coin(bonus_type, bonus_coin)

        new_state.actor = self.state.get_non_actor()
        new_state.action = action
        new_state.refill_market()
        return new_state

    def calculate_reward(
        self,
        player: Trader,
        old_state: Market,
        new_state: Market
    ) -> float:
        """
        Compute the reward for a player based on state transitions.

        Parameters
        ----------
        player : Trader
            The trader receiving the reward.
        old_state : Market
            The previous market state.
        new_state : Market
            The market state after applying the action.

        Returns
        -------
        float
            Reward for the player (points earned), 0 if game not over.
        """
        if not self.terminal(new_state):
            return 0

        total = 0
        for good in GoodType:
            total += sum(new_state.player_coins[player].goods_coins[good])
        for bonus in BonusType:
            total += sum(new_state.player_coins[player].bonus_coins[bonus])
        return total

    def output(self):
        """
        Pretty-print the current game state using the `rich` library.
        """
        from rich.console import Console
        from rich.panel import Panel
        from rich.columns import Columns
        from rich.text import Text
        from rich.rule import Rule

        console = Console()

        def format_goods(goods, width=3, include_camels=True):
            """
            Format goods dictionary into string for display.

            Parameters
            ----------
            goods : dict[GoodType, int]
                Mapping from good types to counts.
            width : int
                Width per good item.
            include_camels : bool
                Whether to include camels in output.

            Returns
            -------
            str
                A formatted string of goods.
            """
            return "".join(
                (g.value.ljust(width)) * goods[g]
                for g in GoodType
                if include_camels or g != GoodType.CAMEL
            )

        def make_action_panel():
            action_text = Text()

            action = self.state.action

            if not action:
                return Panel(
                "None",
                title=Text("Action", style="magenta bold"),
                border_style="magenta",
                padding=(1, 2),
                width=70
            )

            actor = self.old_state.actor
            
            # Action details
            action_text.append(f"By: {actor.name}\n", style="bold")
            action_text.append(f"Type: {action.trader_action_type.value}\n", style="bold")

            # Offered goods
            action_text.append("\nOffered Goods:\n", style="bold")
            offered_goods = format_goods(action.offered_goods)
            action_text.append(offered_goods + "\n")

            # Requested goods
            action_text.append("\nRequested Goods:\n", style="bold")
            requested_goods = format_goods(action.requested_goods)
            action_text.append(requested_goods + "\n")

            return Panel(
                action_text,
                title=Text("Action", style="magenta bold"),
                border_style="magenta",
                padding=(1, 2),
                width=70
            )
        
        def make_market_panel():
            text = Text()
            text.append("Current Goods:\n", style="bold")
            text.append(format_goods(self.state.goods) + "\n")
            text.append(f"\nRemaining Goods: {len(self.state.reserved_goods)}\n")

            text.append("\nCoins:\n", style="bold")
            for good in GoodType:
                if good != GoodType.CAMEL:
                    coins = self.state.coins.goods_coins.get(good, [])
                    text.append(f"{good.value.ljust(3)}: {coins}\n")

            text.append("\nCount Bonus Coins:\n", style="bold")
            text.append("  ".join(
                f"{b.value}: {len(self.state.coins.bonus_coins.get(b, []))}"
                for b in BonusType
            ))

            return Panel(text, title=Text("Market", style="yellow bold"),
                        border_style="yellow", padding=(1, 2), width=70)


        def make_player_panel(player, is_terminal):
            if not player:
                return Panel(f"Player {player.name} not found", title="Error", border_style="red")

            goods = self.state.player_goods[player]
            goods_coins = self.state.player_coins[player].goods_coins
            bonus_coins = self.state.player_coins[player].bonus_coins

            body = Text()
            body.append(f"Goods:\n{format_goods(goods, include_camels=False)}\n")
            body.append(f"Count{GoodType.CAMEL.value}: {goods[GoodType.CAMEL]}\n")

            goods_value = sum(sum(coins) for coins in goods_coins.values())
            bonus_value = sum(sum(coins) for coins in bonus_coins.values())

            if is_terminal:
                camel_bonus = self.state.camel_bonus if (
                    goods[GoodType.CAMEL] > self.state.player_goods[self.state.get_non_actor()][GoodType.CAMEL]
                ) else 0
                total = goods_value + bonus_value + camel_bonus
                body.append(f"Value All Coins: {total}\n")
            else:
                body.append(f"Value Goods Coins: {goods_value}\n\n")
                body.append("Count Bonus Coins:\n", style="bold")
                body.append("  ".join(
                    f"{b.value}: {len(bonus_coins.get(b, []))}"
                    for b in BonusType
                ))

            return Panel(body, title=Text(player.name, style="cyan bold"),
                         border_style="cyan", padding=(1, 2), width=40)

        is_terminal = self.terminal(self.state)
        console.print(Rule(title=f"Round {self.round}"))
        console.print(make_action_panel())
        console.print(make_market_panel())
        console.print()
        console.print(Columns([
            make_player_panel(player_id, False)
            for player_id in self.players
        ]))
        if is_terminal:
            console.print(Rule(title=f"Results"))
            console.print(Columns([
            make_player_panel(player_id, True)
            for player_id in self.players
        ]))

class BasicBazaar(Bazaar):
    def __init__(self, seed, players):

        reserved_goods = (
            [GoodType.CAMEL] * 11 +
            [GoodType.LEATHER] * 10 + 
            [GoodType.SPICE] * 8 +
            [GoodType.FABRIC] * 8 +
            [GoodType.SILVER] * 6 +
            [GoodType.GOLD] * 6 +
            [GoodType.DIAMOND] * 6
        )

        goods_coins = {
            GoodType.DIAMOND: [5, 5, 5, 7, 7],
            GoodType.GOLD: [5, 5, 5, 6, 6], 
            GoodType.SILVER: [5, 5, 5, 5, 5],
            GoodType.FABRIC: [1, 1, 2, 2, 3, 3, 5],
            GoodType.SPICE: [1, 1, 2, 2, 3, 3, 5],
            GoodType.LEATHER: [4, 3, 2, 1, 1, 1, 1, 1, 1],
            GoodType.CAMEL: []
        }

        bonus_coins = {
            BonusType.THREE: [1, 1, 2, 2, 2, 3, 3],
            BonusType.FOUR: [6, 6, 4, 4, 5, 5], 
            BonusType.FIVE: [8, 8, 9, 10, 10],
        }

        camel_bonus = 5
        max_size = 5
        max_trader_size = 7
        initial_trader_size = 5

        initial_state = Market(
            seed=seed,
            players = players,
            actor = players[0],
            action = None,
            reserved_goods = reserved_goods,
            goods_coins = goods_coins,
            bonus_coins = bonus_coins,
            camel_bonus = camel_bonus,
            max_goods_count = max_size,
            max_player_goods_count = max_trader_size,
            initial_player_goods_count = initial_trader_size)
        
        super().__init__(players, initial_state)